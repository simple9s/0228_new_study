import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


class llm4rec(nn.Module):
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
        args=None,
        # ── Soft Prompt 新增参数 ──────────────────────────
        num_soft_prompts: int = 8,   # k: soft token 数量，建议 5-10
    ):
        super().__init__()
        self.device = device
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.args = args
        self.num_soft_prompts = num_soft_prompts

        if llm_model == 'llama':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif llm_model == 'llama-3b':
            model_id = "meta-llama/Llama-3.2-3B-Instruct"
        else:
            raise Exception(f'{llm_model} is not supported')

        print("\n=========")
        if self.args.nn_parameter:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=self.device, torch_dtype=torch.float16)
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map=self.device,
                torch_dtype=torch.float16, load_in_8bit=True)

        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[UserOut]', '[ItemOut]']})
        self.llm_tokenizer.add_special_tokens({'cls_token': "[CLS]"})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        self.llm_model = prepare_model_for_kbit_training(self.llm_model)

        # ── 冻结 LLM 参数 ────────────────────────────────
        for _, param in self.llm_model.named_parameters():
            if args.token:
                param.requires_grad = ('token' in _)
            else:
                param.requires_grad = False

        # ── Soft Prompts: 唯一可学习的 LLM 侧参数 ────────
        d_llm = self.llm_model.config.hidden_size
        # 初始化为 LLM embedding 层均值±std，让软提示"有意义地"开始
        with torch.no_grad():
            emb_weight = self.llm_model.model.embed_tokens.weight
            emb_mean = emb_weight.mean().item()
            emb_std  = emb_weight.std().item()
        if args.use_lora:
            lora_config = LoraConfig(
                r=getattr(args, 'lora_r', 8),
                lora_alpha=getattr(args, 'lora_alpha', 16),
                lora_dropout=getattr(args, 'lora_dropout', 0.05),
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)
            self.llm_model.print_trainable_parameters()
        self.soft_prompts = nn.Parameter(
            torch.normal(emb_mean, emb_std, size=(num_soft_prompts, d_llm))
        )  # shape: [k, d_llm]，requires_grad=True 默认

        # ── CLS tokens (原有逻辑保持不变) ────────────────
        if not args.token:
            if args.nn_parameter:
                self.CLS      = nn.Parameter(torch.normal(0, 1, size=(1, d_llm))).to(device)
                self.CLS_item = nn.Parameter(torch.normal(0, 1, size=(1, d_llm))).to(device)
            else:
                self.CLS = nn.Embedding(1, d_llm).to(device)
                nn.init.normal_(self.CLS.weight,
                                 mean=emb_mean, std=emb_std)
                self.CLS_item = nn.Embedding(1, d_llm).to(device)
                nn.init.normal_(self.CLS_item.weight,
                                 mean=emb_mean, std=emb_std)

        # ── 用户 / 物品预测头 ─────────────────────────────
        self.pred_user = nn.Sequential(
            nn.Linear(d_llm, 2048), nn.LayerNorm(2048),
            nn.LeakyReLU(), nn.Linear(2048, 128))
        nn.init.xavier_normal_(self.pred_user[0].weight)
        nn.init.xavier_normal_(self.pred_user[3].weight)

        self.pred_item = nn.Sequential(
            nn.Linear(d_llm, 2048), nn.LayerNorm(2048),
            nn.LeakyReLU(), nn.Linear(2048, 128))
        nn.init.xavier_normal_(self.pred_item[0].weight)
        nn.init.xavier_normal_(self.pred_item[3].weight)

        self.pred_user_CF2 = nn.Sequential(
            nn.Linear(64, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Linear(128, 128))
        nn.init.xavier_normal_(self.pred_user_CF2[0].weight)
        nn.init.xavier_normal_(self.pred_user_CF2[3].weight)

        self.cf_to_latent2 = nn.Sequential(
            nn.Linear(64, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Linear(128, 128))
        nn.init.xavier_normal_(self.cf_to_latent2[0].weight)
        nn.init.xavier_normal_(self.cf_to_latent2[3].weight)

        # ── 辅助任务预测头 ────────────────────────────────
        # TA head: 给定上下文，从 user_output 预测下一个 item（128-d 空间）
        # RPS head: 将 user_output 映射到 CF 空间做对齐
        # 复用 pred_user / pred_user_CF2 即可，无需新增，见 seqllm_model.py

        self.mse = nn.MSELoss()
        self.max_output_txt_len = max_output_txt_len

    # ─────────────────────────────────────────────────────────────
    # 辅助：将 soft prompts 拼接到 embeds 前端，同时扩展 attention mask
    # ─────────────────────────────────────────────────────────────
    def _prepend_soft_prompts(self, inputs_embeds, attention_mask=None):
        """
        inputs_embeds : [B, T, d]
        attention_mask: [B, T]  or None
        返回:
            new_embeds    : [B, k+T, d]
            new_mask      : [B, k+T]  or None
        """
        B = inputs_embeds.shape[0]
        k = self.num_soft_prompts
        # soft_prompts: [k, d] → [1, k, d] → [B, k, d]
        sp = self.soft_prompts.unsqueeze(0).expand(B, -1, -1)
        # cast to same dtype
        sp = sp.to(inputs_embeds.dtype)
        new_embeds = torch.cat([sp, inputs_embeds], dim=1)  # [B, k+T, d]

        if attention_mask is not None:
            prefix_mask = torch.ones(B, k,
                                     dtype=attention_mask.dtype,
                                     device=attention_mask.device)
            new_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            new_mask = None

        return new_embeds, new_mask

    # ─────────────────────────────────────────────────────────────
    # 原有 helper 方法（保持不变）
    # ─────────────────────────────────────────────────────────────
    def info_nce_loss_batch(self, anchor, log_emb, temperature=0.07):
        B = anchor.shape[0]
        anchor  = F.normalize(anchor,  p=2, dim=1)
        log_emb = F.normalize(log_emb, p=2, dim=1)
        sim = torch.matmul(anchor, log_emb.T) / temperature
        mask = torch.eye(B, device=anchor.device).bool()
        logits = torch.cat([sim[mask].view(B, 1),
                            sim[~mask].view(B, -1)], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)

    def rec_loss(self, anchor, items, temperature=1.0):
        logits = torch.bmm(
            items.view(anchor.shape[0], -1, anchor.shape[1]),
            anchor.unsqueeze(2)).squeeze(2)
        logits = logits / temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def uniformity(self, x, p=2):
        return torch.pdist(x, p=p).pow(2).mul(-p).exp().mean()

    def replace_out_token_all(self, llm_tokens, inputs_embeds, token=[], embs=None):
        for t in token:
            token_id = self.llm_tokenizer(
                t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            for inx in range(len(llm_tokens["input_ids"])):
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]
                if 'Emb' in t:
                    ee = embs[t][inx]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat(
                            (user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                elif 'Rep' in t:
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat(
                            (user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                else:
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                cls_v = (self.CLS[torch.tensor([0]).to(self.device)]
                                         if self.args.nn_parameter
                                         else self.CLS(torch.tensor([0]).to(self.device)))
                                user_vector = torch.cat(
                                    (user_vector[:idx], cls_v, user_vector[idx+1:]), dim=0)
                            elif 'ItemOut' in t:
                                cls_v = (self.CLS_item[torch.tensor([0]).to(self.device)]
                                         if self.args.nn_parameter
                                         else self.CLS_item(torch.tensor([0]).to(self.device)))
                                user_vector = torch.cat(
                                    (user_vector[:idx], cls_v, user_vector[idx+1:]), dim=0)
                vectors.append(user_vector.unsqueeze(0))
            inputs_embeds = torch.cat(vectors)
        return inputs_embeds

    def replace_out_token_all_infer(self, llm_tokens, inputs_embeds,
                                    token=[], embs=None,
                                    user_act=False, item_act=False):
        for t in token:
            token_id = self.llm_tokenizer(
                t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            for inx in range(len(llm_tokens["input_ids"])):
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]
                if 'Emb' in t:
                    ee = [embs[t][inx]]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat(
                            (user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                elif 'Rep' in t:
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat(
                            (user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
                else:
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                cls_v = (self.CLS[torch.tensor([0]).to(self.device)]
                                         if self.args.nn_parameter
                                         else self.CLS(torch.tensor([0]).to(self.device)))
                                user_vector = torch.cat(
                                    (user_vector[:idx], cls_v, user_vector[idx+1:]), dim=0)
                            elif 'ItemOut' in t:
                                cls_v = (self.CLS_item[torch.tensor([0]).to(self.device)]
                                         if self.args.nn_parameter
                                         else self.CLS_item(torch.tensor([0]).to(self.device)))
                                user_vector = torch.cat(
                                    (user_vector[:idx], cls_v, user_vector[idx+1:]), dim=0)
                vectors.append(user_vector.unsqueeze(0))
            inputs_embeds = torch.cat(vectors)
        return inputs_embeds

    def get_embeddings(self, llm_tokens, token, offset=0):
        """
        offset: soft prompt 导致的位置偏移 (= num_soft_prompts)
        """
        token_idx = []
        token_id = self.llm_tokenizer(
            token, return_tensors="pt", add_special_tokens=False).input_ids.item()
        for inx in range(len(llm_tokens['input_ids'])):
            idx_tensor = (llm_tokens['input_ids'][inx] == token_id).nonzero().view(-1)
            token_idx.append(idx_tensor + offset)   # ← 加上偏移
        return token_idx

    # ─────────────────────────────────────────────────────────────
    # forward
    # ─────────────────────────────────────────────────────────────
    def forward(self, samples, mode=0):
        if mode == 0:
            return self.train_mode0(samples)
        elif mode == 1:
            return self.train_mode1(samples)

    def train_mode0(self, samples):
        """
        核心改动：
          1. 先做原有的 token 替换（HistoryEmb / UserOut / ItemOut）
          2. 再将 soft_prompts 拼接到序列最前方
          3. get_embeddings 时传 offset=num_soft_prompts
        """
        max_input_length = 1024
        k = self.num_soft_prompts

        log_emb    = samples['log_emb']
        llm_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt", padding="longest",
            truncation=True, max_length=max_input_length,
        ).to(self.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

        # 替换特殊 token 嵌入（与原版一致）
        inputs_embeds = self.replace_out_token_all(
            llm_tokens, inputs_embeds,
            token=['[UserOut]', '[HistoryEmb]'],
            embs={'[HistoryEmb]': samples['interact']})

        # ── ① 拼接 soft prompts ──────────────────────────
        inputs_embeds, new_mask = self._prepend_soft_prompts(
            inputs_embeds, llm_tokens.get('attention_mask'))

        # 候选物品分支
        candi_tokens = self.llm_tokenizer(
            samples['candidates_pos'],
            return_tensors="pt", padding="longest",
            truncation=True, max_length=max_input_length,
        ).to(self.device)
        candi_embeds = self.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
        candi_embeds = self.replace_out_token_all_infer(
            candi_tokens, candi_embeds,
            token=['[ItemOut]', '[HistoryEmb]'],
            embs={'[HistoryEmb]': samples['candidate_embs']})

        # ── ② 候选分支也拼接 soft prompts ───────────────
        candi_embeds, candi_mask = self._prepend_soft_prompts(
            candi_embeds, candi_tokens.get('attention_mask'))

        with torch.amp.autocast('cuda'):
            # 候选物品表示
            candi_outputs = self.llm_model.forward(
                inputs_embeds=candi_embeds,
                attention_mask=candi_mask,
                output_hidden_states=True)
            # ── ③ get_embeddings 加 offset ───────────────
            indx = self.get_embeddings(candi_tokens, '[ItemOut]', offset=k)
            item_outputs = torch.cat([
                candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                for i in range(len(indx))])

            # 用户表示
            outputs = self.llm_model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=new_mask,
                output_hidden_states=True)
            indx = self.get_embeddings(llm_tokens, '[UserOut]', offset=k)
            user_outputs = torch.cat([
                outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                for i in range(len(indx))])

        user_outputs = self.pred_user(user_outputs)
        item_outputs = self.pred_item(item_outputs)

        rec_loss = self.rec_loss(user_outputs, item_outputs)

        log_emb = self.pred_user_CF2(log_emb)

        user_outputs_n = F.normalize(user_outputs, p=2, dim=1)
        log_emb_n      = F.normalize(log_emb,      p=2, dim=1)

        match_loss = self.mse(user_outputs_n, log_emb_n)
        match_loss += (self.uniformity(user_outputs_n) + self.uniformity(log_emb_n))

        loss = rec_loss + match_loss
        return loss, rec_loss.item(), match_loss.item()