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
        # ── Soft Prompt 参数 ──────────────────────────────
        num_soft_prompts: int = 8,   # k: soft token 数量，建议 50-100 对齐 DELRec
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

        # ── Soft Prompts ──────────────────────────────────
        # soft_prompts     : 阶段一 TA+RPS 两任务共享同一组，联合优化
        #                    对齐 DELRec：两任务梯度累积在同一组参数上
        # soft_prompts_lsr : 阶段二专用，阶段一结束后从 soft_prompts 热启动后冻结
        #                    对齐 DELRec 阶段二 load_state_dict 机制
        d_llm = self.llm_model.config.hidden_size
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

        # 阶段一共享软提示（TA + RPS 联合优化同一组参数，对齐 DELRec）
        self.soft_prompts = nn.Parameter(
            torch.normal(emb_mean, emb_std, size=(num_soft_prompts, d_llm))
        )  # shape: [k, d_llm]

        # 阶段二软提示（阶段一结束后从 soft_prompts 热启动，冻结不更新）
        self.soft_prompts_lsr = nn.Parameter(
            torch.normal(emb_mean, emb_std, size=(num_soft_prompts, d_llm)),
            requires_grad=False
        )  # shape: [k, d_llm]

        # ── CLS tokens（原有逻辑保持不变）────────────────
        if not args.token:
            if args.nn_parameter:
                self.CLS      = nn.Parameter(torch.normal(0, 1, size=(1, d_llm)))
                self.CLS_item = nn.Parameter(torch.normal(0, 1, size=(1, d_llm)))
            else:
                self.CLS = nn.Embedding(1, d_llm)
                nn.init.normal_(self.CLS.weight, mean=emb_mean, std=emb_std)
                self.CLS_item = nn.Embedding(1, d_llm)
                nn.init.normal_(self.CLS_item.weight, mean=emb_mean, std=emb_std)

        # ── 用户 / 物品预测头 ─────────────────────────────
        # pred_user        : Stage 2 主训练用
        # pred_user_ta     : Stage 1 TA 任务专用，梯度路径与 RPS 完全隔离
        # pred_user_rps    : Stage 1 RPS 任务专用，梯度路径与 TA 完全隔离
        self.pred_user = nn.Sequential(
            nn.Linear(d_llm, 2048), nn.LayerNorm(2048),
            nn.LeakyReLU(), nn.Linear(2048, 128))
        nn.init.xavier_normal_(self.pred_user[0].weight)
        nn.init.xavier_normal_(self.pred_user[3].weight)

        self.pred_user_ta = nn.Sequential(
            nn.Linear(d_llm, 2048), nn.LayerNorm(2048),
            nn.LeakyReLU(), nn.Linear(2048, 128))
        nn.init.xavier_normal_(self.pred_user_ta[0].weight)
        nn.init.xavier_normal_(self.pred_user_ta[3].weight)

        self.pred_user_rps = nn.Sequential(
            nn.Linear(d_llm, 2048), nn.LayerNorm(2048),
            nn.LeakyReLU(), nn.Linear(2048, 128))
        nn.init.xavier_normal_(self.pred_user_rps[0].weight)
        nn.init.xavier_normal_(self.pred_user_rps[3].weight)

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

        self.mse = nn.MSELoss()
        self.max_output_txt_len = max_output_txt_len

    # ─────────────────────────────────────────────────────────────
    # _prepend_soft_prompts
    #   task='stage1' → 使用 soft_prompts（阶段一 TA+RPS 共享）
    #   task='stage2' → 使用 soft_prompts_lsr（阶段二专用，冻结）
    # ─────────────────────────────────────────────────────────────
    def _prepend_soft_prompts(self, inputs_embeds, attention_mask=None, task='stage1'):
        """
        inputs_embeds : [B, T, d]
        attention_mask: [B, T] or None
        task          : 'stage1' | 'stage2'
        返回:
            new_embeds : [B, k+T, d]
            new_mask   : [B, k+T] or None
        """
        B = inputs_embeds.shape[0]
        k = self.num_soft_prompts

        if task == 'stage2':
            sp = self.soft_prompts_lsr   # 冻结，阶段一热启动
        else:
            sp = self.soft_prompts       # 阶段一 TA+RPS 共享

        # [k, d] → [1, k, d] → [B, k, d]，cast 到相同 dtype
        sp = sp.unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
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
            token_idx.append(idx_tensor + offset)
        return token_idx

    # ─────────────────────────────────────────────────────────────
    # forward（Stage 2 走 train_mode0，task='stage2' 使用 soft_prompts_lsr）
    # ─────────────────────────────────────────────────────────────
    def forward(self, samples, mode=0):
        if mode == 0:
            return self.train_mode0(samples)
        elif mode == 1:
            raise NotImplementedError("train_mode1 is not implemented")

    def train_mode0(self, samples):
        """
        Stage 2 主训练 forward。
        soft_prompts（阶段一共享）已冻结，soft_prompts_lsr（阶段二专用）已冻结。
        Stage 2 使用 task='stage2'（soft_prompts_lsr，由 soft_prompts 热启动后冻结），
        对齐 DELRec 阶段二设计。
        """
        max_input_length = 1024 + self.num_soft_prompts
        k = self.num_soft_prompts

        log_emb    = samples['log_emb']
        llm_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt", padding="longest",
            truncation=True, max_length=max_input_length,
        ).to(self.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])

        inputs_embeds = self.replace_out_token_all(
            llm_tokens, inputs_embeds,
            token=['[UserOut]', '[HistoryEmb]'],
            embs={'[HistoryEmb]': samples['interact']})

        # Stage 2 使用 soft_prompts_lsr（由 soft_prompts 热启动后冻结，对齐 DELRec）
        inputs_embeds, new_mask = self._prepend_soft_prompts(
            inputs_embeds, llm_tokens.get('attention_mask'), task='stage2')

        # ── 候选分支：分 chunk 推理，显存峰值与 candidate_num 解耦 ─────────────
        # 候选 forward 不需要梯度（item embedding 只用于点积打分），
        # 按 candi_chunk_size 切片逐批过 LLM，最后 cat 成 [B*C, 128]。
        # 这样峰值显存只取决于 chunk 大小，而非 B*C 总量。
        candi_chunk_size = getattr(self.args, 'candi_chunk_size', 4)
        all_candidates   = samples['candidates_pos']      # list, len = B*C
        all_candi_embs   = samples['candidate_embs']      # [B*C, rec_dim]
        item_outputs_list = []

        # 候选分支：LLM forward 在 no_grad 内（节省显存），
        # pred_item 移到 no_grad 外面，保证其参数能收到梯度。
        # 注意：item_hidden 用 detach() 截断 LLM 侧梯度，
        # 梯度只流经 pred_item，不回传到 LLM，这与 Stage1 的 item 编码器设计一致。
        item_hidden_list = []
        with torch.no_grad():
            for chunk_start in range(0, len(all_candidates), candi_chunk_size):
                chunk_end   = chunk_start + candi_chunk_size
                chunk_texts = all_candidates[chunk_start:chunk_end]
                chunk_embs  = all_candi_embs[chunk_start:chunk_end]

                c_tokens = self.llm_tokenizer(
                    chunk_texts,
                    return_tensors="pt", padding="longest",
                    truncation=True, max_length=max_input_length,
                ).to(self.device)
                c_embeds = self.llm_model.get_input_embeddings()(c_tokens['input_ids'])
                c_embeds = self.replace_out_token_all_infer(
                    c_tokens, c_embeds,
                    token=['[ItemOut]', '[HistoryEmb]'],
                    embs={'[HistoryEmb]': chunk_embs})
                c_embeds, c_mask = self._prepend_soft_prompts(
                    c_embeds, c_tokens.get('attention_mask'), task='stage2')

                with torch.amp.autocast('cuda'):
                    c_out = self.llm_model.forward(
                        inputs_embeds=c_embeds,
                        attention_mask=c_mask,
                        output_hidden_states=True)
                    c_indx = self.get_embeddings(c_tokens, '[ItemOut]', offset=k)
                    chunk_hidden = torch.cat([
                        c_out.hidden_states[-1][i, c_indx[i]].mean(axis=0).unsqueeze(0)
                        for i in range(len(c_indx))])

                # 只保存 LLM 隐藏态，detach 截断 LLM 梯度
                item_hidden_list.append(chunk_hidden.detach().float())
                del c_out, c_embeds, c_mask, chunk_hidden

        # pred_item 在 no_grad 外：参数可以正常收到来自 lsr_loss 的梯度
        item_hidden = torch.cat(item_hidden_list, dim=0)   # [B*C, d_llm]
        with torch.amp.autocast('cuda'):
            item_outputs = self.pred_item(item_hidden)     # [B*C, 128]，有梯度

        # ── 用户分支：需要梯度，正常 forward ────────────────────────────────────
        with torch.amp.autocast('cuda'):
            outputs = self.llm_model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=new_mask,
                output_hidden_states=True)
            indx = self.get_embeddings(llm_tokens, '[UserOut]', offset=k)
            user_hidden = torch.cat([
                outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                for i in range(len(indx))])

        user_outputs = self.pred_user(user_hidden.float())  # [B, 128]，有梯度

        # ── lsr_loss：候选集分类，对齐 DELRec 原版 Stage 2 ───────────────────
        # rec_loss 与 lsr_loss 本质相同（同样的 user/item，同样的点积 CE），
        # 保留 lsr_loss 作为唯一的排序监督信号，避免梯度重复叠加。
        # 正样本 index=0（make_candidate_text 中 target 固定排第一位）。
        B = user_outputs.shape[0]
        candidate_num   = item_outputs.shape[0] // B
        item_outputs_3d = item_outputs.view(B, candidate_num, -1)      # [B, C, 128]
        item_outputs_3d = item_outputs_3d.to(user_outputs.dtype)
        lsr_logits      = torch.bmm(item_outputs_3d,
                                    user_outputs.unsqueeze(2)).squeeze(2)  # [B, C]
        lsr_labels = torch.zeros(B, dtype=torch.long, device=lsr_logits.device)
        lsr_loss   = F.cross_entropy(lsr_logits, lsr_labels)

        # ── match_loss：对齐 SASRec log_emb（结果蒸馏）───────────────────────
        log_emb = self.pred_user_CF2(log_emb)
        user_outputs_n = F.normalize(user_outputs, p=2, dim=1)
        log_emb_n      = F.normalize(log_emb,      p=2, dim=1)
        match_loss  = self.mse(user_outputs_n, log_emb_n)
        match_loss += (self.uniformity(user_outputs_n) + self.uniformity(log_emb_n))

        # ── 两路 loss 合并 ────────────────────────────────────────────────────
        # lsr_loss  : 候选集分类，覆盖排序监督（替代原 rec_loss，避免梯度重复）
        # match_loss: MSE 对齐 SASRec log_emb + uniformity 防退化（结果蒸馏）
        lsr_weight = getattr(self.args, 'lsr_loss_weight', 0.5)
        loss = lsr_weight * lsr_loss + match_loss
        return loss, lsr_loss.item(), match_loss.item(), user_outputs