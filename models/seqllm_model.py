"""
seqllm_model.py  —— 在 LLM-SRec 基础上集成 Soft Prompt + 辅助任务
新增：
  TA  (Temporal Analysis) loss  : 用用户表示预测序列中倒数第二个 item
  RPS (Recommendation Pattern Simulating) loss: 让用户表示对齐 SASRec 的排序
差异化学习率：soft_prompts 用 1e-3，其余可训练参数用 1e-4
"""

import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.seqllm4rec import *          # 已修改版
from sentence_transformers import SentenceTransformer
from datetime import datetime

from tqdm import trange, tqdm

try:
    import habana_frameworks.torch.core as htcore
except Exception:
    pass


class llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device

        with open(f'./SeqRec/data_{args.rec_pre_trained_data}/text_name_dict.json.gz', 'rb') as ft:
            self.text_name_dict = pickle.load(ft)

        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)

        self.item_num    = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim   = 768

        self.mse = nn.MSELoss()
        self.l1  = nn.L1Loss()
        self.all_embs = None
        self.maxlen   = args.maxlen

        # ── 评测指标累计器 ────────────────────────────────
        self.NDCG = self.HIT = 0
        self.NDCG_20 = self.HIT_20 = 0
        self.NDCG_1  = self.HIT_1  = 0
        self.NDCG_5  = self.HIT_5  = 0
        self.rec_NDCG = self.rec_HIT = 0
        self.lan_NDCG = self.lan_HIT = 0
        self.num_user = self.yes = 0
        self.extract_embs_list = []
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

        # ── LLM（含 Soft Prompts）────────────────────────
        num_soft_prompts = getattr(args, 'num_soft_prompts', 8)
        self.llm = llm4rec(device=self.device, llm_model=args.llm,
                           args=self.args,
                           num_soft_prompts=num_soft_prompts)

        # ── Item embedding 投影层 ─────────────────────────
        d_llm = self.llm.llm_model.config.hidden_size
        self.item_emb_proj = nn.Sequential(
            nn.Linear(self.rec_sys_dim, d_llm),
            nn.LayerNorm(d_llm),
            nn.LeakyReLU(),
            nn.Linear(d_llm, d_llm))
        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        nn.init.xavier_normal_(self.item_emb_proj[3].weight)

        # ────────────────────────────────────────────────────
        # 辅助任务权重（可通过 args 覆盖）
        # ────────────────────────────────────────────────────
        self.ta_loss_weight  = getattr(args, 'ta_loss_weight',  0.3)
        self.rps_loss_weight = getattr(args, 'rps_loss_weight', 0.3)

        self.users = self.NDCG = self.HT = 0.0

    # ────────────────────────────────────────────────────────────
    # 差异化学习率优化器工厂
    # ────────────────────────────────────────────────────────────
    def build_optimizer(self):
        """
        soft_prompts → lr=1e-3
        其余可训练参数 → lr=1e-4
        """
        soft_prompt_params = [self.llm.soft_prompts]
        other_params = [p for n, p in self.named_parameters()
                        if p.requires_grad and p is not self.llm.soft_prompts]

        param_groups = [
            {'params': soft_prompt_params, 'lr': 1e-3},
            {'params': other_params,       'lr': self.args.stage2_lr},
        ]
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.98))
        return optimizer

    # ────────────────────────────────────────────────────────────
    # 辅助任务：TA Loss
    # 取序列中倒数第二个 item 作为 "context anchor"，
    # 用它的 CF embedding 与用户表示做对齐，
    # 相当于要求用户表示能感知到最近的交互 item。
    # ────────────────────────────────────────────────────────────
    def _compute_ta_loss(self, user_outputs, seq_batch):
        """
        user_outputs : [B, 128]  (经过 pred_user 的用户表示)
        seq_batch    : list/ndarray of shape [B, maxlen]，原始 item id 序列
        """
        ta_losses = []
        for i in range(len(seq_batch)):
            valid_ids = seq_batch[i][seq_batch[i] > 0]
            if len(valid_ids) < 2:
                continue
            # 倒数第二个 item 为 TA 的预测目标
            ta_target_id  = int(valid_ids[-2])
            ta_neg_ids    = []
            while len(ta_neg_ids) < 3:          # 3 个负样本
                neg = np.random.randint(1, self.item_num + 1)
                if neg != ta_target_id:
                    ta_neg_ids.append(neg)

            all_ids = [ta_target_id] + ta_neg_ids
            item_embs = self.item_emb_proj(          # [4, d_llm]
                self.get_item_emb(all_ids))
            item_embs = self.llm.pred_item(item_embs)   # [4, 128]
            # 相似度打分 [4]
            scores = torch.matmul(item_embs,
                                   user_outputs[i].unsqueeze(-1)).squeeze(-1)
            label  = torch.tensor(0, device=self.device)  # 正样本在第 0 位
            ta_losses.append(torch.nn.functional.cross_entropy(
                scores.unsqueeze(0), label.unsqueeze(0)))

        if len(ta_losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(ta_losses).mean()

    # ────────────────────────────────────────────────────────────
    # 辅助任务：RPS Loss
    # 让用户表示对齐 SASRec 的 top-1 推荐结果，
    # 即：user_output 与 SASRec 预测的 item 的相似度 > 随机负样本
    # ────────────────────────────────────────────────────────────
    def _compute_rps_loss(self, user_outputs, pos_batch, seq_batch):
        """
        user_outputs : [B, 128]
        pos_batch    : [B, maxlen]  正样本 item ids（pos[-1] 即 SASRec 预测目标）
        seq_batch    : [B, maxlen]  用于排除历史
        """
        rps_losses = []
        for i in range(len(pos_batch)):
            target_id = int(pos_batch[i][-1])
            if target_id == 0:
                continue
            neg_ids = []
            while len(neg_ids) < 3:
                neg = np.random.randint(1, self.item_num + 1)
                if neg != target_id and neg not in seq_batch[i]:
                    neg_ids.append(neg)

            all_ids   = [target_id] + neg_ids
            item_embs = self.item_emb_proj(self.get_item_emb(all_ids))
            item_embs = self.llm.pred_item(item_embs)          # [4, 128]

            scores = torch.matmul(item_embs,
                                   user_outputs[i].unsqueeze(-1)).squeeze(-1)
            label  = torch.tensor(0, device=self.device)
            rps_losses.append(torch.nn.functional.cross_entropy(
                scores.unsqueeze(0), label.unsqueeze(0)))

        if len(rps_losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(rps_losses).mean()

    # ────────────────────────────────────────────────────────────
    # 保存 / 加载（新增 soft_prompts）
    # ────────────────────────────────────────────────────────────
    def save_model(self, args, epoch2=None, best=False):
        out_dir = f'./models/{args.save_dir}/'
        if best:
            out_dir = out_dir[:-1] + 'best/'
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_{args.llm}_{epoch2}_'

        if args.train:
            torch.save(self.item_emb_proj.state_dict(),      out_dir + 'item_proj.pt')
            torch.save(self.llm.pred_user.state_dict(),      out_dir + 'pred_user.pt')
            torch.save(self.llm.pred_item.state_dict(),      out_dir + 'pred_item.pt')
            # ── 保存 soft prompts ──────────────────────────
            torch.save(self.llm.soft_prompts.data,           out_dir + 'soft_prompts.pt')
            if not args.token:
                if args.nn_parameter:
                    torch.save(self.llm.CLS.data,            out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.data,       out_dir + 'CLS_item.pt')
                else:
                    torch.save(self.llm.CLS.state_dict(),    out_dir + 'CLS.pt')
                    torch.save(self.llm.CLS_item.state_dict(), out_dir + 'CLS_item.pt')
            if args.token:
                torch.save(self.llm.llm_model.model.embed_tokens.state_dict(),
                           out_dir + 'token.pt')

    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = (f'./models/{args.save_dir}/'
                   f'{args.rec_pre_trained_data}_{args.llm}_{phase2_epoch}_')

        self.item_emb_proj.load_state_dict(
            torch.load(out_dir + 'item_proj.pt', map_location=self.device))
        self.llm.pred_user.load_state_dict(
            torch.load(out_dir + 'pred_user.pt', map_location=self.device))
        self.llm.pred_item.load_state_dict(
            torch.load(out_dir + 'pred_item.pt', map_location=self.device))

        # ── 加载 soft prompts（向后兼容：文件不存在则跳过）──
        sp_path = out_dir + 'soft_prompts.pt'
        if os.path.exists(sp_path):
            self.llm.soft_prompts.data = torch.load(sp_path, map_location=self.device)

        if not args.token:
            if args.nn_parameter:
                self.llm.CLS.data = torch.load(
                    out_dir + 'CLS.pt', map_location=self.device)
                self.llm.CLS_item.data = torch.load(
                    out_dir + 'CLS_item.pt', map_location=self.device)
            else:
                self.llm.CLS.load_state_dict(
                    torch.load(out_dir + 'CLS.pt', map_location=self.device))
                self.llm.CLS_item.load_state_dict(
                    torch.load(out_dir + 'CLS_item.pt', map_location=self.device))
        if args.token:
            self.llm.llm_model.model.embed_tokens.load_state_dict(
                torch.load(out_dir + 'token.pt', map_location=self.device))

    # ────────────────────────────────────────────────────────────
    # 原有方法（保持不变）
    # ────────────────────────────────────────────────────────────
    def find_item_text(self, item, title_flag=True, description_flag=True):
        t, d = 'title', 'description'
        t_, d_ = 'No Title', 'No Description'
        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        else:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]

    def find_item_time(self, item, user, title_flag=True, description_flag=True):
        l = [datetime.utcfromtimestamp(
             int(self.text_name_dict['time'][i][user]) / 1000) for i in item]
        return [l_.strftime('%Y-%m-%d') for l_ in l]

    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t, d = 'title', 'description'
        t_, d_ = 'No Title', 'No Description'
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        else:
            return f'"{self.text_name_dict[d].get(item,d_)}"'

    def get_item_emb(self, item_ids):
        with torch.no_grad():
            if self.args.nn_parameter:
                return self.recsys.model.item_emb[
                    torch.LongTensor(item_ids).to(self.device)]
            else:
                return self.recsys.model.item_emb(
                    torch.LongTensor(item_ids).to(self.device))

    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        elif mode == 'generate_batch':
            self.generate_batch(data)
            print(self.args.save_dir, self.args.rec_pre_trained_data)
            print('test (NDCG@10: %.4f, HR@10: %.4f), Num User: %.4f'
                  % (self.NDCG / self.users, self.HT / self.users, self.users))
            print('test (NDCG@20: %.4f, HR@20: %.4f), Num User: %.4f'
                  % (self.NDCG_20 / self.users, self.HIT_20 / self.users, self.users))
            print('test (NDCG@1:  %.4f, HR@1:  %.4f), Num User: %.4f'
                  % (self.NDCG_1 / self.users, self.HIT_1 / self.users, self.users))
            print('test (NDCG@5:  %.4f, HR@5:  %.4f), Num User: %.4f'
                  % (self.NDCG_5 / self.users, self.HIT_5 / self.users, self.users))
        elif mode == 'extract':
            self.extract_emb(data)

    # ────────────────────────────────────────────────────────────
    # pre_train_phase2（核心训练步，新增辅助损失）
    # ────────────────────────────────────────────────────────────
    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        print(self.args.save_dir, self.args.rec_pre_trained_data, self.args.llm)
        optimizer.zero_grad()

        u, seq, pos, neg = data

        text_input      = []
        candidates_pos  = []
        interact_embs   = []
        candidate_embs  = []

        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')

        for i in range(len(u)):
            target_item_id    = pos[i][-1]
            target_item_title = self.find_item_text_single(
                target_item_id, title_flag=True, description_flag=False)

            interact_text, interact_ids = self.make_interact_text(
                seq[i][seq[i] > 0], 10, u[i])

            candidate_num = 4
            candidate_text, candidate_ids = self.make_candidate_text(
                seq[i][seq[i] > 0], candidate_num,
                target_item_id, target_item_title, task='RecTask')

            input_text  = 'This user has made a series of purchases in the following order: '
            input_text += interact_text
            input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"

            text_input.append(input_text)
            candidates_pos += candidate_text

            interact_embs.append(
                self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(
                self.item_emb_proj(self.get_item_emb([candidate_ids])).squeeze(0))

        candidate_embs_cat = torch.cat(candidate_embs)

        samples = {
            'text_input':    text_input,
            'log_emb':       log_emb,
            'candidates_pos': candidates_pos,
            'interact':       interact_embs,
            'candidate_embs': candidate_embs_cat,
        }

        # ── 主损失（含 soft prompts）──────────────────────
        loss, rec_loss, match_loss = self.llm(samples, mode=0)

        # ── 辅助任务：TA Loss ─────────────────────────────
        # 需要拿到 user_outputs（128-d）重新计算避免 autograd 问题
        # 简化版：复用 rec_loss 的 user_outputs；
        # 这里用一次独立的前向抓取用户表示（不影响主损失图）
        if self.ta_loss_weight > 0 or self.rps_loss_weight > 0:
            user_outputs_aux = self._get_user_outputs(samples)  # [B, 128]

            if self.ta_loss_weight > 0:
                ta_loss = self._compute_ta_loss(
                    user_outputs_aux.detach(), seq)
                loss = loss + self.ta_loss_weight * ta_loss
            else:
                ta_loss = torch.tensor(0.0)

            if self.rps_loss_weight > 0:
                rps_loss = self._compute_rps_loss(
                    user_outputs_aux.detach(), pos, seq)
                loss = loss + self.rps_loss_weight * rps_loss
            else:
                rps_loss = torch.tensor(0.0)
        else:
            ta_loss  = torch.tensor(0.0)
            rps_loss = torch.tensor(0.0)

        print(f"Epoch {epoch}/{total_epoch} step {step}/{total_step} | "
              f"rec={rec_loss:.4f} match={match_loss:.4f} "
              f"TA={ta_loss.item():.4f} RPS={rps_loss.item():.4f}")

        loss.backward()
        if self.args.nn_parameter:
            htcore.mark_step()
        optimizer.step()
        if self.args.nn_parameter:
            htcore.mark_step()

    def _get_user_outputs(self, samples):
        """
        从已有的 text_input / interact 中，以 no_grad=False 方式
        抓取 user_outputs（用于辅助损失）。
        注意：此函数会再次经历 soft_prompts 拼接，
        梯度流向 soft_prompts / pred_user。
        """
        max_input_length = 1024
        k = self.llm.num_soft_prompts

        llm_tokens = self.llm.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt", padding="longest",
            truncation=True, max_length=max_input_length,
        ).to(self.device)

        inputs_embeds = self.llm.llm_model.get_input_embeddings()(
            llm_tokens['input_ids'])

        inputs_embeds = self.llm.replace_out_token_all(
            llm_tokens, inputs_embeds,
            token=['[UserOut]', '[HistoryEmb]'],
            embs={'[HistoryEmb]': samples['interact']})

        inputs_embeds, new_mask = self.llm._prepend_soft_prompts(
            inputs_embeds, llm_tokens.get('attention_mask'))

        with torch.amp.autocast('cuda'):
            outputs = self.llm.llm_model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=new_mask,
                output_hidden_states=True)
            indx = self.llm.get_embeddings(llm_tokens, '[UserOut]', offset=k)
            user_outputs = torch.cat([
                outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                for i in range(len(indx))])

        return self.llm.pred_user(user_outputs)   # [B, 128]

    def make_interact_text(self, interact_ids, interact_max_num, user):
        interact_item_titles_ = self.find_item_text(
            interact_ids, title_flag=True, description_flag=False)
        count = 1
        if interact_max_num == 'all':
            times = self.find_item_time(interact_ids, user)
        else:
            times = self.find_item_time(interact_ids[-interact_max_num:], user)

        interact_text = []
        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(
                    f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                count += 1
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(
                    f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
                count += 1
            interact_ids = interact_ids[-interact_max_num:]

        return ','.join(interact_text), interact_ids

    def make_candidate_text(self, interact_ids, candidate_num, target_item_id,
                            target_item_title, candi_set=None, task='ItemTask'):
        neg_item_id = []
        if candi_set is None:
            while len(neg_item_id) < 99:
                t = np.random.randint(1, self.item_num + 1)
                if not (t in interact_ids or t in neg_item_id):
                    neg_item_id.append(t)
        else:
            his   = set(interact_ids)
            items = list(candi_set.difference(his))
            if len(items) > 99:
                neg_item_id = random.sample(items, 99)
            else:
                while len(neg_item_id) < 49:
                    t = np.random.randint(1, self.item_num + 1)
                    if not (t in interact_ids or t in neg_item_id):
                        neg_item_id.append(t)
        random.shuffle(neg_item_id)

        candidate_ids   = [target_item_id]
        candidate_text  = [
            'The item title and item embedding are as follows: '
            + target_item_title
            + "[HistoryEmb], then generate item representation token:[ItemOut]"
        ]
        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append(
                'The item title and item embedding are as follows: '
                + self.find_item_text_single(neg_candidate,
                                             title_flag=True, description_flag=False)
                + "[HistoryEmb], then generate item representation token:[ItemOut]")
            candidate_ids.append(neg_candidate)

        return candidate_text, candidate_ids

    def make_candidate(self, interact_ids, candidate_num, target_item_id,
                       target_item_title, candi_set=None, task='ItemTask'):
        neg_item_id = []
        while len(neg_item_id) < 99:
            t = np.random.randint(1, self.item_num + 1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        random.shuffle(neg_item_id)
        return [target_item_id] + neg_item_id[:candidate_num - 1]

    # ────────────────────────────────────────────────────────────
    # generate_batch / extract_emb（保持原版逻辑，仅加 offset）
    # ────────────────────────────────────────────────────────────
    def split_into_batches(self, itemnum, m):
        numbers = list(range(1, itemnum + 1))
        return [numbers[i:i + m] for i in range(0, itemnum, m)]

    def generate_batch(self, data):
        k = self.llm.num_soft_prompts
        if self.all_embs is None:
            batch_ = 128
            if self.args.llm == 'llama':
                batch_ = 64
            if self.args.rec_pre_trained_data in ('Electronics', 'Books'):
                batch_ = 64
                if self.args.llm == 'llama':
                    batch_ = 32
            batches = self.split_into_batches(self.item_num, batch_)
            self.all_embs = []
            max_input_length = 1024

            for bat in tqdm(batches):
                candidate_text = []
                candidate_ids  = []
                candidate_embs = []
                for neg_candidate in bat:
                    candidate_text.append(
                        'The item title and item embedding are as follows: '
                        + self.find_item_text_single(neg_candidate,
                                                     title_flag=True,
                                                     description_flag=False)
                        + "[HistoryEmb], then generate item representation token:[ItemOut]")
                    candidate_ids.append(neg_candidate)

                with torch.no_grad():
                    candi_tokens = self.llm.llm_tokenizer(
                        candidate_text, return_tensors="pt",
                        padding="longest", truncation=True,
                        max_length=max_input_length).to(self.device)
                    candidate_embs.append(
                        self.item_emb_proj(self.get_item_emb(candidate_ids)))
                    candi_embeds = self.llm.llm_model.get_input_embeddings()(
                        candi_tokens['input_ids'])
                    candi_embeds = self.llm.replace_out_token_all_infer(
                        candi_tokens, candi_embeds,
                        token=['[ItemOut]', '[HistoryEmb]'],
                        embs={'[HistoryEmb]': candidate_embs[0]})
                    # ── soft prompts ──────────────────────
                    candi_embeds, candi_mask = self.llm._prepend_soft_prompts(
                        candi_embeds, candi_tokens.get('attention_mask'))

                    with torch.amp.autocast('cuda'):
                        candi_outputs = self.llm.llm_model.forward(
                            inputs_embeds=candi_embeds,
                            attention_mask=candi_mask,
                            output_hidden_states=True)
                        indx = self.llm.get_embeddings(
                            candi_tokens, '[ItemOut]', offset=k)
                        item_outputs = torch.cat([
                            candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                            for i in range(len(indx))])
                        item_outputs = self.llm.pred_item(item_outputs)

                    self.all_embs.append(item_outputs)
                    del candi_outputs, item_outputs
            self.all_embs = torch.cat(self.all_embs)

        u, seq, pos, neg, rank, candi_set, files = data
        text_input    = []
        interact_embs = []
        candidate     = []

        with torch.no_grad():
            for i in range(len(u)):
                target_item_id    = pos[i]
                target_item_title = self.find_item_text_single(
                    target_item_id, title_flag=True, description_flag=False)
                interact_text, interact_ids = self.make_interact_text(
                    seq[i][seq[i] > 0], 10, u[i])
                candidate_ids = self.make_candidate(
                    seq[i][seq[i] > 0], 100, target_item_id, target_item_title, candi_set)
                candidate.append(candidate_ids)

                input_text  = 'This user has made a series of purchases in the following order: '
                input_text += interact_text
                input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"
                text_input.append(input_text)
                interact_embs.append(
                    self.item_emb_proj(self.get_item_emb(interact_ids)))

            max_input_length = 1024
            llm_tokens = self.llm.llm_tokenizer(
                text_input, return_tensors="pt",
                padding="longest", truncation=True,
                max_length=max_input_length).to(self.device)
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(
                llm_tokens['input_ids'])
            inputs_embeds = self.llm.replace_out_token_all(
                llm_tokens, inputs_embeds,
                token=['[UserOut]', '[HistoryEmb]'],
                embs={'[HistoryEmb]': interact_embs})
            # ── soft prompts ──────────────────────────────
            inputs_embeds, new_mask = self.llm._prepend_soft_prompts(
                inputs_embeds, llm_tokens.get('attention_mask'))

            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,
                    attention_mask=new_mask,
                    output_hidden_states=True)
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]', offset=k)
                user_outputs = torch.cat([
                    outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                    for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)

                for i in range(len(candidate)):
                    item_outputs = self.all_embs[np.array(candidate[i]) - 1]
                    logits = torch.mm(
                        item_outputs, user_outputs[i].unsqueeze(0).T).squeeze(-1)
                    logits = -1 * logits
                    rank   = logits.argsort().argsort()[0].item()

                    if rank < 10:
                        self.NDCG  += 1 / np.log2(rank + 2)
                        self.HT    += 1
                    if rank < 20:
                        self.NDCG_20 += 1 / np.log2(rank + 2)
                        self.HIT_20  += 1
                    if rank < 1:
                        self.NDCG_1 += 1 / np.log2(rank + 2)
                        self.HIT_1  += 1
                    if rank < 5:
                        self.NDCG_5 += 1 / np.log2(rank + 2)
                        self.HIT_5  += 1
                    self.users += 1
        return self.NDCG

    def extract_emb(self, data):
        k = self.llm.num_soft_prompts
        u, seq, pos, neg, original_seq, rank, files = data
        text_input    = []
        interact_embs = []

        with torch.no_grad():
            for i in range(len(u)):
                interact_text, interact_ids = self.make_interact_text(
                    seq[i][seq[i] > 0], 10, u[i])
                input_text  = 'This user has made a series of purchases in the following order: '
                input_text += interact_text
                input_text += ". Based on this sequence of purchases, generate user representation token:[UserOut]"
                text_input.append(input_text)
                interact_embs.append(
                    self.item_emb_proj(self.get_item_emb(interact_ids)))

            max_input_length = 1024
            llm_tokens = self.llm.llm_tokenizer(
                text_input, return_tensors="pt",
                padding="longest", truncation=True,
                max_length=max_input_length).to(self.device)
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(
                llm_tokens['input_ids'])
            inputs_embeds = self.llm.replace_out_token_all(
                llm_tokens, inputs_embeds,
                token=['[UserOut]', '[HistoryEmb]'],
                embs={'[HistoryEmb]': interact_embs})
            inputs_embeds, new_mask = self.llm._prepend_soft_prompts(
                inputs_embeds, llm_tokens.get('attention_mask'))

            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(
                    inputs_embeds=inputs_embeds,
                    attention_mask=new_mask,
                    output_hidden_states=True)
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]', offset=k)
                user_outputs = torch.cat([
                    outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                    for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)
                self.extract_embs_list.append(user_outputs.detach().cpu())
        return 0