"""
seqllm_model.py  —— 两阶段 DELRec 实现
Stage 1: 只训练 soft_prompts，用 TA + RPS(真实SASRec预测) loss
Stage 2: 冻结 soft_prompts，训练 rec + match loss
"""

import random
import pickle
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.seqllm4rec import *
from datetime import datetime
from tqdm import trange, tqdm

try:
    import habana_frameworks.torch.core as htcore
except Exception:
    pass


class llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        with open(f'./SeqRec/data_{args.rec_pre_trained_data}/text_name_dict.json.gz', 'rb') as ft:
            self.text_name_dict = pickle.load(ft)

        self.recsys = RecSys(args.recsys, args.rec_pre_trained_data, self.device)
        self.item_num    = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units

        self.mse = nn.MSELoss()
        self.all_embs = None
        self.maxlen   = args.maxlen

        self.NDCG = self.HIT = 0
        self.NDCG_20 = self.HIT_20 = 0
        self.NDCG_1  = self.HIT_1  = 0
        self.NDCG_5  = self.HIT_5  = 0
        self.users = self.HT = 0.0
        self.extract_embs_list = []

        num_soft_prompts = getattr(args, 'num_soft_prompts', 8)
        self.llm = llm4rec(device=self.device, llm_model=args.llm,
                           args=self.args, num_soft_prompts=num_soft_prompts)

        d_llm = self.llm.llm_model.config.hidden_size
        self.item_emb_proj = nn.Sequential(
            nn.Linear(self.rec_sys_dim, d_llm), nn.LayerNorm(d_llm),
            nn.LeakyReLU(), nn.Linear(d_llm, d_llm))
        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        nn.init.xavier_normal_(self.item_emb_proj[3].weight)

        self.ta_loss_weight  = getattr(args, 'ta_loss_weight',  0.3)
        self.rps_loss_weight = getattr(args, 'rps_loss_weight', 0.3)

    # ── SASRec 真实 top-k 预测 ────────────────────────────────────────────────
    def get_sasrec_topk(self, u, seq, k=10):
        """
        直接调 self.recsys.model 拿 SASRec 的用户表示，
        与全量 item embedding 做内积，返回 top-k item id（1-indexed）
        """
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, None, None, mode='log_only')

            if self.args.nn_parameter:
                item_embs = self.recsys.model.item_emb
            else:
                item_embs = self.recsys.model.item_emb.weight

            scores = torch.matmul(log_emb, item_embs.T)  # [B, item_num]

            # 屏蔽 padding 行（第0列）
            scores[:, 0] = -1e9
            # 屏蔽历史：全程在 CUDA 上操作
            for i in range(len(u)):
                hist = seq[i][seq[i] > 0]  # numpy
                hist_idx = torch.LongTensor(hist).to(self.device)  # CUDA
                scores[i].scatter_(0, hist_idx, -1e9)

            topk = scores.topk(k, dim=-1).indices
        return topk.cpu().numpy().tolist()

    # ── 优化器 ────────────────────────────────────────────────────────────────
    def build_optimizer_stage1(self):
        self.llm.soft_prompts.requires_grad = True
        params = ([self.llm.soft_prompts]
                  + list(self.llm.pred_user.parameters())
                  + list(self.item_emb_proj.parameters())
                  + list(self.llm.pred_item.parameters()))
        return torch.optim.Adam(
            params, lr=getattr(self.args, 'stage1_lr', 1e-3), betas=(0.9, 0.98))

    def build_optimizer_stage2(self):
        """Stage 2：冻结 soft_prompts，其余可训练参数用 stage2_lr"""
        self.llm.soft_prompts.requires_grad = False
        other_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(other_params, lr=self.args.stage2_lr, betas=(0.9, 0.98))

    def _compute_ta_loss_128(self, user_outputs, seq_batch):
        ta_losses = []
        for i in range(len(seq_batch)):
            valid_ids = seq_batch[i][seq_batch[i] > 0]
            if len(valid_ids) < 2:
                continue
            ta_target_id = int(valid_ids[-2])
            ta_neg_ids = []
            while len(ta_neg_ids) < 3:
                neg = np.random.randint(1, self.item_num + 1)
                if neg != ta_target_id:
                    ta_neg_ids.append(neg)
            all_ids = [ta_target_id] + ta_neg_ids
            item_embs = self.llm.pred_item(
                    self.item_emb_proj(self.get_item_emb(all_ids)))  # [4, 128]
            scores = torch.matmul(item_embs, user_outputs[i].unsqueeze(-1)).squeeze(-1)
            label = torch.tensor(0, device=self.device)
            ta_losses.append(F.cross_entropy(scores.unsqueeze(0), label.unsqueeze(0)))
        if not ta_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(ta_losses).mean()

    def _compute_rps_loss_128(self, user_outputs, sasrec_topk, seq_batch):
        rps_losses = []
        for i in range(len(sasrec_topk)):
            target_id = int(sasrec_topk[i][0])
            if target_id == 0:
                continue
            history = set(seq_batch[i][seq_batch[i] > 0].tolist())
            neg_ids = []
            while len(neg_ids) < 3:
                neg = np.random.randint(1, self.item_num + 1)
                if neg != target_id and neg not in history:
                    neg_ids.append(neg)
            all_ids = [target_id] + neg_ids
            item_embs = self.llm.pred_item(
                    self.item_emb_proj(self.get_item_emb(all_ids)))  # [4, 128]
            scores = torch.matmul(item_embs, user_outputs[i].unsqueeze(-1)).squeeze(-1)
            label = torch.tensor(0, device=self.device)
            rps_losses.append(F.cross_entropy(scores.unsqueeze(0), label.unsqueeze(0)))
        if not rps_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(rps_losses).mean()

    # ── Stage 1 训练步 ────────────────────────────────────────────────────────
    def train_stage1(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        optimizer.zero_grad()

        u, seq, pos, neg = data

        # SASRec 真实 top-k 预测（用于 RPS）
        sasrec_topk = self.get_sasrec_topk(u, seq, k=10)

        # 构建输入文本和 interact embeddings
        text_input = []
        interact_embs = []

        for i in range(len(u)):
            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10, u[i])
            input_text = ('This user has made a series of purchases in the following order: '
                          + interact_text
                          + '. Based on this sequence of purchases, '
                            'generate user representation token:[UserOut]')
            text_input.append(input_text)
            with torch.no_grad():
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))

        k = self.llm.num_soft_prompts

        # tokenize
        llm_tokens = self.llm.llm_tokenizer(
            text_input, return_tensors="pt", padding="longest",
            truncation=True, max_length=1024).to(self.device)

        # embed + 替换特殊token（item embedding 侧全部 no_grad）
        with torch.no_grad():
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = self.llm.replace_out_token_all(
                llm_tokens, inputs_embeds,
                token=['[UserOut]', '[HistoryEmb]'],
                embs={'[HistoryEmb]': interact_embs})

        # soft_prompts 拼接（soft_prompts 有梯度，此后计算图保持连接）
        inputs_embeds, new_mask = self.llm._prepend_soft_prompts(
            inputs_embeds, llm_tokens.get('attention_mask'))

        # LLM forward（冻结，但梯度可以通过 inputs_embeds 流向 soft_prompts）
        with torch.amp.autocast('cuda'):
            outputs = self.llm.llm_model.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=new_mask,
                output_hidden_states=True)
            indx = self.llm.get_embeddings(llm_tokens, '[UserOut]', offset=k)
            user_hidden = torch.cat([
                outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                for i in range(len(indx))])

        # pred_user 不加 no_grad：梯度路径 loss → user_outputs → pred_user → user_hidden → soft_prompts
        user_outputs = self.llm.pred_user(user_hidden)  # [B, 128]

        # 计算辅助损失（item侧全部 no_grad，只有用户侧有梯度）
        loss = torch.tensor(0.0, device=self.device)

        if self.ta_loss_weight > 0:
            ta_loss = self._compute_ta_loss_128(user_outputs, seq)
            loss = loss + self.ta_loss_weight * ta_loss
        else:
            ta_loss = torch.tensor(0.0)

        if self.rps_loss_weight > 0:
            rps_loss = self._compute_rps_loss_128(user_outputs, sasrec_topk, seq)
            loss = loss + self.rps_loss_weight * rps_loss
        else:
            rps_loss = torch.tensor(0.0)

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        print(f"[Stage1] Epoch {epoch}/{total_epoch} step {step}/{total_step} | "
              f"TA={ta_loss.item():.4f} RPS={rps_loss.item():.4f}")

    def _compute_ta_loss_raw(self, user_hidden, seq_batch):
        """
        用原始 user_hidden（d_llm 维）做 TA loss，
        item 侧也映射到 d_llm 空间（item_emb_proj 输出），
        梯度流向 user_hidden → soft_prompts
        """
        ta_losses = []
        for i in range(len(seq_batch)):
            valid_ids = seq_batch[i][seq_batch[i] > 0]
            if len(valid_ids) < 2:
                continue
            ta_target_id = int(valid_ids[-2])
            ta_neg_ids   = []
            while len(ta_neg_ids) < 3:
                neg = np.random.randint(1, self.item_num + 1)
                if neg != ta_target_id:
                    ta_neg_ids.append(neg)

            all_ids = [ta_target_id] + ta_neg_ids
            with torch.no_grad():
                # item_emb_proj 输出在 d_llm 空间，和 user_hidden 维度一致
                item_embs = self.item_emb_proj(self.get_item_emb(all_ids))  # [4, d_llm]
                item_embs = F.normalize(item_embs, p=2, dim=-1)

            u_vec  = F.normalize(user_hidden[i], p=2, dim=-1)               # [d_llm]
            scores = torch.matmul(item_embs, u_vec.unsqueeze(-1)).squeeze(-1)  # [4]
            label  = torch.tensor(0, device=self.device)
            ta_losses.append(F.cross_entropy(scores.unsqueeze(0), label.unsqueeze(0)))

        if not ta_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(ta_losses).mean()

    def _compute_rps_loss_raw(self, user_hidden, sasrec_topk, seq_batch):
        """用 user_hidden 做 RPS loss，梯度流向 soft_prompts"""
        rps_losses = []
        for i in range(len(sasrec_topk)):
            target_id = int(sasrec_topk[i][0])
            if target_id == 0:
                continue
            history = set(seq_batch[i][seq_batch[i] > 0].tolist())
            neg_ids = []
            while len(neg_ids) < 3:
                neg = np.random.randint(1, self.item_num + 1)
                if neg != target_id and neg not in history:
                    neg_ids.append(neg)

            all_ids = [target_id] + neg_ids
            with torch.no_grad():
                item_embs = self.item_emb_proj(self.get_item_emb(all_ids))  # [4, d_llm]
                item_embs = F.normalize(item_embs, p=2, dim=-1)

            u_vec  = F.normalize(user_hidden[i], p=2, dim=-1)
            scores = torch.matmul(item_embs, u_vec.unsqueeze(-1)).squeeze(-1)
            label  = torch.tensor(0, device=self.device)
            rps_losses.append(F.cross_entropy(scores.unsqueeze(0), label.unsqueeze(0)))

        if not rps_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(rps_losses).mean()

    # ── Stage 2 训练步 ────────────────────────────────────────────────────────
    def train_stage2(self, data, optimizer, batch_iter):
        """
        soft_prompts 已冻结，只训练 rec + match loss
        """
        epoch, total_epoch, step, total_step = batch_iter
        optimizer.zero_grad()

        u, seq, pos, neg = data

        text_input     = []
        candidates_pos = []
        interact_embs  = []
        candidate_embs = []

        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode='log_only')

        for i in range(len(u)):
            target_item_id    = pos[i][-1]
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10, u[i])
            candidate_num = 4
            candidate_text, candidate_ids = self.make_candidate_text(
                seq[i][seq[i] > 0], candidate_num, target_item_id, target_item_title)

            input_text = ('This user has made a series of purchases in the following order: '
                          + interact_text
                          + '. Based on this sequence of purchases, '
                            'generate user representation token:[UserOut]')
            text_input.append(input_text)
            candidates_pos += candidate_text
            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            candidate_embs.append(
                self.item_emb_proj(self.get_item_emb([candidate_ids])).squeeze(0))

        samples = {
            'text_input':     text_input,
            'log_emb':        log_emb,
            'candidates_pos': candidates_pos,
            'interact':       interact_embs,
            'candidate_embs': torch.cat(candidate_embs),
        }

        loss, rec_loss, match_loss, _ = self.llm(samples, mode=0)

        print(f"[Stage2] Epoch {epoch}/{total_epoch} step {step}/{total_step} | "
              f"rec={rec_loss:.4f} match={match_loss:.4f}")

        loss.backward()
        if self.args.nn_parameter:
            htcore.mark_step()
        optimizer.step()
        if self.args.nn_parameter:
            htcore.mark_step()

    # ── forward 统一入口 ──────────────────────────────────────────────────────
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase2'):
        if mode == 'phase1':
            self.train_stage1(data, optimizer, batch_iter)
        elif mode == 'phase2':
            self.train_stage2(data, optimizer, batch_iter)
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

    # ── 保存 / 加载 ───────────────────────────────────────────────────────────
    def save_model(self, args, epoch2=None, best=False):
        out_dir = f'./models/{args.save_dir}/'
        if best:
            out_dir = out_dir[:-1] + 'best/'
        create_dir(out_dir)
        out_dir += f'{args.rec_pre_trained_data}_{args.llm}_{epoch2}_'
        if args.train:
            torch.save(self.item_emb_proj.state_dict(),  out_dir + 'item_proj.pt')
            torch.save(self.llm.pred_user.state_dict(),  out_dir + 'pred_user.pt')
            torch.save(self.llm.pred_item.state_dict(),  out_dir + 'pred_item.pt')
            torch.save(self.llm.soft_prompts.data,       out_dir + 'soft_prompts.pt')
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

    def load_model(self, args, phase2_epoch=None):
        out_dir = (f'./models/{args.save_dir}/'
                   f'{args.rec_pre_trained_data}_{args.llm}_{phase2_epoch}_')
        self.item_emb_proj.load_state_dict(
            torch.load(out_dir + 'item_proj.pt', map_location=self.device))
        self.llm.pred_user.load_state_dict(
            torch.load(out_dir + 'pred_user.pt', map_location=self.device))
        self.llm.pred_item.load_state_dict(
            torch.load(out_dir + 'pred_item.pt', map_location=self.device))
        sp_path = out_dir + 'soft_prompts.pt'
        if os.path.exists(sp_path):
            self.llm.soft_prompts.data = torch.load(sp_path, map_location=self.device)
        if not args.token:
            if args.nn_parameter:
                self.llm.CLS.data      = torch.load(out_dir + 'CLS.pt', map_location=self.device)
                self.llm.CLS_item.data = torch.load(out_dir + 'CLS_item.pt', map_location=self.device)
            else:
                self.llm.CLS.load_state_dict(torch.load(out_dir + 'CLS.pt', map_location=self.device))
                self.llm.CLS_item.load_state_dict(torch.load(out_dir + 'CLS_item.pt', map_location=self.device))
        if args.token:
            self.llm.llm_model.model.embed_tokens.load_state_dict(
                torch.load(out_dir + 'token.pt', map_location=self.device))

    # ── 工具方法 ──────────────────────────────────────────────────────────────
    def find_item_text(self, item, title_flag=True, description_flag=True):
        t, d, t_, d_ = 'title', 'description', 'No Title', 'No Description'
        if title_flag and description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]

    def find_item_time(self, item, user):
        l = [datetime.utcfromtimestamp(int(self.text_name_dict['time'][i][user]) / 1000) for i in item]
        return [l_.strftime('%Y-%m-%d') for l_ in l]

    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t, d, t_, d_ = 'title', 'description', 'No Title', 'No Description'
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        return f'"{self.text_name_dict[d].get(item,d_)}"'

    def get_item_emb(self, item_ids):
        with torch.no_grad():
            if self.args.nn_parameter:
                return self.recsys.model.item_emb[torch.LongTensor(item_ids).to(self.device)]
            return self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))

    def make_interact_text(self, interact_ids, interact_max_num, user):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False)
        count = 1
        times = self.find_item_time(interact_ids[-interact_max_num:] if interact_max_num != 'all' else interact_ids, user)
        interact_text = []
        titles = interact_item_titles_ if interact_max_num == 'all' else interact_item_titles_[-interact_max_num:]
        for title in titles:
            interact_text.append(f'Item No.{count}, Time: {times[count-1]}, ' + title + '[HistoryEmb]')
            count += 1
        if interact_max_num != 'all':
            interact_ids = interact_ids[-interact_max_num:]
        return ','.join(interact_text), interact_ids

    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set=None, task='ItemTask'):
        neg_item_id = []
        while len(neg_item_id) < 99:
            t = np.random.randint(1, self.item_num + 1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        random.shuffle(neg_item_id)
        candidate_ids  = [target_item_id]
        candidate_text = ['The item title and item embedding are as follows: '
                          + target_item_title
                          + "[HistoryEmb], then generate item representation token:[ItemOut]"]
        for neg_candidate in neg_item_id[:candidate_num - 1]:
            candidate_text.append('The item title and item embedding are as follows: '
                                   + self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False)
                                   + "[HistoryEmb], then generate item representation token:[ItemOut]")
            candidate_ids.append(neg_candidate)
        return candidate_text, candidate_ids

    def make_candidate(self, interact_ids, candidate_num, target_item_id, target_item_title, candi_set=None):
        neg_item_id = []
        while len(neg_item_id) < 99:
            t = np.random.randint(1, self.item_num + 1)
            if not (t in interact_ids or t in neg_item_id):
                neg_item_id.append(t)
        random.shuffle(neg_item_id)
        return [target_item_id] + neg_item_id[:candidate_num - 1]

    def split_into_batches(self, itemnum, m):
        return [list(range(1, itemnum + 1))[i:i + m] for i in range(0, itemnum, m)]

    def generate_batch(self, data):
        k = self.llm.num_soft_prompts
        if self.all_embs is None:
            batch_ = 32 if (self.args.llm == 'llama' and self.args.rec_pre_trained_data in ('Electronics', 'Books')) else \
                     64 if (self.args.llm == 'llama' or self.args.rec_pre_trained_data in ('Electronics', 'Books')) else 128
            batches = self.split_into_batches(self.item_num, batch_)
            self.all_embs = []
            for bat in tqdm(batches):
                candidate_text, candidate_ids, candidate_embs = [], [], []
                for nc in bat:
                    candidate_text.append('The item title and item embedding are as follows: '
                                          + self.find_item_text_single(nc, title_flag=True, description_flag=False)
                                          + "[HistoryEmb], then generate item representation token:[ItemOut]")
                    candidate_ids.append(nc)
                with torch.no_grad():
                    candi_tokens = self.llm.llm_tokenizer(candidate_text, return_tensors="pt",
                                                          padding="longest", truncation=True, max_length=1024).to(self.device)
                    candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
                    candi_embeds = self.llm.llm_model.get_input_embeddings()(candi_tokens['input_ids'])
                    candi_embeds = self.llm.replace_out_token_all_infer(
                        candi_tokens, candi_embeds, token=['[ItemOut]', '[HistoryEmb]'],
                        embs={'[HistoryEmb]': candidate_embs[0]})
                    candi_embeds, candi_mask = self.llm._prepend_soft_prompts(
                        candi_embeds, candi_tokens.get('attention_mask'))
                    with torch.amp.autocast('cuda'):
                        candi_outputs = self.llm.llm_model.forward(
                            inputs_embeds=candi_embeds, attention_mask=candi_mask, output_hidden_states=True)
                        indx = self.llm.get_embeddings(candi_tokens, '[ItemOut]', offset=k)
                        item_outputs = torch.cat([candi_outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0)
                                                  for i in range(len(indx))])
                        item_outputs = self.llm.pred_item(item_outputs)
                    self.all_embs.append(item_outputs)
                    del candi_outputs, item_outputs
            self.all_embs = torch.cat(self.all_embs)

        u, seq, pos, neg, rank, candi_set, files = data
        text_input, interact_embs, candidate = [], [], []
        with torch.no_grad():
            for i in range(len(u)):
                target_item_id    = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False)
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10, u[i])
                candidate_ids = self.make_candidate(seq[i][seq[i] > 0], 100, target_item_id, target_item_title, candi_set)
                candidate.append(candidate_ids)
                text_input.append('This user has made a series of purchases in the following order: '
                                   + interact_text
                                   + '. Based on this sequence of purchases, generate user representation token:[UserOut]')
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))

            llm_tokens = self.llm.llm_tokenizer(text_input, return_tensors="pt", padding="longest",
                                                 truncation=True, max_length=1024).to(self.device)
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = self.llm.replace_out_token_all(
                llm_tokens, inputs_embeds, token=['[UserOut]', '[HistoryEmb]'],
                embs={'[HistoryEmb]': interact_embs})
            inputs_embeds, new_mask = self.llm._prepend_soft_prompts(inputs_embeds, llm_tokens.get('attention_mask'))
            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(inputs_embeds=inputs_embeds, attention_mask=new_mask, output_hidden_states=True)
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]', offset=k)
                user_outputs = torch.cat([outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)
                for i in range(len(candidate)):
                    item_outputs = self.all_embs[torch.LongTensor(candidate[i]).to(self.device) - 1]
                    logits = -1 * torch.mm(item_outputs, user_outputs[i].unsqueeze(0).T).squeeze(-1)
                    rank   = logits.argsort().argsort()[0].item()
                    if rank < 10:  self.NDCG += 1 / np.log2(rank + 2); self.HT += 1
                    if rank < 20:  self.NDCG_20 += 1 / np.log2(rank + 2); self.HIT_20 += 1
                    if rank < 1:   self.NDCG_1  += 1 / np.log2(rank + 2); self.HIT_1  += 1
                    if rank < 5:   self.NDCG_5  += 1 / np.log2(rank + 2); self.HIT_5  += 1
                    self.users += 1
        return self.NDCG

    def extract_emb(self, data):
        k = self.llm.num_soft_prompts
        u, seq, pos, neg, original_seq, rank, files = data
        text_input, interact_embs = [], []
        with torch.no_grad():
            for i in range(len(u)):
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i] > 0], 10, u[i])
                text_input.append('This user has made a series of purchases in the following order: '
                                   + interact_text
                                   + '. Based on this sequence of purchases, generate user representation token:[UserOut]')
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids)))
            llm_tokens = self.llm.llm_tokenizer(text_input, return_tensors="pt", padding="longest",
                                                 truncation=True, max_length=1024).to(self.device)
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = self.llm.replace_out_token_all(
                llm_tokens, inputs_embeds, token=['[UserOut]', '[HistoryEmb]'],
                embs={'[HistoryEmb]': interact_embs})
            inputs_embeds, new_mask = self.llm._prepend_soft_prompts(inputs_embeds, llm_tokens.get('attention_mask'))
            with torch.cuda.amp.autocast():
                outputs = self.llm.llm_model.forward(inputs_embeds=inputs_embeds, attention_mask=new_mask, output_hidden_states=True)
                indx = self.llm.get_embeddings(llm_tokens, '[UserOut]', offset=k)
                user_outputs = torch.cat([outputs.hidden_states[-1][i, indx[i]].mean(axis=0).unsqueeze(0) for i in range(len(indx))])
                user_outputs = self.llm.pred_user(user_outputs)
                self.extract_embs_list.append(user_outputs.detach().cpu())
        return 0