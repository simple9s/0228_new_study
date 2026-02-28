"""
train_model.py  —— 在原版基础上：
  * 用 model.build_optimizer() 替代手工 Adam，实现差异化学习率
    (soft_prompts: 1e-3, 其余: stage2_lr=1e-4)
  * 新增命令行参数 --num_soft_prompts / --ta_loss_weight / --rps_loss_weight
"""

import os
import torch
import random
import time
import sys
import numpy as np

from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LambdaLR

from models.seqllm_model import *
from SeqRec.sasrec.utils import data_partition, SeqDataset, SeqDataset_Inference, SeqDataset_Validation


def save_checkpoint(args, epoch, model, optimizer, scheduler,
                    best_perform, early_stop, checkpoint_dir='./checkpoints1/'):
    create_dir(checkpoint_dir)
    path = os.path.join(checkpoint_dir,
                        f'{args.save_dir}_{args.rec_pre_trained_data}_checkpoint.pth')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': (model.module.state_dict()
                             if isinstance(model, DDP) else model.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_perform': best_perform,
        'early_stop':   early_stop,
        'args':         args,
        'random_state':       random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state().cpu(),
        'cuda_random_state':  (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}: {path}")


def load_checkpoint(args, model, optimizer, scheduler,
                    checkpoint_dir='./checkpoints1/'):
    path = os.path.join(checkpoint_dir,
                        f'{args.save_dir}_{args.rec_pre_trained_data}_checkpoint.pth')
    if not os.path.exists(path):
        print("No checkpoint found. Starting from scratch.")
        return None
    print(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    (model.module if isinstance(model, DDP) else model).load_state_dict(
        ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    random.setstate(ckpt['random_state'])
    np.random.set_state(ckpt['numpy_random_state'])
    torch.set_rng_state(ckpt['torch_random_state'])
    if ckpt['cuda_random_state'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt['cuda_random_state'])
    print(f"Resumed from epoch {ckpt['epoch']}, "
          f"best_perform: {ckpt['best_perform']}, early_stop: {ckpt['early_stop']}")
    return ckpt['epoch'] + 1, ckpt['best_perform'], ckpt['early_stop']


def setup_ddp(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)
    if args.device.type == 'hpu':
        import habana_frameworks.torch.distributed.hccl
        init_process_group(backend="hccl", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def train_model(args):
    print('LLMRec start train\n')
    if args.multi_gpu:
        mp.spawn(train_model_, args=(args.world_size, args),
                 nprocs=args.world_size, join=True)
    else:
        train_model_(0, 0, args)


def train_model_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size, args)
        args.device = ('cuda:' + str(rank)
                       if args.device != 'hpu' else torch.device('hpu'))
    random.seed(0)

    model = llmrec_model(args).to(args.device)

    dataset = data_partition(
        args.rec_pre_trained_data, args,
        path=f'./SeqRec/data_{args.rec_pre_trained_data}/{args.rec_pre_trained_data}')
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset
    print(f'user num: {usernum}  item num: {itemnum}')
    num_batch = len(user_train) // args.batch_size
    cc = sum(len(user_train[u]) for u in user_train)
    print('average sequence length: %.2f' % (cc / len(user_train)))

    train_data_set = SeqDataset(user_train, len(user_train.keys()), itemnum, args.maxlen)
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size,
                                       sampler=DistributedSampler(train_data_set, shuffle=True),
                                       pin_memory=True)
        model = DDP(model, static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size=args.batch_size,
                                       pin_memory=True, shuffle=True)

    # ── 差异化学习率优化器 ─────────────────────────────────
    base_model = model.module if isinstance(model, DDP) else model
    adam_optimizer = base_model.build_optimizer()
    scheduler = LambdaLR(adam_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    ckpt = load_checkpoint(args, model, adam_optimizer, scheduler)
    if ckpt is not None:
        epoch_start_idx, best_perform, early_stop = ckpt
    else:
        epoch_start_idx, best_perform, early_stop = 1, 0, 0

    early_thres = 5
    t0 = time.time()

    # 推理集准备
    eval_set_use = eval_set[1]
    users = (random.sample(list(eval_set_use), 10000)
             if len(eval_set_use) > 10000 else list(eval_set_use))
    user_list = [u for u in users if len(user_test[u]) >= 1]

    inference_data_set = SeqDataset_Inference(
        user_train, user_valid, user_test, user_list, itemnum, args.maxlen)
    if args.multi_gpu:
        inference_data_loader = DataLoader(
            inference_data_set, batch_size=args.batch_size_infer,
            sampler=DistributedSampler(inference_data_set, shuffle=True), pin_memory=True)
    else:
        inference_data_loader = DataLoader(
            inference_data_set, batch_size=args.batch_size_infer, pin_memory=True)

    # ── 训练循环 ───────────────────────────────────────────
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        model.train()
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)

        for step, data in enumerate(train_data_loader):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model([u, seq, pos, neg], optimizer=adam_optimizer,
                  batch_iter=[epoch, args.num_epochs + 1, step, num_batch],
                  mode='phase2')

            if step % (num_batch // 1) == 0 and step != 0:
                # ── 验证 ───────────────────────────────────
                _reset_metrics(model, args)
                eval_set_use = eval_set[0]
                v_users = (random.sample(list(eval_set_use), 10000)
                           if len(eval_set_use) > 10000 else list(eval_set_use))
                user_list_valid = [u for u in v_users if len(user_valid[u]) >= 1]
                valid_data_set = SeqDataset_Validation(
                    user_train, user_valid, user_list_valid, itemnum, args.maxlen)
                valid_data_loader = DataLoader(
                    valid_data_set, batch_size=args.batch_size_infer,
                    pin_memory=True, shuffle=True)

                model.eval()
                with torch.no_grad():
                    for _, vdata in enumerate(valid_data_loader):
                        vu, vseq, vpos, vneg = vdata
                        vu, vseq, vpos, vneg = (vu.numpy(), vseq.numpy(),
                                                vpos.numpy(), vneg.numpy())
                        model([vu, vseq, vpos, vneg, 0, None, 'original'],
                              mode='generate_batch')

                m = model.module if args.multi_gpu else model
                perform = m.HT / m.users

                if perform >= best_perform:
                    best_perform = perform
                    if rank == 0:
                        (model.module if args.multi_gpu else model).save_model(
                            args, epoch2=epoch, best=True)
                    _reset_metrics(model, args)
                    # 正式测试集评测
                    with torch.no_grad():
                        for _, tdata in enumerate(inference_data_loader):
                            tu, tseq, tpos, tneg = tdata
                            tu, tseq, tpos, tneg = (tu.numpy(), tseq.numpy(),
                                                    tpos.numpy(), tneg.numpy())
                            model([tu, tseq, tpos, tneg, 0, None, 'original'],
                                  mode='generate_batch')
                    _write_results(args, model, epoch)
                    early_stop = 0
                else:
                    (model.module if args.multi_gpu else model).save_model(
                        args, epoch2=epoch)
                    early_stop += 1

                if early_stop == early_thres:
                    sys.exit("Terminating Train")
                model.train()
                scheduler.step()
                if rank == 0:
                    save_checkpoint(args, epoch, model, adam_optimizer, scheduler,
                                    best_perform, early_stop)

        # ── epoch 末尾验证 ─────────────────────────────────
        if rank == 0:
            _run_epoch_end_eval(
                args, epoch, model, adam_optimizer, scheduler,
                inference_data_loader, eval_set, user_valid, user_train,
                usernum, itemnum, best_perform, early_stop, early_thres)

    print(f'train time: {time.time() - t0:.1f}s')
    if args.multi_gpu:
        destroy_process_group()


# ─────────────────────────────────────────────────────────────────
# Helper functions（减少重复代码）
# ─────────────────────────────────────────────────────────────────
def _reset_metrics(model, args):
    m = model.module if args.multi_gpu else model
    for attr in ('users', 'NDCG', 'HT', 'NDCG_20', 'HIT_20',
                 'NDCG_1', 'HIT_1', 'NDCG_5', 'HIT_5'):
        setattr(m, attr, 0.0)
    m.all_embs = None


def _write_results(args, model, epoch):
    out_dir = f'./models/{args.save_dir}best/'
    create_dir(out_dir)
    path = out_dir + f'{args.rec_pre_trained_data}_{args.llm}_{epoch}_results.txt'
    m = model.module if args.multi_gpu else model
    with open(path, 'a') as f:
        f.write(f'NDCG1: {m.NDCG_1/m.users:.6f}, HR1: {m.HIT_1/m.users:.6f}\n')
        f.write(f'NDCG5: {m.NDCG_5/m.users:.6f}, HR5: {m.HIT_5/m.users:.6f}\n')
        f.write(f'NDCG10: {m.NDCG/m.users:.6f}, HR10: {m.HT/m.users:.6f}\n')
        f.write(f'NDCG20: {m.NDCG_20/m.users:.6f}, HR20: {m.HIT_20/m.users:.6f}\n')


def _run_epoch_end_eval(args, epoch, model, optimizer, scheduler,
                        inference_data_loader, eval_set,
                        user_valid, user_train, usernum, itemnum,
                        best_perform, early_stop, early_thres):
    model.eval()
    _reset_metrics(model, args)

    eval_set_use = eval_set[0]
    v_users = (random.sample(list(eval_set_use), 10000)
               if len(eval_set_use) > 10000 else list(eval_set_use))
    user_list_valid = [u for u in v_users if len(user_valid[u]) >= 1]
    valid_data_set = SeqDataset_Validation(
        user_train, user_valid, user_list_valid, itemnum, args.maxlen)
    valid_data_loader = DataLoader(
        valid_data_set, batch_size=args.batch_size_infer,
        pin_memory=True, shuffle=True)

    with torch.no_grad():
        for _, vdata in enumerate(valid_data_loader):
            vu, vseq, vpos, vneg = vdata
            vu, vseq, vpos, vneg = vu.numpy(), vseq.numpy(), vpos.numpy(), vneg.numpy()
            model([vu, vseq, vpos, vneg, 0, None, 'original'], mode='generate_batch')

    m = model.module if args.multi_gpu else model
    perform = m.HT / m.users

    if perform >= best_perform:
        best_perform = perform
        m.save_model(args, epoch2=epoch, best=True)
        _reset_metrics(model, args)
        with torch.no_grad():
            for _, tdata in enumerate(inference_data_loader):
                tu, tseq, tpos, tneg = tdata
                tu, tseq, tpos, tneg = tu.numpy(), tseq.numpy(), tpos.numpy(), tneg.numpy()
                model([tu, tseq, tpos, tneg, 0, None, 'original'], mode='generate_batch')
        _write_results(args, model, epoch)
        early_stop = 0
    else:
        m.save_model(args, epoch2=epoch)
        early_stop += 1

    if early_stop == early_thres:
        sys.exit("Terminating Train")

    model.train()
    scheduler.step()
    save_checkpoint(args, epoch, model, optimizer, scheduler, best_perform, early_stop)
    _reset_metrics(model, args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()