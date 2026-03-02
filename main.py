import os

import utils
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

import sys
import argparse
from utils import *
from train_model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--llm", type=str, default='llama-3b',
                        help='llama, llama-3b')
    parser.add_argument("--recsys", type=str, default='sasrec')
    parser.add_argument("--rec_pre_trained_data", type=str, default='Movies_and_TV')
    parser.add_argument("--train",   action='store_true')
    parser.add_argument("--extract", action='store_true')
    parser.add_argument("--token",   action='store_true')
    parser.add_argument("--resume",  action='store_true')

    parser.add_argument("--save_dir", type=str, default='seqllm')

    parser.add_argument('--batch_size',       default=20, type=int)
    parser.add_argument('--batch_size_infer', default=20, type=int)
    parser.add_argument('--infer_epoch',      default=1,  type=int)
    parser.add_argument('--maxlen',           default=128, type=int)
    parser.add_argument('--num_epochs',       default=10,  type=int)
    parser.add_argument("--stage2_lr",        type=float, default=0.0001)
    parser.add_argument('--nn_parameter',     default=False, action='store_true')

    # ── Soft Prompt 新增参数 ────────────────────────────────
    parser.add_argument('--num_soft_prompts', default=8, type=int,
                        help='Number of learnable soft prompt tokens (k). '
                             'Set 0 to disable (equivalent to original model).')
    parser.add_argument('--ta_loss_weight',   default=0.3, type=float,
                        help='Weight for Temporal Analysis auxiliary loss. '
                             'Set 0 to disable.')
    parser.add_argument('--rps_loss_weight',  default=0.3, type=float,
                        help='Weight for Recommendation Pattern Simulating loss. '
                             'Set 0 to disable.')

    parser.add_argument('--use_lora', default=False, action='store_true')
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--lora_lr', default=5e-4, type=float)
    # ────────────────────────────────────────────────────────

    args = parser.parse_args()
    utils.set_seed(42)

    if args.device == 'hpu':
        args.device = torch.device('hpu')
    else:
        args.device = 'cuda:' + str(args.device)

    if args.train:
        train_model(args)