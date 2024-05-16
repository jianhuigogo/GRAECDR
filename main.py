import argparse

from transformers import enable_full_determinism

from src.train_helper.Trainer import Trainer
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_category', type=str, default='CDs_and_Vinyl')
    parser.add_argument('--src_category', type=str, default='Movies_and_TV', choices=['Movies_and_TV', 'Books', 'CDs_and_Vinyl', 'Toys_and_Games'])
    parser.add_argument('--raw_data_path', type=str, default='/data/***/code/MPFCDR/data/raw')
    parser.add_argument('--processed_data_path', type=str, default='/data/***/code/MPFCDR/data/processed')
    parser.add_argument('--user_proportions', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--size_features', type=int, default=300)
    parser.add_argument('--feature_types', nargs='+', type=str, default=['category', 'brand', 'aspect', 'avg'])
    parser.add_argument('--transfer_types', type=str, default='group', choices=['personal', 'group', 'unify'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='visual')
    parser.add_argument('--seed', type=int, default='42')
    parser.add_argument('--rate_learning', type=float, default=1e-5)
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    enable_full_determinism(seed=args.seed)
    torch.cuda.set_device(args.local_rank)
    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'visual':
        trainer.visualize()
