import argparse

from numpy import mean
from transformers import enable_full_determinism

from src.train_helper.Trainer import Trainer
import torch


def add_arguments(args):
    # ['Movies_and_TV', 'Books', 'CDs_and_Vinyl', 'Toys_and_Games']
    args.tgt_category = 'Toys_and_Games'
    args.src_category = 'Movies_and_TV'
    args.raw_data_path = '/data/***/code/MPFCDR/data/raw'
    args.processed_data_path = '/data/***/code/MPFCDR/data/processed'
    args.batch_size = 200
    args.size_features = 300
    args.num_workers = 0
    args.mode = 'eval'
    args.local_rank = 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    for user_proportions in [1.0, 0.5, 0.3]:
        args.user_proportions = user_proportions
        for transfer_types in ['group']:
            # for transfer_types in ['group']:
            args.transfer_types = transfer_types
            args.feature_types = ['category', 'brand', 'aspect', 'avg']
            if transfer_types == 'group':
                for feature_types in [['category', 'brand', 'aspect', 'avg'], ['category', 'brand', 'aspect'], ['category'], ['brand'], ['aspect'], ['category', 'aspect'], ['brand', 'aspect']]:
                    # for feature_types in [['item']]:
                    args.feature_types = feature_types
                    MAE_list = []
                    RMSE_list = []
                    print('=================================================')
                    for seed in [42, 43, 44]:
                        args.seed = seed
                        add_arguments(args)
                        enable_full_determinism(seed=args.seed)
                        torch.cuda.set_device(args.local_rank)
                        trainer = Trainer(args)
                        MAE, RMSE, valid_loss = trainer.eval()
                        MAE_list.append(MAE)
                        RMSE_list.append(RMSE)
                        print('src:{},tgt:{},feature_types:{},transfer_types:{},seed:{},user_proportions:{}'.format(args.src_category, args.tgt_category, args.feature_types, args.transfer_types, args.seed, user_proportions))
                        print('MAE:{}, RMSE:{}'.format(mean(MAE), mean(RMSE)))
                    print('Mean MAE:{}, Mean RMSE:{}'.format(mean(MAE_list), mean(RMSE_list)))

            else:
                MAE_list = []
                RMSE_list = []
                print('=================================================')
                for seed in [42, 43, 44]:
                    args.seed = seed
                    add_arguments(args)
                    enable_full_determinism(seed=args.seed)
                    trainer = Trainer(args)
                    MAE, RMSE, valid_loss = trainer.eval()
                    MAE_list.append(MAE)
                    RMSE_list.append(RMSE)
                    print('src:{},tgt:{},feature_types:{},transfer_types:{},seed:{},user_proportions:{}'.format(args.src_category, args.tgt_category, args.feature_types, args.transfer_types, args.seed, user_proportions))
                    print('MAE:{}, RMSE:{}'.format(mean(MAE), mean(RMSE)))
                print('Mean MAE:{}, Mean RMSE:{}'.format(mean(MAE_list), mean(RMSE_list)))
