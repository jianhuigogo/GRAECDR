#!/bin/bash

src=$1
tgt=$2
gpu=$3

num_workers=12
mode=train
for user_proportions in 1.0 0.5 0.3;do
  for bz in 256;do
        for seed in 42 43 44;do
          python main.py --transfer_types='group' --mode=$mode --feature_types 'category' 'brand' 'aspect' 'avg' --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
          python main.py --transfer_types='personal' --mode=$mode --feature_types 'category' 'brand' 'aspect' 'avg' --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
#
          python main.py --transfer_types='group' --feature_types  'category' 'brand' 'aspect' --mode=$mode --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
          python main.py --transfer_types='group' --feature_types 'category'  --mode=$mode --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
          python main.py --transfer_types='group' --feature_types 'brand'  --mode=$mode --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
          python main.py --transfer_types='group' --feature_types 'aspect'  --mode=$mode --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
          python main.py --transfer_types='group' --feature_types 'category' 'aspect'  --mode=$mode --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
          python main.py --transfer_types='group' --feature_types 'brand' 'aspect'  --mode=$mode --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
#          python main.py --transfer_types='group' --feature_types 'category' 'brand'  --mode=$mode --local_rank=$gpu --seed=$seed --num_workers=$num_workers --src_category=$src --tgt_category=$tgt --user_proportions=$user_proportions
        done
  done
done
