import argparse
import os
import pickle

import numpy.random
from ordered_set import OrderedSet

from src.data_handler.AspectHandler import AspectHandler
from src.data_handler.ReviewHandler import ReviewHandler
from src.misc import init_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_category', type=str, default='Toys_and_Games')
    parser.add_argument('--src_category', type=str, default='Movies_and_TV')
    parser.add_argument('--raw_data_path', type=str, default='/data/***/code/MPFCDR/data/raw')
    parser.add_argument('--meta_data_path', type=str, default='/data/***/code/personalized-review-recommendation-cross-domain/data/meta_data')
    parser.add_argument('--processed_data_path', type=str, default='/data/***/code/MPFCDR/data/processed')
    parser.add_argument('--llm_url', type=str, default='http://192.168.0.90:6000')
    parser.add_argument('--user_proportions', type=float, default=1.0)
    args = parser.parse_args()

    processed_data_path = os.path.join(args.processed_data_path, 'tgt_{}_src_{}'.format(args.tgt_category, args.src_category))
    user_proportions_data_path = os.path.join(processed_data_path, str(int(args.user_proportions * 100)))

    aspectHandler = AspectHandler(args)

    args.raw_data_path = os.path.join(args.raw_data_path, '{}-{}-{}.pkl'.format(args.src_category, args.tgt_category, args.user_proportions))

    if not os.path.exists(user_proportions_data_path):
        os.makedirs(user_proportions_data_path)
    logging = init_logging(log_file=user_proportions_data_path + '/data.log', stdout=True)

    udict, idict_s, idict_t, coldstart_user_set, common_user_set, _, _, _, train_common_s, train_common_t, coldstart_vali, coldstart_test, coldstart_s = pickle.load(open(args.raw_data_path, 'rb'))
    common_user_set = OrderedSet([k for k, v in udict.items() if v in common_user_set])
    coldstart_user_set = OrderedSet([k for k, v in udict.items() if v in coldstart_user_set])

    logging.info('step1:构建商品评论数据集，target域，挑选最具代表性的评论，排除测试用户')
    reviewHandler = ReviewHandler(args)
    item2reviews_tgt = reviewHandler.get_target_item_reviews(idict_t, coldstart_user_set)
    pickle.dump(item2reviews_tgt, open(os.path.join(user_proportions_data_path, 'item2reviews_tgt.pkl'), 'wb'))

    logging.info('step2:构建用户评论及用户商品的数据集，source域')
    user2reviews_src, user2items_src = reviewHandler.get_user_reviews(common_user_set.union(coldstart_user_set), args.src_meta_reviews_path)

    pickle.dump(user2reviews_src, open(os.path.join(user_proportions_data_path, 'user2reviews_src.pkl'), 'wb'))
    pickle.dump(user2items_src, open(os.path.join(user_proportions_data_path, 'user2items_src.pkl'), 'wb'))

    user2reviews_tgt, user2items_tgt = reviewHandler.get_user_reviews(common_user_set.union(coldstart_user_set), args.tgt_meta_reviews_path)

    pickle.dump(user2reviews_tgt, open(os.path.join(user_proportions_data_path, 'user2reviews_tgt.pkl'), 'wb'))
    pickle.dump(user2items_tgt, open(os.path.join(user_proportions_data_path, 'user2items_tgt.pkl'), 'wb'))

    logging.info('step3:抽取source域用户评论aspects')
    user2aspect_src = aspectHandler.process(user2reviews_src)
    pickle.dump(user2aspect_src, open(os.path.join(user_proportions_data_path, 'user2aspect_src.pkl'), 'wb'))
    logging.info('step4:抽取target域用户评论aspects')
    user2aspect_tgt = aspectHandler.process(user2reviews_tgt)
    pickle.dump(user2aspect_tgt, open(os.path.join(user_proportions_data_path, 'user2aspect_tgt.pkl'), 'wb'))
    logging.info('step5:抽取target域商品评论aspects')
    item2aspects_tgt = aspectHandler.process(item2reviews_tgt)
    pickle.dump(item2aspects_tgt, open(os.path.join(user_proportions_data_path, 'item2aspects_tgt.pkl'), 'wb'))
