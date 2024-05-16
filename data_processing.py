import argparse
import os
import pickle

from ordered_set import OrderedSet

from src.data_handler.DatasetHandler import DatasetHandler
from src.misc import init_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_category', type=str, default='CDs_and_Vinyl')
    parser.add_argument('--src_category', type=str, default='Movies_and_TV')
    parser.add_argument('--raw_data_path', type=str, default='/data/***/code/MPFCDR/data/raw')
    parser.add_argument('--processed_data_path', type=str, default='/data/***/code/MPFCDR/data/processed')
    parser.add_argument('--meta_data_path', type=str, default='/data/***/code/personalized-review-recommendation-cross-domain/data/meta_data')
    parser.add_argument('--user_proportions', type=float, default=1.0)
    args = parser.parse_args()
    processed_data_path = os.path.join(args.processed_data_path, 'tgt_{}_src_{}'.format(args.tgt_category, args.src_category))
    user_proportions_data_path = os.path.join(processed_data_path, str(int(args.user_proportions * 100)))
    args.raw_data_path = os.path.join(args.raw_data_path, '{}-{}-{}.pkl'.format(args.src_category, args.tgt_category, args.user_proportions))

    logging = init_logging(log_file=user_proportions_data_path + '/data.log', stdout=True)

    logging.info('step1:加载处理好的数据')

    udict, idict_s, idict_t, coldstart_user_set, common_user_set, _, _, _, train_common_s, train_common_t, coldstart_vali, coldstart_test, coldstart_s = pickle.load(open(args.raw_data_path, 'rb'))
    common_user_set = OrderedSet([k for k, v in udict.items() if v in common_user_set])
    coldstart_user_set = OrderedSet([k for k, v in udict.items() if v in coldstart_user_set])

    item2reviews_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'item2reviews_tgt.pkl'), 'rb'))
    user2reviews_src = pickle.load(open(os.path.join(user_proportions_data_path, 'user2reviews_src.pkl'), 'rb'))
    user2reviews_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'user2reviews_tgt.pkl'), 'rb'))
    user2items_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'user2items_tgt.pkl'), 'rb'))
    user2items_src = pickle.load(open(os.path.join(user_proportions_data_path, 'user2items_src.pkl'), 'rb'))
    user2aspects_src = pickle.load(open(os.path.join(user_proportions_data_path, 'user2aspect_src.pkl'), 'rb'))
    user2aspects_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'user2aspect_tgt.pkl'), 'rb'))
    item2aspects_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'item2aspects_tgt.pkl'), 'rb'))

    data_handler = DatasetHandler()

    logging.info('step2:用户商品购买历史抽取')
    processed_user2items_tgt = data_handler.get_user2items(user2items_tgt, common_user_set.union(coldstart_user_set), idict_t)
    processed_user2items_src = data_handler.get_user2items(user2items_src, common_user_set.union(coldstart_user_set), idict_s)

    pickle.dump(processed_user2items_tgt, open(os.path.join(user_proportions_data_path, 'processed_user2items_tgt.pkl'), 'wb'))
    pickle.dump(processed_user2items_src, open(os.path.join(user_proportions_data_path, 'processed_user2items_src.pkl'), 'wb'))

    logging.info('step3:用户购买评论中的aspect信息抽取')
    processed_user2aspects_src, aspect_vocab_src = data_handler.get_user2aspects(user2aspects_src, common_user_set.union(coldstart_user_set))
    processed_user2aspects_tgt, aspect_vocab_tgt = data_handler.get_user2aspects(user2aspects_tgt, common_user_set.union(coldstart_user_set))

    pickle.dump(processed_user2aspects_src, open(os.path.join(user_proportions_data_path, 'processed_user2aspects_src.pkl'), 'wb'))
    pickle.dump(processed_user2aspects_tgt, open(os.path.join(user_proportions_data_path, 'processed_user2aspects_tgt.pkl'), 'wb'))
    pickle.dump(aspect_vocab_src, open(os.path.join(user_proportions_data_path, 'aspect_vocab_src.pkl'), 'wb'))
    pickle.dump(aspect_vocab_tgt, open(os.path.join(user_proportions_data_path, 'aspect_vocab_tgt.pkl'), 'wb'))

    logging.info('step4:用户购买商品的category/brand信息抽取')
    args.src_meta_path = os.path.join(args.meta_data_path, 'meta_{}.json.gz'.format(args.src_category))
    args.tgt_meta_path = os.path.join(args.meta_data_path, 'meta_{}.json.gz'.format(args.tgt_category))

    processed_user2categories_src, processed_user2brand_src, category_vocab_src, brand_vocab_src = data_handler.get_user2categories_brands(user2items_src, common_user_set.union(coldstart_user_set), args.src_meta_path, idict_s)
    processed_user2categories_tgt, processed_user2brand_tgt, category_vocab_tgt, brand_vocab_tgt = data_handler.get_user2categories_brands(user2items_tgt, common_user_set.union(coldstart_user_set), args.tgt_meta_path, idict_t)

    pickle.dump(processed_user2categories_src, open(os.path.join(user_proportions_data_path, 'processed_user2categories_src.pkl'), 'wb'))
    pickle.dump(processed_user2categories_tgt, open(os.path.join(user_proportions_data_path, 'processed_user2categories_tgt.pkl'), 'wb'))
    pickle.dump(category_vocab_src, open(os.path.join(user_proportions_data_path, 'category_vocab_src.pkl'), 'wb'))
    pickle.dump(category_vocab_tgt, open(os.path.join(user_proportions_data_path, 'category_vocab_tgt.pkl'), 'wb'))

    pickle.dump(processed_user2brand_src, open(os.path.join(user_proportions_data_path, 'processed_user2brand_src.pkl'), 'wb'))
    pickle.dump(processed_user2brand_tgt, open(os.path.join(user_proportions_data_path, 'processed_user2brand_tgt.pkl'), 'wb'))
    pickle.dump(brand_vocab_src, open(os.path.join(user_proportions_data_path, 'brand_vocab_src.pkl'), 'wb'))
    pickle.dump(brand_vocab_tgt, open(os.path.join(user_proportions_data_path, 'brand_vocab_tgt.pkl'), 'wb'))

    logging.info('step4:目标域商品评论aspect信息')
    processed_item2aspects_tgt = data_handler.get_item2aspects(item2aspects_tgt, aspect_vocab_tgt)

    pickle.dump(processed_item2aspects_tgt, open(os.path.join(user_proportions_data_path, 'processed_item2aspects_tgt.pkl'), 'wb'))

    logging.info('step5:目标域商品category/brand信息')
    args.tgt_meta_path = os.path.join(args.meta_data_path, 'meta_{}.json.gz'.format(args.tgt_category))
    processed_item2categories_tgt, processed_item2brands_tgt = data_handler.get_item2categories_brands(idict_t, args.tgt_meta_path, category_vocab_tgt, brand_vocab_tgt)

    pickle.dump(processed_item2categories_tgt, open(os.path.join(user_proportions_data_path, 'processed_item2categories_tgt.pkl'), 'wb'))
    pickle.dump(processed_item2brands_tgt, open(os.path.join(user_proportions_data_path, 'processed_item2brands_tgt.pkl'), 'wb'))

    logging.info('step6:目标域商品的平均得分（训练集）')
    args.tgt_meta_reviews_path = os.path.join(args.meta_data_path, 'reviews_{}_5.json.gz'.format(args.tgt_category))
    processed_item2ratings_tgt = data_handler.get_item2ratings(args.tgt_meta_reviews_path, idict_t, coldstart_user_set)

    pickle.dump(processed_item2ratings_tgt, open(os.path.join(user_proportions_data_path, 'processed_item2ratings_tgt.pkl'), 'wb'))

    logging.info('step7:数据统计')
    logging.info('category_vocab_src:{},category_vocab_tgt:{},brand_vocab_src:{},brand_voacb_tgt:{},aspect_vocab_src:{},aspect_vocab_tgt:{}'.format(len(category_vocab_src), len(category_vocab_tgt), len(brand_vocab_src), len(brand_vocab_tgt), len(aspect_vocab_src), len(aspect_vocab_tgt)))
    logging.info('common user:{}'.format(len(common_user_set)))
    logging.info('cold start user:{}'.format(len(coldstart_user_set)))
    logging.info('item_src:{}'.format(len(idict_s)))
    logging.info('item_tgt:{}'.format(len(idict_t)))
