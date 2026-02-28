import random
import pickle
import json

import gzip
import numpy as np
from tqdm import tqdm

from collections import defaultdict
from download_amazon import ensure_amazon_2018,ensure_amazon_2023


def smart_open(path, encoding='utf-8'):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding=encoding)
    else:
        return open(path, 'r', encoding=encoding)


def preprocess_raw_5core(fname, year = 2018):
    random.seed(42)
    np.random.seed(42)

    print(f"Loading dataset for {fname}...")
    if year == 2018:
        review_path, meta_path = ensure_amazon_2018(fname)
    elif year == 2023:
        review_path, meta_path = ensure_amazon_2023(fname)
    else:
        raise ValueError("Year must be 2018 or 2023")

    # --------- 1. 读取 review 并构造 last-out ---------

    user_hist = defaultdict(list)

    with smart_open(review_path) as f:
        for line in f:
            d = json.loads(line)

            if year == 2018:
                user = d.get('reviewerID')
                item = d.get('asin')
                ts = d.get('unixReviewTime', 0)
            else:
                user = d.get('user_id')
                item = d.get('parent_asin')
                ts = d.get('timestamp', 0)

            if user and item:
                user_hist[user].append({
                    'user_id': user,
                    'parent_asin': item,
                    'timestamp': ts
                })

    dataset = {'train': [], 'valid': [], 'test': []}

    for u, seq in user_hist.items():
        seq = sorted(seq, key=lambda x: x['timestamp'])
        if len(seq) < 3:
            continue
        dataset['train'].extend(seq[:-2])
        dataset['valid'].append(seq[-2])
        dataset['test'].append(seq[-1])

    print("Using custom last-out split.")

    # --------- 2. 读取 meta ---------
    meta_dataset = {'full': []}

    with smart_open(meta_path) as f:
        for line in f:
            d = json.loads(line)

            if year == 2018:
                asin = d.get('asin')
                title = d.get('title')
                desc = d.get('description')
            else:
                asin = d.get('parent_asin')
                title = d.get('title')
                desc = d.get('description') or d.get('features')

            if asin:
                meta_dataset['full'].append({
                    'parent_asin': asin,
                    'title': title,
                    'description': desc if isinstance(desc, list) else [desc]
                })

    print("Load Meta Data")
    meta_dict = {}
    for l in tqdm(meta_dataset['full']):
        meta_dict[l['parent_asin']] = [l['title'], l['description']]
    del meta_dataset

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = defaultdict(list)
    User_s = {'train': defaultdict(list), 'valid': defaultdict(list), 'test': defaultdict(list)}
    id2asin = dict()
    time_dict = defaultdict(dict)
    for t in ['train', 'valid', 'test']:
        d = dataset[t]

        for l in tqdm(d):

            user_id = l['user_id']
            asin = l['parent_asin']

            if user_id in usermap:
                userid = usermap[user_id]
            else:
                usernum += 1
                userid = usernum
                usermap[user_id] = userid

            if asin in itemmap:
                itemid = itemmap[asin]
            else:
                itemnum += 1
                itemid = itemnum
                itemmap[asin] = itemid

            User[userid].append(itemid)
            User_s[t][userid].append(itemid)
            id2asin[itemid] = asin
            time_dict[itemid][userid] = l['timestamp']

    sample_size = int(len(User.keys()))
    print('num users raw', sample_size)

    sample_rate = {
        'Gift_Cards': 1.0,
        'Magazine_Subscriptions': 1.0,
        'All_Beauty': 1.0,
        'Prime_Pantry': 1.0,
        'Luxury_Beauty': 1.0,
        'Appliances': 1.0,
        'Amazon_Fashion': 1.0,
        'Software': 1.0,
        'Digital_Music': 0.5,
        'Musical_Instruments': 0.5,
        'Industrial_and_Scientific': 0.5,
        'Video_Games': 0.5,
        'Arts_Crafts_and_Sewing': 0.2,
        'CDs_and_Vinyl': 0.2,
        'Grocery_and_Gourmet_Food': 0.2,
        'Kindle_Store': 0.2,
        'Office_Products': 0.2,
        'Patio_Lawn_and_Garden': 0.2,
        'Pet_Supplies': 0.2,
        'Toys_and_Games': 0.2,
        'Movies_and_TV': 0.2,
        'Tools_and_Home_Improvement': 0.2,
        'Automotive': 0.1,
        'Cell_Phones_and_Accessories': 0.1,
        'Sports_and_Outdoors': 0.1,
        'Electronics': 0.05,
        'Home_and_Kitchen': 0.05,
        'Clothing_Shoes_and_Jewelry': 0.05,
        'Books': 0.02,
    }

    sample_ratio = sample_rate[fname]
    use_key = random.sample(list(User.keys()), int(sample_size * sample_ratio))

    print('num sample user', len(use_key))

    CountU = defaultdict(int)
    CountI = defaultdict(int)

    usermap_final = dict()
    itemmap_final = dict()
    usernum_final = 0
    itemnum_final = 0
    use_key_dict = defaultdict(int)
    use_train_dict = defaultdict(int)
    for key in use_key:
        use_key_dict[key] = 1

        for t in ['train', 'valid', 'test']:
            for i_ in User_s[t][key]:
                CountI[i_] += 1
                CountU[key] += 1

    text_dict = {'time': defaultdict(dict), 'description': {}, 'title': {}}
    for t in ['train', 'valid', 'test']:
        d = dataset[t]
        use_id = defaultdict(int)
        f = open(f'./../data_{fname}/{fname}_{t}.txt', 'w')
        for l in tqdm(d):

            user_id = l['user_id']
            asin = l['parent_asin']
            user_id_ = usermap[user_id]
            if use_id[user_id_] == 0:
                use_id[user_id_] = 1
                pass
            else:
                continue

            if use_key_dict[user_id_] == 1 and CountU[user_id_] > 4:

                use_items = []
                for it in User_s[t][user_id_]:
                    if CountI[it] > 4:
                        use_items.append(it)
                if t == 'train':
                    if len(use_items) > 4:
                        use_train_dict[user_id_] = 1
                        if user_id_ in usermap_final:
                            userid = usermap_final[user_id_]
                        else:
                            usernum_final += 1
                            userid = usernum_final
                            usermap_final[user_id_] = userid
                        for it in use_items:
                            if it in itemmap_final:
                                itemid = itemmap_final[it]
                            else:
                                itemnum_final += 1
                                itemid = itemnum_final
                                itemmap_final[it] = itemid

                            # ===== 方式2：先检查key存在性，再检查值 =====
                            asin_key = id2asin[it]
                            if asin_key in meta_dict:
                                d = meta_dict[asin_key][1]
                                if d == None or len(d) == 0:
                                    text_dict['description'][itemid] = 'Empty description'
                                else:
                                    text_dict['description'][itemid] = d[0]

                                texts = meta_dict[asin_key][0]
                                if texts == None or len(texts) == 0:
                                    text_dict['title'][itemid] = 'Empty title'
                                else:
                                    text_dict['title'][itemid] = texts
                            else:
                                text_dict['description'][itemid] = 'Empty description'
                                text_dict['title'][itemid] = 'Empty title'

                            text_dict['time'][itemid][userid] = time_dict[it][user_id_]
                            f.write('%d %d\n' % (userid, itemid))

                else:
                    if use_train_dict[user_id_] == 1:
                        for it in User_s[t][user_id_]:
                            if CountI[it] > 4 and it in itemmap_final:
                                if user_id_ in usermap_final:
                                    userid = usermap_final[user_id_]
                                else:
                                    usernum_final += 1
                                    userid = usernum_final
                                    usermap_final[user_id_] = userid
                                itemid = itemmap_final[it]
                                asin_key = id2asin[it]
                                if asin_key in meta_dict:
                                    d = meta_dict[asin_key][1]
                                    if d == None or len(d) == 0:
                                        text_dict['description'][itemid] = 'Empty description'
                                    else:
                                        text_dict['description'][itemid] = d[0]

                                    texts = meta_dict[asin_key][0]
                                    if texts == None or len(texts) == 0:
                                        text_dict['title'][itemid] = 'Empty title'
                                    else:
                                        text_dict['title'][itemid] = texts
                                else:
                                    text_dict['description'][itemid] = 'Empty description'
                                    text_dict['title'][itemid] = 'Empty title'

                                text_dict['time'][itemid][userid] = time_dict[it][user_id_]
                                f.write('%d %d\n' % (userid, itemid))
        f.close()
        with open(f'./../data_{fname}/text_name_dict.json.gz', 'wb') as tf:
            pickle.dump(text_dict, tf)

    del text_dict
    del meta_dict
    del dataset