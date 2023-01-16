import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
import blink.main_dense as main_dense
import argparse
import json
import re
from tqdm import tqdm
from string import punctuation

in_file = open('/home/gzk/deepfake_detection/data/grover_kws_graph_gain_sample_100_904p_1224p_1573p.jsonl', 'r',encoding='utf8')
data = in_file.readlines()

def load_wikidata(data_dir):
    wiki5m_alias2qid, wiki5m_qid2alias = {}, {}
    with open(os.path.join(data_dir, "wikidata5m_entity.txt"), 'r',
              encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            v = line.strip().split("\t")
            if len(v) < 2:
                continue
            qid = v[0] # entity idx
            wiki5m_qid2alias[qid] = v[1:]  # {qid: ['xx', ...]}
            for alias in v[1:]: # all entity name

                wiki5m_alias2qid[alias] = qid # {'xx xxx xx': qid}

    # d_ent = wiki5m_alias2qid
    print('wikidata5m_entity.txt (Wikidata5M) loaded!')

    wiki5m_pid2alias = {}
    with open(os.path.join(data_dir, "wikidata5m_relation.txt"), 'r',
              encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            v = line.strip().split("\t")
            if len(v) < 2:
                continue
            wiki5m_pid2alias[v[0]] = v[1]
    print('wikidata5m_relation.txt (Wikidata5M) loaded!')

    head_cluster, tail_cluster = {}, {} # {'entity_id': [(relation_id, entity_id), ...]}
    total = 0
    triple_file = 'wikidata5m_all_triplet.txt'
    with open(os.path.join(data_dir, triple_file), 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            v = line.strip().split("\t")
            if len(v) != 3:
                continue
            h, r, t = v
            # if (h, r, t) not in fewrel_triples:
            if h in head_cluster:
                head_cluster[h].append((r, t))
            else:
                head_cluster[h] = [(r, t)]
            if t in tail_cluster:
                tail_cluster[t].append((r, h))
            else:
                tail_cluster[t] = [(r, h)]
            # else:
            #     num_del += 1
            total += 1
    print('wikidata5m_triplet.txt (Wikidata5M) loaded!')
    # print('deleted {} triples from Wikidata5M.'.format(num_del))

    print('entity num: {}'.format(len(head_cluster.keys())))

    '''
    d_ent: {'entity alias': qid}
    head_cluster: {'entity_id': [(relation_id, entity_id), ...]}
    '''
    return wiki5m_alias2qid, wiki5m_qid2alias, wiki5m_pid2alias, head_cluster, tail_cluster

wikidata5m_dir = '/home/gzk/deepfake_detection/data'

# get wikidata
wiki5m_alias2qid, wiki5m_qid2alias, wiki5m_pid2alias, head_cluster, tail_cluster = load_wikidata(wikidata5m_dir)

out_line = []
out_line_train = []
out_line_dev = []
out_line_test = []
new_line = {}
entity_list = []
rate_list = []
rate_list_link = []

def preprocess_English(content):
    text = re.sub(r'[{}]+'.format(punctuation),'',content)
    return text

models_path = "/home/gzk/deepfake_detection/BLINK/models/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 5,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}

args = argparse.Namespace(**config)
models = main_dense.load_models(args, logger=False)
print("loaded")

entity_len_list = []
mention_len_blink_list = []

for line in tqdm(data):
    line = json.loads(line)
    article = line['article']



    # # get entity id
    # qid_list = []
    # for i in range(len(entity_list_blink)):
    #     try:
    #         qid_list.append(wiki5m_alias2qid[entity_list_blink[i]])
    #     except:
    #         qid_list.append('')
    
    new_line = line  
    entity_list = []
    keywords = line['information']['keywords']
    for i in range(len(keywords)):
        for j in range(len(keywords[i]['keywords']['entity'])):
            entity_list.append(keywords[i]['keywords']['entity'][j])  #allennlp ner mention
    # entity linking
    if entity_list == [] or line['information']["graph"]["edges"] == []:
        mention_list_blink = []
        entity_list_blink = []
        # continue
    else:
        mention_list_blink = []
        entity_list_blink = []
        try:
            _, _, _, _, _, mentions,predictions, scores, = main_dense.run(args, None, *models, test_data=article)
            for mention in mentions:
                mention_name = mention['text']
                mention_list_blink.append(mention_name)
            for prediction in predictions:
                entity_list_blink.append(prediction[0])
        except:
            mention_list_blink = []
            entity_list_blink = []
    # 将blink entity linking 的结果与fast ner提取出的mention结果对齐
    entity_list_blink_new = ['' for _ in range(len(entity_list))]
   
    for i in range(len(entity_list)):
        for j in range(i, len(mention_list_blink)):  
            if preprocess_English(entity_list[i]).strip().lower() == preprocess_English(mention_list_blink[j]).strip().lower() :            
                entity_list_blink_new[i] = entity_list_blink[j]
                # qid_list_fast[i] = qid_list[j]
    for i in range(len(entity_list)):
        if entity_list_blink_new[i] == '':
            entity_list_blink_new[i] = entity_list[i]
            

    # get entity id
    qid_list_fast = ['' for i in range(len(entity_list))]
    for i in range(len(entity_list_blink_new)):
        # try:
        #     qid_list_fast.append(wiki5m_alias2qid[entity_list_blink_new[i]])
        # except:
        #     qid_list_fast.append('')
        if entity_list_blink_new[i] in wiki5m_alias2qid.keys():
            qid_list_fast[i] = wiki5m_alias2qid[entity_list_blink_new[i]]
        else:
            qid_list_fast[i] = ''
            print("entity not:", entity_list_blink_new[i])

    # relation extraction
    entity_list_one = []
    qid_list_one =[]
    for i,entity in enumerate(entity_list_blink_new):
        if entity == '':
            continue
        elif entity in entity_list_one:
            continue
        else:
            entity_list_one.append(entity)
            qid_list_one.append(qid_list_fast[i])

    relation_list = []  
    for i in range(len(entity_list_one)):     
        relation_list.append([])        
        for j in range(len(entity_list_one)): 
            relation_list[i].append('')

    for i in range(len(entity_list_one)):
        for j in range(len(entity_list_one)):
            if i == j:
                continue
            else:
                if qid_list_one[i] == '' or qid_list_one[j] == '':
                    relation_list[i][j] = ''
                # elif head_cluster[qid_list_one[i]]: 
                elif qid_list_one[i] in head_cluster.keys():                  
                    for r in head_cluster[qid_list_one[i]]:
                        if r[1] == qid_list_one[j]:
                            relation_list[i][j] = wiki5m_pid2alias[r[0]]
                else:
                    relation_list[i][j] = ''

    print("fast_ner: ",entity_list)
    print(entity_list_blink_new)
    print("relation: ",relation_list)

    # fast ner 与 blink entity linking 兼容比例
    count = 0
    rate = 0.0
    for i in range(len(entity_list_blink_new)):
        if entity_list_blink_new[i] == '':
            count += 1
    if len(entity_list_blink_new) == 0:
        rate = 0.0
    else:
        rate = (len(entity_list_blink_new) - count)/len(entity_list_blink_new)
    rate_list.append(rate)

    # 获取entity_id的比例
    for i in range(len(qid_list_fast)):
        if qid_list_fast[i] == '':
            count += 1
    if len(qid_list_fast) == 0:
        rate_link = 0.0
    else:
        rate_link = (len(qid_list_fast) - count)/len(qid_list_fast)
    rate_list_link.append(rate_link)

sum = 0.0

for i in range(len(rate_list)):
    sum += rate_list[i]
rate_all = sum/len(rate_list)
print('blink_rate: ',rate_all)

sum_link = 0.0
for i in range(len(rate_list_link)):
    sum_link += rate_list_link[i]
rate_link_all = sum_link/len(rate_list_link)
print('blink_link_rate',rate_link_all)




