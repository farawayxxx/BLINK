import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import re
import json
import nltk
import argparse
from tqdm import tqdm
from string import punctuation
import blink.main_dense as main_dense

wikidata5m_dir = '/home/gzk/deepfake_detection/data'

in_file = open('/home/gzk/deepfake_detection/data/grover_kws_graph_gain.jsonl', 'r',encoding='utf8')

out_file = open('/home/gzk/deepfake_detection/data/grover_gain_blink2.json', 'w', encoding= 'utf-8')
train_file = open('/home/gzk/deepfake_detection/data/grover_gain_blink_train2.json', 'w', encoding= 'utf-8')
dev_file = open('/home/gzk/deepfake_detection/data/grover_gain_blink_dev2.json', 'w', encoding= 'utf-8')
test_file = open('/home/gzk/deepfake_detection/data/grover_gain_blink_test.json', 'w', encoding= 'utf-8')

count = 0
out_line, out_line_train, out_line_dev, out_line_test = [], [], [], []

def preprocess_English(content):
    text = re.sub(r'[{}]+'.format(punctuation),'',content)
    return text

def preprocess(content):
    text = preprocess_English(content).lower().replace(' ','')
    return text

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
    print('entity num: {}'.format(len(head_cluster.keys())))
    '''
    d_ent: {'entity alias': qid}
    head_cluster: {'entity_id': [(relation_id, entity_id), ...]}
    '''
    return wiki5m_alias2qid, wiki5m_qid2alias, wiki5m_pid2alias, head_cluster, tail_cluster

# load wikidata
wiki5m_alias2qid, wiki5m_qid2alias, wiki5m_pid2alias, head_cluster, tail_cluster = load_wikidata(wikidata5m_dir)
id_list = []
# blink
models_path = "/home/gzk/deepfake_detection/BLINK/models/" # the path where you stored the BLINK models
config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": True, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}
args = argparse.Namespace(**config)
models = main_dense.load_models(args, logger=False)
print("blink model loaded")

data = in_file.readlines()
for line in tqdm(data):
    count += 1       
    line = json.loads(line)
    new_line = line 
    article = new_line['article']
    sen_idx,vertex = [], []
    keywords = line['information']['keywords']

    # get fast ner result
    mention_fast_list, mention_tag_fast_list = [], []
    for i in range(len(keywords)):
        for j in range(len(keywords[i]['keywords']['entity'])):
            mention_fast_list.append(keywords[i]['keywords']['entity'][j])  #allennlp ner
            mention_tag_fast_list.append(keywords[i]['keywords']['entity_tag'][j])
            sen_idx.append(i)
            tmp_vertex = [keywords[i]['keywords']['entity'][j],keywords[i]['keywords']['entity_tag'][j],i,0,0]
            vertex.append(tmp_vertex)

    # blink, ner and entity linking
    if mention_fast_list == [] or line['information']["graph"]["edges"] == []:
        id_list.append(count)
        mention_blink_list, entity_blink_list = [], []
    else:
        mention_blink_list, entity_blink_list = [], []
        try:
            _, _, _, _, _, mentions,predictions, _, = main_dense.run(args, None, *models, test_data=article)
            for mention in mentions:
                mention_blink_list.append(mention['text'])
            for prediction in predictions:
                entity_blink_list.append(prediction[0])
        except:
            mention_blink_list, entity_blink_list = [], []

    # get mention pos
    sens = nltk.sent_tokenize(article)
    resplit_sens = []
    sen_start_idx = [0]
    sen_idx_pair = []
    for sen in sens:
        resplit_sens+=[s.strip() for s in sen.split('\n') if s.strip()!='']
    sens = resplit_sens
    words = []
    for sen in sens:
        sen_token = nltk.word_tokenize(sen)
        sen_idx_pair.append(tuple([sen_start_idx[-1],sen_start_idx[-1]+len(sen_token)]))
        sen_start_idx.append(sen_start_idx[-1]+len(sen_token))
        words.append(sen_token)
    new_line['sens'] = words
    pos = []
    for i,node in enumerate(vertex):
        tmp_entity = nltk.word_tokenize(node[0])
        tmp_sen_idx = node[2]
        pos0, pos1 = 0,1
        len_sen = len(words[tmp_sen_idx])
        for i in range(len(tmp_entity)):
            for j in range( len_sen):
                if tmp_entity[i] == words[tmp_sen_idx][j]:
                    pos0 = j
                    pos1 = pos0 + len(tmp_entity)

                    if(pos0 == pos1):
                        print(tmp_entity)
                    break    
        if(pos0>0 and pos1>0):
            node[3] = pos0  -1
            node[4] = pos1  -1
        else:
            node[3] = pos0
            node[4] = pos1
    
    # find corresponding entity
    entity_blink_list_new = ['' for i in range(len(mention_fast_list))]
    for i in range(len(mention_fast_list)):
        for j in range(len(mention_blink_list)):  
            # if preprocess_English(mention_fast_list[i]).strip().lower() == preprocess_English(mention_blink_list[j]).strip().lower() : 
            if preprocess(mention_fast_list[i]) == preprocess(mention_blink_list[j]) :      
                entity_blink_list_new[i] = entity_blink_list[j]
    for i in range(len(mention_fast_list)):
        if entity_blink_list_new[i] == '':
            entity_blink_list_new[i] = mention_fast_list[i]

    # get entity id
    qid_fast_list = ['' for i in range(len(mention_fast_list))]
    for i in range(len(entity_blink_list_new)):
        if entity_blink_list_new[i] in wiki5m_alias2qid.keys():
            qid_fast_list[i] = wiki5m_alias2qid[entity_blink_list_new[i]]
        else:
            qid_fast_list[i] = ''
            print("entity not:", entity_blink_list_new[i])

    # relation extraction
    entity_one_list = []
    qid_one_list =[]
    for i,entity in enumerate(entity_blink_list_new):
        if entity == '':
            continue
        elif entity in entity_one_list:
            continue
        else:
            entity_one_list.append(entity)
            qid_one_list.append(qid_fast_list[i])

    relation_list = [] 
    relation_pid_list = []
    for i in range(len(entity_one_list)):     
        relation_list.append([])
        relation_pid_list.append([])      
        for j in range(len(entity_one_list)): 
            relation_list[i].append('')
            relation_pid_list[i].append('')
    for i in range(len(entity_one_list)):
        for j in range(len(entity_one_list)):
            if i == j:
                continue
            else:
                if qid_one_list[i] == '' or qid_one_list[j] == '':
                    relation_list[i][j] = ''
                    relation_pid_list[i][j] = ''
                elif qid_one_list[i] in head_cluster.keys():                  
                    for r in head_cluster[qid_one_list[i]]:
                        if r[1] == qid_one_list[j]:
                            relation_pid_list[i][j]= r[0]
                            if r[0] in wiki5m_pid2alias.keys():
                                relation_list[i][j] = wiki5m_pid2alias[r[0]]
                            else:
                                relation_list[i][j] = ''

                else:
                    relation_list[i][j] = ''
                    relation_pid_list[i][j] = ''

    # entity info 
    idx = []
    group = [[] for _ in range(len(mention_fast_list))]
    for i in range(len(mention_fast_list)):     
        entity = {}
        entity['pos'] = [0,0]
        if i in idx:
            continue
        entity['name'] = vertex[i][0]  # mention
        entity['type'] = vertex[i][1]
        entity['sent_id'] = vertex[i][2]
        entity['pos'][0] = vertex[i][3]
        entity['pos'][1] = vertex[i][4]
        
        group[i].append(entity)
        for j in range(len(mention_fast_list)):
            entity = {}
            if i == j:
                continue
            # if(preprocess_English(vertex[i][0]).strip().lower() == preprocess_English(vertex[j][0]).strip().lower()):
            if(entity_blink_list_new[i] == entity_blink_list_new[j]):
                    ans = 1
                    idx.append(j)
            else:
                ans = 0
            if (ans != 0):
                entity['name'] = vertex[j][0]
                entity['type'] = vertex[j][1]
                entity['sent_id'] = vertex[j][2]
                entity['pos'] = [0,0]
                entity['pos'][0] = vertex[j][3]
                entity['pos'][1] = vertex[j][4]

                group[i].append(entity)

    new_line['vertexSet'] = [res for res in group if res != []]
    new_line['mention_list'] = mention_fast_list
    new_line['entity_list'] = entity_blink_list_new
    new_line['entity_qid_list'] = qid_fast_list
    new_line['entity_one_list'] = entity_one_list # get mention id
    new_line['entity_one_qid_list'] = qid_one_list
    new_line['relation_list'] = relation_list
    new_line['relation_pid_list'] = relation_pid_list

    out_line.append(new_line)

    # train, valid, test split
    if(new_line['split'] == 'train'):
        out_line_train.append(new_line)
    elif(new_line['split'] == 'test'):
        out_line_test.append(new_line)
    else:
        out_line_dev.append(new_line)
    
    # if count % 100 == 0:
    #     out_file.write(json.dumps(out_line))
    #     train_file.write(json.dumps(out_line_train))
    #     dev_file.write(json.dumps(out_line_dev))
    #     test_file.write(json.dumps(out_line_test))
    #     out_line, out_line_train, out_line_dev, out_line_test = [], [], [], []
    # else:
    #     continue


out_file.write(json.dumps(out_line)+'\n')

train_file.write(json.dumps(out_line_train)+'\n')
dev_file.write(json.dumps(out_line_dev)+'\n')
test_file.write(json.dumps(out_line_test)+'\n')
print("no mention: ",id_list)