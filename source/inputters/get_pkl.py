import pickle
import jieba
import re
import json
import numpy as np

word2id = {}
id2word = {}
total_cnt = {}  # 统计语料中词语出现的次数
trainingSamples = []
#['<pad>', '<unk>', '<bos>', '<eos>']
word2id['<pad>'] = 0
word2id['<unk>'] = 1
word2id['<bos>'] = 2
word2id['<eos>'] = 3
id2word[0] = '<pad>'
id2word[1] = '<unk>'
id2word[2] = '<bos>'
id2word[3] = '<eos>'
vocab={}

def get_ids(sentence):
    res = []
    words = jieba.lcut(sentence)
    for word in words:
        # if total_cnt[word] <= 0:  # 出现次数小于等于3的词语，标为<unknown> TODO: 如果要修改标为<unknown>的条件，修改这里
        #     res.append(1)
        if word in word2id:
            res.append(word2id[word])
        else:
            res.append(len(word2id))
            id2word[len(word2id)] = word
            word2id[word] = len(word2id)
    return res


def get_sentence_cnt(sentence):
    words = jieba.lcut(sentence)
    for word in words:
        if word in total_cnt:
            total_cnt[word] += 1
        else:
            total_cnt[word] = 1

def get_cnt():
    def yuliao0():
        with open('./source/inputters/data/唐诗新语料.txt', 'r', encoding='utf-8') as f:
            for line in f:
                one = re.split('\t', line)
                for i in range(len(one)):
                   get_sentence_cnt(re.split('\n', one[i])[0])


    def yuliao1():
      count = 1
      with open('./source/inputters/data/唐诗新语料.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                count = 1
                continue
            if count == 1:
                que = re.split(r'\n', line)
                get_sentence_cnt(que[0])
                count = 2
                continue
            if count == 2:
                count = 3
                anw = re.split(r'\n', line)
                get_sentence_cnt(anw[0])
                continue
            if count >= 3:
                count = count + 1
                kno = re.split(r'\n', line)
                get_sentence_cnt(kno[0])
                continue

    def yuliao2():
      count = 1
      with open('./source/inputters/data/小黄鸡辅助语料.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if "E\n" in line:
                count = 1
                continue
            if count == 1:
                que = re.split(r'M\s', line)
                get_sentence_cnt(que[1][:-1])
                count = 2
                continue
            if count == 2:
                count = 0
                anw = re.split(r'M\s', line)
                if len(anw) == 1: continue
                get_sentence_cnt(anw[1][:-1])

    def yuliao3():
      with open("./source/inputters/data/film/train.json", 'r', encoding="UTF-8") as f:
        Alllist = json.load(f)
        for x in Alllist:
            memberjson = x.get('messages')
            for i in memberjson:
                membermessage = i.get('message')
                memberknowledge = i.get('attrs')
                get_sentence_cnt(membermessage)
                if memberknowledge:
                    for j in memberknowledge:
                        knowname = j.get('name')
                        knowrelation = j.get('attrname')
                        knowvalue = j.get('attrvalue')
                        knowfinal = knowname + knowrelation + '是' + knowvalue
                    get_sentence_cnt(knowfinal)
      with open("./source/inputters/data/music/train.json", 'r', encoding="UTF-8") as f:
        Alllist = json.load(f)
        for x in Alllist:
            memberjson = x.get('messages')
            for i in memberjson:
                membermessage = i.get('message')
                memberknowledge = i.get('attrs')
                get_sentence_cnt(membermessage)
                if memberknowledge:
                    for j in memberknowledge:
                        knowname = j.get('name')
                        knowrelation = j.get('attrname')
                        knowvalue = j.get('attrvalue')
                        knowfinal = knowname + knowrelation + '是' + knowvalue
                    get_sentence_cnt(knowfinal)
      with open("./source/inputters/data/travel/train.json", 'r', encoding="UTF-8") as f:
        Alllist = json.load(f)
        for x in Alllist:
            memberjson = x.get('messages')
            for i in memberjson:
                membermessage = i.get('message')
                memberknowledge = i.get('attrs')
                get_sentence_cnt(membermessage)
                if memberknowledge:
                    for j in memberknowledge:
                        knowname = j.get('name')
                        knowrelation = j.get('attrname')
                        knowvalue = j.get('attrvalue')
                        knowfinal = knowname + knowrelation + '是' + knowvalue
                    get_sentence_cnt(knowfinal)
    yuliao0()
    #yuliao1()
    #yuliao2()
    #yuliao3()



def get_vocab():
    get_cnt()
    src={}
    tgt={}
    cue={}
    srcli=[]
    for i in id2word.values():  # 遍历字典中的值
        srcli.append(i)
    src['itos'] = srcli
    #2020.11.1加入预训练好的词向量
    word2vec = _load_raw_word2vec(srcli)
    word2veclist = []
    for j in word2vec.values():  # 遍历字典中的值
        word2veclist.append(j)
    src['embeddings'] = word2veclist
    #####2020.11.1#################
    #src['embeddings'] = None
    tgt = src
    cue = src
    vocab['src']=src
    vocab['tgt']=tgt
    vocab['cue']=cue
    return vocab


def _load_raw_word2vec(srcli):
    #从腾讯大词表里提取所需要的词向量
    #[’李白‘：[0.09992,0.499923....200维]]
    raw_word2vec = {}
    with open("/home/ccnunlp/zdh_project/generative_poet_talker/Tencent_AILab_ChineseEmbedding.txt", 'r',
              encoding='utf-8') as file:
        file.readline()
        for line in file:
            word, vec = line.split(" ", 1)
            raw_word2vec[word] = vec
    word2vec = {}
    for vocab in srcli:
        str_vec = raw_word2vec.get(vocab, None)
        if str_vec is not None:
            word2vec[vocab] = np.fromstring(str_vec, sep=" ").tolist()
        else:
            word2vec[vocab] = np.random.randn(200).tolist()
    return word2vec

def read_train_data():
    # 文件中数据格式为一行问句，一行答句
    data_raw = []
    data_test=[]
    one_dict = {}
    count = 1
    cue=[]
    with open('./source/inputters/data/唐诗新语料.txt', 'r', encoding='utf-8') as f:
        for line in f:
            one = re.split('\t', line)
            for i in range(len(one)):
                one[i]=one[i].replace(' ','')
            src = get_ids(one[0])
            tgt = get_ids(one[1])
            kno = get_ids(re.split('\n', one[2])[0])
            if len(kno) ==0:
                print("caonima")
                continue
            cue.append(kno)
            one_dict['src'] = [src]
            one_dict['tgt']=tgt
            one_dict['cue']=cue
            data_raw.append(one_dict)
            data_test.append(one_dict)
            one_dict = {}
            cue = []
    #'./source/inputters/data/李白训练语料.txt'
    # with open('./source/inputters/data/唐诗新语料.txt', 'r', encoding='utf-8') as f:
    #    for line in f:
    #        if line == '\n':
    #            count = 1
    #            one_dict['src']=[src]
    #            one_dict['tgt']=tgt
    #            one_dict['cue']=cue
    #            data_raw.append(one_dict)
    #            data_test.append(one_dict)
    #            one_dict = {}
    #            cue = []
    #            continue
    #        if count == 1:
    #            src = get_ids(re.split(r'\n', line)[0])
    #            count = 2
    #            continue
    #        if count == 2:
    #            count = 3
    #            tgt = get_ids(re.split(r'\n', line)[0])
    #            continue
    #        if count >= 3:
    #             count = count + 1
    #             kno = get_ids(re.split(r'\n', line)[0])
    #             cue.append(kno)
    #             continue
    #transxiaohuangji(data_raw)
    #trans_json(data_raw=data_raw, path="./source/inputters/data/film/train.json")
    #trans_json(data_raw=data_raw, path="./source/inputters/data/music/train.json")
    #trans_json(data_raw=data_raw, path="./source/inputters/data/travel/train.json")
    return data_raw,data_test

def trans_json(data_raw,path):
   #data_raw = []
   one_dict = {}
   src = []
   flag = 0
   with open(path, 'r', encoding="UTF-8") as f:
        Alllist = json.load(f)
        for x in Alllist:
           memberjson = x.get('messages')
           for i in memberjson:
              membermessage = i.get('message')
              if(len(membermessage)<5):
                  continue
              memberknowledge = i.get('attrs')
              t2id = get_ids(membermessage)
              if memberknowledge:
                  flag = 1
                  for j in memberknowledge:
                     knowname=j.get('name')
                     knowrelation = j.get('attrname')
                     knowvalue = j.get('attrvalue')
                     knowfinal=knowname+knowrelation+'是'+knowvalue
                  know2id=get_ids(knowfinal)
              else:
                  flag = 0

              if flag == 0:
                  src = t2id
              if flag == 1 and src!=[]:
                  one_dict['src'] = [src]
                  one_dict['tgt'] = t2id
                  one_dict['cue'] = [know2id]
                  data_raw.append(one_dict)
                  one_dict = {}
                  src = []
           src=[]
   return data_raw

def transxiaohuangji(data_raw):
    one_dict = {}
    count = 1
    src = []
    with open('./source/inputters/data/小黄鸡辅助语料.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if "E\n" in line:
                count = 1
                continue
            if count == 1:
                que = re.split(r'M\s', line)
                src = get_ids(que[1][:-1])
                count = 2
                continue
            if count == 2:
                count = 0
                anw = re.split(r'M\s', line)
                if len(anw)==1:continue
                tgt = get_ids(anw[1][:-1])
                one_dict['src'] = [src]
                one_dict['tgt'] = tgt
                one_dict['cue'] = [[1]]
                data_raw.append(one_dict)
                one_dict = {}


if __name__ == '__main__':
    get_cnt()
    data = read_train_data()
    get_vocab()
    #trans_json()
    # print('word2id:', word2id)
    # print('id2word:', id2word)
    # print('trainingSamples:', trainingSamples)
    print('dict_len: ', len(word2id))
    print('success')
