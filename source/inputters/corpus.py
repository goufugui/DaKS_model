#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/corpus.py
"""

import os
import torch
import jieba
import json
from tqdm import tqdm
from source.inputters.field import tokenize
from source.inputters.field import TextField
from source.inputters.field import NumberField
from source.inputters.dataset import Dataset


class Corpus(object):
    """
    Corpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        prepared_data_file = data_prefix + "_" + str(max_vocab_size) + ".data.pt"
        prepared_vocab_file = data_prefix + "_" + str(max_vocab_size) + ".vocab.pt"

        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(data_dir, prepared_vocab_file)
        self.fields = {}
        self.filter_pred = None
        self.sort_fn = None
        self.data = None

    def load(self):
        """
        load
        """
        if not (os.path.exists(self.prepared_data_file) and     # ./data/demo_30000.data.pt
                os.path.exists(self.prepared_vocab_file)):      # ./data/demo_30000.vocab.pt
            self.build()                                        # 第一次load数据 进行build 并保存
        self.load_vocab(self.prepared_vocab_file)
        self.load_data(self.prepared_data_file)

        self.padding_idx = self.TGT.stoi[self.TGT.pad_token]

    def reload(self, data_type='test'):
        """
        reload
        """
        data_file = os.path.join(self.data_dir, self.data_prefix + "." + data_type)  # demo.test
        data_raw = self.read_data(data_file, data_type="test")
        # [{'src' : '...', 'tgt' : '...', 'cue': ['词1'，'词2',...]}]
        data_examples = self.build_examples(data_raw)
        # [{'src': '词索引值序列', 'tgt': '...', 'cue': ['index1'，'index2', ...]}, {}]
        self.data[data_type] = Dataset(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        load_data
        """
        # data {}
        # data['train'] data['valid'] data['test']
        # [{'src': '词索引值序列', 'tgt': '...', 'cue': ['index1'，'index2', ...]}, {}]
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        self.data = {"train": Dataset(data['train']),
                     "valid": Dataset(data["valid"]),
                     "test": Dataset(data["test"])}
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_vocab(self, prepared_vocab_file):
        """
        load_vocab
        """
        # vocab dict{}
        # vocab['src']  vocab['tgt']  vocab['cue']
        # {"itos": self.itos, "embeddings": self.embeddings}
        prepared_vocab_file = prepared_vocab_file or self.prepared_vocab_file
        print("Loading prepared vocab from {} ...".format(prepared_vocab_file))
        vocab_dict = torch.load(prepared_vocab_file)

        for name, vocab in vocab_dict.items():  # vocab {"itos": self.itos, "embeddings": self.embeddings}
            if name in self.fields:
                self.fields[name].load_vocab(vocab)
        print("Vocabulary size of fields:",
              " ".join("{}-{}".format(name.upper(), field.vocab_size) 
                for name, field in self.fields.items() 
                    if isinstance(field, TextField)))

    def read_data(self, data_file, data_type=None):
        """
        Returns
        -------
        data: ``List[Dict]``
        """
        raise NotImplementedError

    def build_vocab(self, data):    # 建立词典
        """
        Args
        ----
        data: ``List[Dict]``
        """
        # data [{'src' : '...', 'tgt' : '...', 'cue': ['词1'，'词2',...]},{ },]
        # self.fields {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE}
        field_data_dict = {}
        for name in data[0].keys():
            field = self.fields.get(name)
            if isinstance(field, TextField):
                xs = [x[name] for x in data]
                if field not in field_data_dict:
                    field_data_dict[field] = xs
                else:
                    field_data_dict[field] += xs
        # field_data_dict {self.SRC : ['内容1','内容2']，self.TGT :['',''], self.CUE : [['1','1'..],['2']]}

        # 建立三个词典 vocab_dict[src]  vocab_dict[tgt]  vocab_dict[cue]
        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)
                vocab_dict[name] = field.dump_vocab()
                # {"itos": self.itos, "embeddings": self.embeddings}
                # self.itos  list[词, ...]
        return vocab_dict

    def build_examples(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        # data [{'src': '...', 'tgt': '...', 'cue': ['词1'，'词2', ...]}, {}]
        examples = []
        for raw_data in tqdm(data):
            example = {}
            for name, strings in raw_data.items():
                if name=='src':
                    example[name] = [self.fields[name].numericalize(strings)]
                else:
                    example[name] = self.fields[name].numericalize(strings)  # token->index
            examples.append(example)
        # examples [{'src': '词索引值序列', 'tgt': '...', 'cue': ['index1'，'index2', ...]}, {}] token->index
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples
        # examples [{'src': '词索引值序列', 'tgt': '...', 'cue': ['index1'，'index2', ...]}, {}]

    def build(self):
        """
        build
        """
        print("Start to build corpus!")
        train_file = os.path.join("source/inputters/data/music/train.json")   # demo.train
        valid_file = os.path.join("source/inputters/data/music/dev.json")
        test_file = os.path.join("source/inputters/data/music/test.json")

        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")   # demo.train
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")
        # [{'src' : '...', 'tgt' : '...', 'cue': ['关系1'，'关系2',...]},{}]
        vocab = self.build_vocab(train_raw)
        # 建立三个词典 vocab[src]  vocab[tgt]  vocab[cue]
        # {"itos": self.itos, "embeddings": self.embeddings}

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        print("Building VALID examples ...")
        valid_data = self.build_examples(valid_raw)
        print("Building TEST examples ...")
        test_data = self.build_examples(test_raw)
        # [{'src': [词索引值序列], 'tgt': [...], 'cue': [[]，[], ...]}, {}]

        data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}

        print("Saving prepared vocab ...")
        torch.save(vocab, self.prepared_vocab_file)
        print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))
        # vocab dict{}
        # vocab['src']  vocab['tgt']  vocab['cue']
        # {"itos": self.itos, "embeddings": self.embeddings}

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))
        # data {}
        # data['train'] data['valid'] data['test']
        # [{'src': '词索引值序列', 'tgt': '...', 'cue': ['index1'，'index2', ...]}, {}]

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        """
        # data {}
        # data['train'] data['valid'] data['test']
        # [{'src': '词索引值序列', 'tgt': '...', 'cue': ['index1'，'index2', ...]}, {}]
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader


class SrcTgtCorpus(Corpus):
    # source target
    """
    SrcTgtCorpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None,
                 min_len=0,
                 max_len=100,
                 embed_file=None,
                 share_vocab=False):
        super(SrcTgtCorpus, self).__init__(data_dir=data_dir,
                                           data_prefix=data_prefix,
                                           min_freq=min_freq,
                                           max_vocab_size=max_vocab_size)
        self.min_len = min_len
        self.max_len = max_len
        self.share_vocab = share_vocab

        self.SRC = TextField(tokenize_fn=tokenize,
                             embed_file=embed_file)
        if self.share_vocab:
            self.TGT = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)

        self.fields = {'src': self.SRC, 'tgt': self.TGT}

        def src_filter_pred(src):
            """
            src_filter_pred
            """
            return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len

        def tgt_filter_pred(tgt):
            """
            tgt_filter_pred
            """
            return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len

        self.filter_pred = lambda ex: src_filter_pred(ex['src']) and tgt_filter_pred(ex['tgt'])

    def read_data(self, data_file, data_type="train"):
        """
        read_data
        """
        data = []
        filtered = 0
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                src, tgt = line.strip().split('\t')[:2]
                data.append({'src': src, 'tgt': tgt})

        filtered_num = len(data)
        if self.filter_pred is not None:
            data = [ex for ex in data if self.filter_pred(ex)]
        filtered_num -= len(data)
        print(
            "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data


class KnowledgeCorpus(Corpus):
    """
    KnowledgeCorpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None,
                 min_len=0,
                 max_len=100,
                 embed_file=None,
                 share_vocab=False,
                 with_label=False):
        super(KnowledgeCorpus, self).__init__(data_dir=data_dir,
                                              data_prefix=data_prefix,
                                              min_freq=min_freq,
                                              max_vocab_size=max_vocab_size)
        self.min_len = min_len
        self.max_len = max_len
        self.share_vocab = share_vocab
        self.with_label = with_label

        self.SRC = TextField(tokenize_fn=tokenize,  # 初始化
                             embed_file=embed_file)
        if self.share_vocab:
            self.TGT = self.SRC
            self.CUE = self.SRC

            self.GOAL = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)
            self.CUE = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)

            self.GOAL = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)

        if self.with_label:
            self.INDEX = NumberField()
            self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE, 'index': self.INDEX}

            # self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE, 'goal': self.GOAL, 'index': self.INDEX}
        else:
            self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE}

            # self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE, 'goal':self.GOAL}

        def src_filter_pred(src):
            """
            src_filter_pred
            """
            return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len

        def tgt_filter_pred(tgt):
            """
            tgt_filter_pred
            """
            return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len

        self.filter_pred = lambda ex: src_filter_pred(ex['src']) and tgt_filter_pred(ex['tgt'])

    def read_data(self, data_file, data_type="train"):
        """
        read_data
        """
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            datas = json.load(f)
            for dataa in datas:
                messages = dataa['messages']
                turn = []
                cue = []
                for message in messages:
                    sent = message['message']
                    turn.append(sent)

                    if 'attrs' in message:
                        know = []
                        for j in message['attrs']:
                            knowname = j.get('name')
                            knowrelation = j.get('attrname')
                            knowvalue = j.get('attrvalue')
                            knowfinal = knowname + knowrelation + '是' + knowvalue
                            know.append(knowfinal)
                        cue.append(know)
                    else:
                        cue.append([sent])
                for i in range(len(turn) - 1):
                    posts = ''
                    for j in range(max(0, (i - 5)), i + 1):
                        posts = posts + turn[j]
                    # prev_post = posts[-1]
                    response = turn[i + 1]
                    ccue = cue[i + 1]
                    data.append({'src': posts, 'tgt': response, 'cue': ccue})
        # with open(data_file, "r", encoding="utf-8") as f:
        #     for line in f:
        #         if self.with_label:
        #             src, tgt, knowledge, label = line.strip().split('\t')[:4]
        #             filter_knowledge = []
        #             for sent in knowledge.split(''):
        #                 filter_knowledge.append(' '.join(sent.split()[:self.max_len]))
        #             data.append({'src': src, 'tgt': tgt, 'cue': filter_knowledge, 'index': label})
        #         else:
        #             src, tgt, knowledge = line.strip().split('\t')[:3]
        #             # src 所有的历史信息包括话题、知识（即除了tgt以前的所有内容） tgt：机器回复的话 knowledge：知识
        #             filter_knowledge = []
        #             for sent in knowledge.split(''):   # sent 是每一个词 split 以'\x01'分割 不是空格
        #                 filter_knowledge.append(' '.join(sent.split()[:self.max_len]))  # 合并成list
        #             data.append({'src': src, 'tgt': tgt, 'cue':filter_knowledge})
        #             # [{'src' : '...', 'tgt' : '...', 'cue': ['关系1'，'关系2',...]}]
        #
        #
        #         # else:
        #         #     src, tgt, knowledge, goal = line.strip().split('\t')[:4]
        #         #     filter_knowledge = []
        #         #     for sent in knowledge.split(''):  # sent 是每一个词 split 以'\x01'分割 不是空格
        #         #         filter_knowledge.append(' '.join(sent.split()[:self.max_len]))  # 合并成list
        #         #     data.append({'src': src, 'tgt': tgt, 'cue': filter_knowledge, 'goal': goal})

        filtered_num = len(data)
        if self.filter_pred is not None:    # 如果过滤
            data = [ex for ex in data if self.filter_pred(ex)]
        filtered_num -= len(data)
        print(
            "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data
        # [{'src' : '...', 'tgt' : '...', 'cue': ['词1'，'词2',...]},{}]
