#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/field.py
"""
import jieba
import re
import nltk
import torch
from tqdm import tqdm
from collections import Counter

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
NUM = "<num>"


def tokenize(s):
    """
    tokenize
    """
    s = re.sub('\d+', NUM, s).lower()   # 数字换成 NUM
    # tokens = nltk.RegexpTokenizer(r'\w+|<sil>|[^\w\s]+').tokenize(s)
    tokens = s.split(' ')
    return tokens


class Field(object):
    """
    Field
    """
    def __init__(self,
                 sequential=False,
                 dtype=None):
        self.sequential = sequential
        self.dtype = dtype if dtype is not None else int

    def str2num(self, string):
        """
        str2num
        """
        raise NotImplementedError

    def num2str(self, number):
        """
        num2str
        """
        raise NotImplementedError

    def numericalize(self, strings):    # str->num
        """
        numericalize
        """
        if isinstance(strings, str):
            return self.str2num(strings)
        else:
            return [self.numericalize(s) for s in strings]

    def denumericalize(self, numbers):   # num->str
        """
        denumericalize
        """
        if isinstance(numbers, torch.Tensor):
            with torch.cuda.device_of(numbers):
                numbers = numbers.tolist()
        if self.sequential:
            if not isinstance(numbers[0], list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]
        else:
            if not isinstance(numbers, list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]


class NumberField(Field):
    """
    NumberField
    """
    def __init__(self,
                 sequential=False,
                 dtype=None):
        super(NumberField, self).__init__(sequential=sequential,
                                          dtype=dtype)

    def str2num(self, string):
        """
        str2num
        """
        if self.sequential:
            return [self.dtype(s) for s in string.split(" ")]
        else:
            return self.dtype(string)

    def num2str(self, number):
        """
        num2str
        """
        if self.sequential:
            return " ".join([str(x) for x in number])
        else:
            return str(number)


class TextField(Field):
    """
    TextField
    """
    def __init__(self,
                 tokenize_fn=None,
                 pad_token=PAD,
                 unk_token=UNK,
                 bos_token=BOS,
                 eos_token=EOS,
                 special_tokens=None,
                 embed_file=None):
        super(TextField, self).__init__(sequential=True,
                                        dtype=int)
        # self.tokenize_fn = tokenize_fn if tokenize_fn is not None else str.split
        self.tokenize_fn = str.split
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.embed_file = embed_file

        specials = [self.pad_token, self.unk_token,
                    self.bos_token, self.eos_token]
        self.specials = [x for x in specials if x is not None]
        # ['<pad>', '<unk>', '<bos>', '<eos>']

        if special_tokens is not None:
            for token in special_tokens:
                if token not in self.specials:
                    self.specials.append(token)

        self.itos = []
        self.stoi = {}
        self.vocab_size = 0
        self.embeddings = None

    def build_vocab(self, texts, min_freq=0, max_size=None):
        """
        build_vocab
        """
        # texts
        # self.SRC : ['内容1','内容2', ...]
        # self.TGT :['目标句1','目标句2', ...]
        # self.CUE : [['(一个词)1','1'..],['2']]
        def flatten(xs):
            """
            flatten
            """

            flat_xs = []
            for x in xs:
                if isinstance(x, str):  # SRC or TGT
                    flat_xs.append(x)   # ['内容1','内容2']
                elif isinstance(x[0], str):
                    flat_xs += x        # ['(一个词)1','1', .. ,'2']
                else:
                    flat_xs += flatten(x)
            return flat_xs

        # flatten texts
        texts = flatten(texts)

        counter = Counter()
        for string in tqdm(texts):
            #tokens = self.tokenize_fn(string)   # str.split 分词
            tokens = jieba.lcut(string)
            counter.update(tokens)      # 统计每个词出现的次数
            # counter.items()  dict_items([('aa', 1), ('c', 1), ('cc', 2)])

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in self.specials:
            del counter[tok]    # 删除特殊符号

        self.itos = list(self.specials)

        if max_size is not None:
            max_size = max_size + len(self.itos)

        # sort by frequency, then alphabetically 频率 字母顺序
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])  # 以第一个元素排序 即按字母
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)    # 再以第二个元素排序 即频率
        # words_and_frequencies [('cc', 2), ('aa', 1), ('c', 1)]

        cover = 0
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:   # 频率过低或词已达到上限
                break
            # list 词
            self.itos.append(word)
            cover += freq
        cover = cover / sum(freq for _, freq in words_and_frequencies)  # 覆盖率
        print(
            "Built vocabulary of size {} (coverage: {:.3f})".format(len(self.itos), cover))

        # dict 词和词索引值
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}  # {tok:index,...}
        self.vocab_size = len(self.itos)

        if self.embed_file is not None:
            self.embeddings = self.build_word_embeddings(self.embed_file)

    def build_word_embeddings(self, embed_file):
        """
        build_word_embeddings
        """
        if isinstance(embed_file, list):
            embeds = [self.build_word_embeddings(e_file)
                      for e_file in embed_file]
        elif isinstance(embed_file, dict):
            embeds = {e_name: self.build_word_embeddings(e_file)
                      for e_name, e_file in embed_file.items()}
        else:
            cover = 0
            print("Building word embeddings from '{}' ...".format(embed_file))
            with open(embed_file, "r") as f:
                num, dim = map(int, f.readline().strip().split())
                embeds = [[0] * dim] * len(self.stoi)
                for line in f:
                    w, vs = line.rstrip().split(maxsplit=1)
                    if w in self.stoi:
                        try:
                            vs = [float(x) for x in vs.split(" ")]
                        except Exception:
                            vs = []
                        if len(vs) == dim:
                            embeds[self.stoi[w]] = vs
                            cover += 1
            rate = cover / len(embeds)
            print("{} words have pretrained {}-D word embeddings (coverage: {:.3f})".format( \
                    cover, dim, rate))
        return embeds

    def dump_vocab(self):
        """
        dump_vocab
        """
        vocab = {"itos": self.itos,
                 "embeddings": self.embeddings}
        return vocab

    def load_vocab(self, vocab):
        """
        load_vocab
        """
        # vocab {"itos": self.itos, "embeddings": self.embeddings}
        self.itos = vocab["itos"]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.embeddings = vocab["embeddings"]

    def str2num(self, string):
        """
        str2num
        """
        tokens = []
        words = jieba.lcut(string)
        unk_idx = self.stoi[self.unk_token]

        if self.bos_token:
            tokens.append(self.bos_token)

        tokens += words

        if self.eos_token:
            tokens.append(self.eos_token)
        indices = [self.stoi.get(tok, unk_idx) for tok in tokens]
        return indices

    def num2str(self, number):
        """
        num2str
        """
        tokens = [self.itos[x] for x in number]
        if tokens[0] == self.bos_token:
            tokens = tokens[1:]
        text = []
        for w in tokens:
            if w != self.eos_token:
                text.append(w)
            else:
                break
        text = [w for w in text if w not in (self.pad_token, )]
        text = " ".join(text)
        return text
