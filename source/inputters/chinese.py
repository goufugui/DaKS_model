import numpy as np
import os
from cotk.wordvector import WordVector


def _load_raw_word2vec():
        raw_word2vec = {}
        with open("/home/ccnunlp/zdh_project/generative_poet_talker/Tencent_AILab_ChineseEmbedding.txt", 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    word, vec = line.split(" ", 1)
                    raw_word2vec[word] = vec
        return raw_word2vec

def load_dict(self, vocab_list):
        r'''
        Refer to :meth:`.WordVector.load_pretrain_embed`.
        '''
        raw_word2vec = self._load_raw_word2vec()

        word2vec = {}
        for vocab in vocab_list:
            str_vec = raw_word2vec.get(vocab, None)
            if str_vec is not None:
                word2vec[vocab] = np.fromstring(str_vec, sep=" ")
        return word2vec

_load_raw_word2vec()