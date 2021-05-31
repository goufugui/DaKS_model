#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/decoders/hgfu_rnn_decoder.py
"""

import torch
import torch.nn as nn

from source.modules.attention import Attention
from source.modules.decoders.state import DecoderState
from source.utils.misc import Pack
from source.utils.misc import sequence_mask


class RNNDecoder(nn.Module):
    """
    A HGFU GRU recurrent neural network decoder.
    Paper <<Towards Implicit Content-Introducing for Generative Short-Text
            Conversation Systems>>
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 attn_hidden_size=None,
                 memory_size=None,
                 feature_size=None,
                 dropout=0.0,
                 concat=False,
                 attn_knowledge= True,
                 is_select_st = True,
                 is_copy_from_knowledge = False,
                 is_copy_from_x = False):
        super(RNNDecoder, self).__init__()

        self.input_size = input_size    # 300
        self.hidden_size = hidden_size  # 800
        self.output_size = output_size  # tgt_vocab_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode     # dot
        self.attn_hidden_size = attn_hidden_size or hidden_size // 2    # 向下取整 400
        self.memory_size = memory_size or hidden_size   # 800
        self.feature_size = feature_size
        self.dropout = dropout
        self.concat = concat    # false
        self.attn_knowledge = attn_knowledge
        self.is_select_st = is_select_st
        self.is_copy_from_knowledge = is_copy_from_knowledge
        self.is_copy_from_x= is_copy_from_x

        self.rnn_input_size = self.input_size   # 300
        self.out_input_size = self.hidden_size  # 800
        self.cue_input_size = self.hidden_size  # 800

        if self.feature_size is not None:
            self.rnn_input_size += self.feature_size
            self.cue_input_size += self.feature_size

        # add
        if self.attn_knowledge:
            self.knowledge_attention = Attention(query_size=self.hidden_size,
                                                memory_size=self.hidden_size,
                                                hidden_size=self.hidden_size,
                                                mode="mlp")
        #
        # self.current_attention = Attention(query_size=self.hidden_size,
        #                                    memory_size=self.hidden_size,
        #                                    hidden_size=self.hidden_size,
        #                                    mode="mlp")

        if self.is_select_st:
            self.fc_memory = nn.Sequential(
                nn.Linear(1, self.hidden_size),
                nn.Tanh())

            self.select_st = Attention(query_size=self.hidden_size,
                                       memory_size=self.hidden_size,
                                       hidden_size=self.hidden_size,
                                       mode="mlp")

        if self.is_copy_from_knowledge:
            self.copy_from_knowledge = Attention(query_size=self.hidden_size,
                                                 memory_size=self.hidden_size,
                                                 hidden_size=self.hidden_size,
                                                 mode="mlp")

        if self.attn_mode is not None:
            # self.attention = Attention(query_size=self.hidden_size,
            #                            memory_size=self.memory_size,
            #                            hidden_size=self.attn_hidden_size,
            #                            # mode=self.attn_mode,
            #                            project=False)
            self.attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.hidden_size,
                                       hidden_size=self.hidden_size,
                                       mode="mlp")
            self.rnn_input_size += self.memory_size  # 1100
            self.cue_input_size += self.memory_size  # 1600
            self.out_input_size += self.memory_size  # 1600

        # self.history_rnn = nn.GRU(input_size=self.hidden_size*2+self.input_size,
        #                           hidden_size=self.hidden_size,
        #                           num_layers=self.num_layers,
        #                           dropout=self.dropout if self.num_layers > 1 else 0,
        #                           batch_first=True)
        #
        # self.current_rnn = nn.GRU(input_size=self.hidden_size*2+self.input_size,
        #                           hidden_size=self.hidden_size,
        #                           num_layers=self.num_layers,
        #                           dropout=self.dropout if self.num_layers > 1 else 0,
        #                           batch_first=True)

        self.rnn = nn.GRU(# input_size=self.rnn_input_size,
                          input_size=self.rnn_input_size+self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        # self.cue_rnn = nn.GRU(input_size=self.rnn_input_size,
        #                       hidden_size=self.hidden_size,
        #                       num_layers=self.num_layers,
        #                       dropout=self.dropout if self.num_layers > 1 else 0,
        #                       batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        if self.concat:
            self.fc3 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.fc3 = nn.Linear(self.hidden_size * 2, 1)

        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6 = nn.Linear(self.hidden_size*2,1)
        self.fc7 = nn.Linear(self.hidden_size,1,bias=False)
        #
        # self.fc7 = nn.Linear(self.hidden_size*2, 1)

        # self.fc8 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc9 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc10 = nn.Linear(self.hidden_size * 2, 1)


        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

        self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_size*2, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.LogSoftmax(dim=-1),
            )

        # add
        # self.output_layer_k = nn.Sequential(
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(self.hidden_size, self.output_size),
        #     nn.LogSoftmax(dim=-1),
        # )
        # if self.out_input_size > self.hidden_size:
        #     self.output_layer = nn.Sequential(
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(self.out_input_size, self.hidden_size),
        #         nn.Linear(self.hidden_size, self.output_size),
        #         nn.Softmax(dim=-1),
        #     )
        # else:
        #     self.output_layer = nn.Sequential(
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(self.out_input_size, self.output_size),
        #         nn.Softmax(dim=-1),
        #     )

        # if self.out_input_size > self.hidden_size:
        #     self.output_layer = nn.Sequential(
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(self.out_input_size, self.hidden_size),
        #         nn.Linear(self.hidden_size, self.output_size),
        #         nn.LogSoftmax(dim=-1),
        #     )
        # else:
        #     self.output_layer = nn.Sequential(
        #         nn.Dropout(p=self.dropout),
        #         nn.Linear(self.out_input_size, self.output_size),
        #         nn.LogSoftmax(dim=-1),
        #     )

    def initialize_state(self,
                         hidden,            # enc_hidden (1,batch_size,800)
                         feature=None,
                         attn_memory=None,  # enc_outputs (batch_size, src最大长度(每个时刻的输出), 800)
                         attn_mask=None,
                         memory_lengths=None,   # lengths (batch_size) 保存每个src长度
                         knowledge=None,
                         cue_outputs=None,
                         cue_mask=None,
                         cue_enc_outputs=None,
                         cue_enc_inputs=None,
                         src_outputs2=None,
                         # src_outputs=None,
                         # current_src_output=None,
                         # src_current_len=None,
                         # src_current_mask=None,
                         # src_mask=None,
                         src_len2=None,
                         src_mask2=None,
                         cue_outputs2=None,
                         cue_len2=None,
                         cue_mask2=None
                            ):
        """
        initialize_state
        """
        if self.feature_size is not None:
            assert feature is not None

        # if self.attn_mode is not None:
        #     assert attn_memory is not None

        if memory_lengths is not None and attn_mask is None:
            max_len = attn_memory.size(1)
            attn_mask = sequence_mask(memory_lengths, max_len).eq(0)

        # if src_current_len is not None and src_current_mask is None:
        #     max_len = current_src_output.size(1)
        #     src_current_mask = sequence_mask(src_current_len, max_len).eq(0)

        if src_len2 is not None and src_mask2 is None:
            max_len = src_outputs2.size(1)
            src_mask2 = sequence_mask(src_len2, max_len).eq(0)

        if cue_len2 is not None and cue_mask2 is None:
            max_len = cue_outputs2.size(1)
            cue_mask2 = sequence_mask(cue_len2, max_len).eq(0)

        init_state = DecoderState(
            hidden=hidden,
            feature=feature,
            attn_memory=attn_memory,    # (batch_size, src最大长度(每个时刻的输出), 800)
            attn_mask=attn_mask,    # (batch_size, src最大长度)
            knowledge=knowledge,    # (batch_size, 1, 800)
            cue_outputs=cue_outputs,
            cue_mask=cue_mask,
            cue_enc_outputs=cue_enc_outputs,
            cue_enc_inputs=cue_enc_inputs,
            src_outputs2=src_outputs2,
            # current_src_output=current_src_output,
            # src_current_len=src_current_len,
            # src_current_mask=src_current_mask,
            src_len2=src_len2,
            src_mask2=src_mask2,
            cue_outputs2=cue_outputs2,
            cue_len2=cue_len2,
            cue_mask2=cue_mask2
        )
        return init_state

    def decode(self, input, state, is_training=False):
        """
        decode
        """
        # 这里面 batch_size 全是 num_valid
        hidden = state.hidden   # enc_hidden (1,batch_size,800)
        rnn_input_list = []
        # cue_input_list = []
        out_input_list = []
        # history_input_list = []
        # current_input_list = []
        output = Pack()

        if self.embedder is not None:   # enc_embedder
            input = self.embedder(input)    # (batch_size,input_size(300))

        # shape: (batch_size, 1, input_size)
        input = input.unsqueeze(1)
        # rnn_input_list.append(input)
        # cue_input_list.append(state.knowledge)  # (batch_size, 1, 800)

        # history_input_list.append(input)
        # current_input_list.append(input)
        # cue_input_list.append(input)
        rnn_input_list.append(input)  ###target

        # add  知识向量，这里用的是加权知识向量，而非选出的ki
        if self.attn_knowledge:
            weighted_cue, cue_attn = self.knowledge_attention(query=hidden[-1].unsqueeze(1),
                                                              memory=state.cue_outputs2,
                                                              mask=state.cue_mask2)
            # weight_cue (batch_size, 1, 800) gt
            # cue_input_list.append(weighted_cue)
            rnn_input_list.append(weighted_cue)
            # out_input_list.append(weighted_cue)
            # cue_input_list.append(weighted_cue)
        else:
            rnn_input_list.append(state.knowledge)
            # state.knowledge (batch_size, 9, 800)
            # cue_input_list.append(state.knowledge)
            # history_input_list.append(state.knowledge)
            # current_input_list.append(state.knowledge)

        if self.feature_size is not None:
            feature = state.feature.unsqueeze(1)
            # rnn_input_list.append(feature)
            # cue_input_list.append(feature)

        if self.attn_mode is not None:  # dot   #注意力机制上下文向量c #2020.8.15单轮需要改动为输入语句而非所有
            # attn_memory = state.attn_memory  # (batch_size, src_max_len, 800)
            # attn_mask = state.attn_mask      # (batch_size, src_max_len)
            attn_memory = state.src_outputs2
            attn_mask = state.src_mask2
            query = hidden[-1].unsqueeze(1)  # (1, batch_size, 800)->(batch_size, 1, 800)
            weighted_context, attn = self.attention(query=query,
                                                    memory=attn_memory,
                                                    mask=attn_mask)
            # 对 attn_memory = enc_outputs 与hidden做 attention
            # weighted_context (batch_size, 1, 800) 加权求和后的enc_outputs
            # attn (batch_size, 1, src_max_len)  每个src的权重

            # rnn_input_list.append(weighted_context)
            # cue_input_list.append(weighted_context)
            out_input_list.append(weighted_context)
            rnn_input_list.append(weighted_context)
            # out_input_list.append(weighted_context)
            output.add(attn=attn)

        # weighted_current, _ = self.current_attention(query=hidden[-1].unsqueeze(1),
        #                                              memory=state.current_src_output,
        #                                              mask=state.src_current_mask)
        # current_input_list.append(weighted_current)

        # history_rnn_input = torch.cat(history_input_list, dim=-1)
        # history_output, history_hidden = self.history_rnn(history_rnn_input, hidden)
        # current_rnn_input = torch.cat(current_input_list, dim=-1)
        # current_output, current_hidden = self.current_rnn(current_rnn_input,hidden)
        #
        # h_history = self.tanh(self.fc8(history_hidden))
        # h_current = self.tanh(self.fc9(current_hidden))
        # p = self.sigmoid(self.fc10(torch.cat([h_history,h_current], dim=-1)))
        # new_hidden = p*h_history+(1-p)*h_current    #(1, batch_size, 800)
        # state.hidden = new_hidden

        rnn_input = torch.cat(rnn_input_list, dim=-1)
        rnn_output, rnn_hidden = self.rnn(rnn_input, hidden)  #解码GRU过程，rnn_input里有知识向量k、注意力向量c、当前状态向量、解码字（训练过程是已经给好的）
        # cue_input = torch.cat(cue_input_list, dim=-1)
        # cue_output, cue_hidden = self.cue_rnn(cue_input, hidden)
        new_hidden=rnn_hidden

        # h_y = self.tanh(self.fc1(rnn_hidden))  # (1, batch_size, 800)
        # h_cue = self.tanh(self.fc2(cue_hidden))  # (1, batch_size, 800)
        # k = self.sigmoid(self.fc3(torch.cat([h_y, h_cue], dim=-1)))  # (1, batch_size, 1)
        # new_hidden = k * h_y + (1 - k) * h_cue  # (1, batch_size, 800)
        # state.hidden = new_hidden

        if  self.is_select_st:  #论文里的HGFU状态更新机制 st = r*syt + (1-r)*skt
            # rnn_input = torch.cat(rnn_input_list, dim=-1)
            # rnn_output, rnn_hidden = self.rnn(rnn_input, hidden)
            # new_hidden = rnn_hidden
            ####毕业论文方法#####
            # memory = self.fc_memory(new_hidden.transpose(0,1).transpose(1,2))
            # weighted_st, _ = self.select_st(query=weighted_cue, #𝑠𝑡′= 𝑔𝑡𝑇𝑠𝑡
            #                                  memory=memory)
            #########2020.11.11###########
            memory = new_hidden.transpose(0, 1)
            fweight = torch.mul(self.sigmoid(self.fc4(weighted_cue)), memory)
            weight = self.softmax(fweight)
            weighted_st = torch.mul(weight,memory)
            ##############################
            # (batch_size, 1, 800)
            st = self.tanh(self.fc4(new_hidden))
            new_st = self.tanh(self.fc5(weighted_st.transpose(0,1)))
            m = self.sigmoid(self.fc6(torch.cat([st, new_st], dim=-1)))
            # new_hidden = m * new_hidden + (1 - m) * weighted_st.transpose(0, 1)
            new_hidden = m * new_hidden + (1 - m) * weighted_st.transpose(0, 1)
            # (1, batch_size, 800)

        state.hidden = new_hidden
        out_input_list.append(new_hidden.transpose(0, 1))
        out_input = torch.cat(out_input_list, dim=-1)


        if is_training:
            return out_input, state, output
            # return out_input, state, output, weighted_cue
            # return new_hidden.transpose(0,1), state, output
        else:
            log_prob = self.output_layer(out_input)  # 已知x，ki ,y[0..t-1],生成yt概率
            # log_prob = self.output_layer(new_hidden.transpose(0,1))

            # log_prob_k = self.output_layer_k(weighted_cue)

            return log_prob, state, output
            # return log_prob, state, output, log_prob_k

    def forward(self, inputs, state):
        """
        forward
        """
        # tgt[0][:, :-1], tgt[1] - 1
        inputs, lengths = inputs
        # inputs (batch_size, max_len)
        # lengths (batch_size)
        batch_size, max_len = inputs.size()  # (batch_size,max_len)

        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.hidden_size*2),    # (batch_size, max_len,1600)
            dtype=torch.float)

        # add 每一步的cue
        # cue_inputs = inputs.new_zeros(
        #     size=(batch_size, max_len, self.hidden_size),  # (batch_size, max_len, tgt_voc_size)
        #     dtype=torch.float)

        # probs = inputs.new_zeros(
        #     size=(batch_size, max_len, self.output_size),
        #     dtype = torch.float
        # )

        # hiddens = inputs.new_zeros(
        #     size=(batch_size, max_len, self.hidden_size),    # (batch_size, max_len,800)
        #     dtype=torch.float)

        # sort by lengths   排序
        sorted_lengths, indices = lengths.sort(descending=True)  # 降序
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)
        # attn_mask (batch_size, src_max_len)
        # attn_memory (batch_size, src_max_len, 800)
        # hidden (1, batch_size, 800)
        # knowledge (batch_size, 1, 800)

        # number of valid input (i.e. not padding index) in each time step
        # batch_size中 每个step有效的输入个数 [128,126，..] 127 是因为有两个tgt只有1个长度
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)  # (max_len)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]   # num_valid batch中有效的个数， i 表示第几个词
            # (num_valid)
            valid_state = state.slice_select(num_valid)     # 选state中 [:num_valid]
            out_input, valid_state, _ = self.decode(
                dec_input, valid_state, is_training=True)
            state.hidden[:, :num_valid] = valid_state.hidden
            out_inputs[:num_valid, i] = out_input.squeeze(1)  # (batch_size,1600) 第i时刻的[weighted_context,new_hidden]

            # hidden, valid_state, _ = self.decode(
            #     dec_input, valid_state, is_training=True)
            # state.hidden[:, :num_valid] = valid_state.hidden
            # hiddens[:num_valid, i] = hidden.squeeze(1)

            # cue_inputs[:num_valid, i] = cue_input.squeeze(1)


        # Resort
        _, inv_indices = indices.sort()  # indices 排序 还原decode之前的索引顺序
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)

        # cue_inputs = cue_inputs.index_select(0, inv_indices)

        # hiddens = hiddens.index_select(0, inv_indices)

        # log_probs = self.output_layer(hiddens)

        log_probs = self.output_layer(out_inputs)   # 已知x，ki ,y[0..t-1],生成yt概率

        # log_probs_k = self.output_layer_k(cue_inputs)   # P(yt|ki)
        return log_probs, state
        # return log_probs, state, log_probs_k
