#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/models/knowledge_seq2seq.py
"""
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.modules.decoders.hgfu_rnn_decoder import RNNDecoder
from source.utils.criterions import NLLLoss
from source.utils.misc import Pack
from source.utils.metrics import accuracy
from source.utils.metrics import attn_accuracy
from source.utils.metrics import perplexity
from source.modules.attention import Attention
from source.utils.misc import sequence_mask

#import source.modules.layers as trans_layers
from source.modules.encoders.transattenion import tattention,padding_mask

class KnowledgeSeq2Seq(BaseModel):
    """
    KnowledgeSeq2Seq
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, padding_idx=None,
                 num_layers=1, bidirectional=True, attn_mode="mlp", attn_hidden_size=None, 
                 with_bridge=False, tie_embedding=False, dropout=0.0, use_gpu=False, use_bow=False,
                 use_kd=False, use_dssm=False, use_posterior=False, weight_control=False, 
                 use_pg=False, use_gs=False, concat=False, pretrain_epoch=0):
        super(KnowledgeSeq2Seq, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size    # 300
        self.hidden_size = hidden_size  # 800
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode  # dot
        self.attn_hidden_size = attn_hidden_size
        self.with_bridge = with_bridge # with_bridge
        self.tie_embedding = tie_embedding  # true
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.use_bow = use_bow  # true
        self.use_dssm = use_dssm    # false
        self.weight_control = weight_control    # false
        self.use_kd = use_kd    # false
        self.use_pg = use_pg    # false
        self.use_gs = use_gs    # false
        self.use_posterior = use_posterior  # true
        self.pretrain_epoch = pretrain_epoch    # 5
        self.baseline = 0
        self.is_current_src = True

        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size, padding_idx=self.padding_idx)

        # add
        # self.fc = nn.Sequential(nn.Linear(self.hidden_size, int(self.hidden_size/2)), nn.Tanh())
        # self.enc_k_encoder = RNNEncoder(input_size=self.embed_size,
        #                                 hidden_size=self.hidden_size,
        #                                 embedder=enc_embedder,
        #                                 num_layers=self.num_layers,
        #                                 bidirectional=self.bidirectional,
        #                                 dropout=self.dropout)


        # self.encoder = RNNEncoder(input_size=self.embed_size,
        #                           hidden_size=self.hidden_size,
        #                           embedder=enc_embedder,
        #                           num_layers=self.num_layers,
        #                           bidirectional=self.bidirectional,
        #                           dropout=self.dropout)

        if self.with_bridge:
            self.bridge = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.Tanh())
            # 800->800 再经过一个tanh

        if self.tie_embedding:  # enc dec knowledge 三个 embedder 相同
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = enc_embedder
            knowledge_embedder = enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size, padding_idx=self.padding_idx)
            knowledge_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                          embedding_dim=self.embed_size,
                                          padding_idx=self.padding_idx)

        self.knowledge_encoder = RNNEncoder(input_size=self.embed_size,
                                            hidden_size=self.hidden_size,
                                            embedder=knowledge_embedder,
                                            num_layers=self.num_layers,
                                            bidirectional=self.bidirectional,
                                            dropout=self.dropout)

        self.knowledge_encoder2 = RNNEncoder(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size,
                                       num_layers=self.num_layers,
                                       bidirectional=self.bidirectional,
                                       dropout=self.dropout)

        # self.tgt_encoder = RNNEncoder(input_size=self.embed_size,
        #                                     hidden_size=self.hidden_size,
        #                                     embedder=knowledge_embedder,
        #                                     num_layers=self.num_layers,
        #                                     bidirectional=self.bidirectional,
        #                                     dropout=self.dropout)

        self.src_encoder = RNNEncoder(input_size=self.embed_size,
                                      hidden_size=self.hidden_size,
                                      embedder=enc_embedder,
                                      num_layers=self.num_layers,
                                      bidirectional=self.bidirectional,
                                      dropout=self.dropout)
        if self.is_current_src:
            self.src_current_encoder = RNNEncoder(input_size=self.embed_size,
                                          hidden_size=self.hidden_size,
                                          embedder=enc_embedder,
                                          num_layers=self.num_layers,
                                          bidirectional=self.bidirectional,
                                          dropout=self.dropout)


        self.src_encoder2 = RNNEncoder(input_size=self.hidden_size,
                                      hidden_size=self.hidden_size,
                                      num_layers=self.num_layers,
                                      bidirectional=self.bidirectional,
                                      dropout=self.dropout)

        self.tattention = tattention()

        self.prior_cue_attention = Attention(query_size=self.hidden_size,
                                         memory_size=self.hidden_size,
                                         hidden_size=self.hidden_size,
                                         mode="mlp")
        self.prior_src_attention = Attention(query_size=self.hidden_size,
                                             memory_size=self.hidden_size,
                                             hidden_size=self.hidden_size,
                                             mode="mlp")

        # self.posterior_cue_attention = Attention(query_size=self.hidden_size,
        #                                      memory_size=self.hidden_size,
        #                                      hidden_size=self.hidden_size,
        #                                      mode="mlp")
        # self.posterior_src_attention = Attention(query_size=self.hidden_size,
        #                                          memory_size=self.hidden_size,
        #                                          hidden_size=self.hidden_size,
        #                                          mode="mlp")

        self.decoder = RNNDecoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                  output_size=self.tgt_vocab_size, embedder=dec_embedder,
                                  num_layers=self.num_layers, attn_mode=self.attn_mode,
                                  memory_size=self.hidden_size, feature_size=None,
                                  dropout=self.dropout, concat=concat)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        if self.use_bow:
            self.bow_output_layer = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                    nn.Tanh(),
                    nn.Linear(in_features=self.hidden_size, out_features=self.tgt_vocab_size),
                    nn.LogSoftmax(dim=-1))
            # 800->800->tanh->800->tgt_vocab_size->logsoftmax

        if self.use_dssm:
            self.dssm_project = nn.Linear(in_features=self.hidden_size,
                                          out_features=self.hidden_size)
            self.mse_loss = torch.nn.MSELoss(reduction='mean')  # 均方损失误差

        if self.use_kd:
            self.knowledge_dropout = nn.Dropout()

        if self.padding_idx is not None:    # 所有token全为1 pad为0
            self.weight = torch.ones(self.tgt_vocab_size)
            self.weight[self.padding_idx] = 0
        else:
            self.weight = None
        self.nll_loss = NLLLoss(weight=self.weight, ignore_index=self.padding_idx,
                                reduction='mean')
        self.kl_loss = torch.nn.KLDivLoss(size_average=True)
        # NLLLoss, input is not restricted to a 2D Tensor
        # KLDivLoss expects a target Tensor of the same size as the input Tensor.

        if self.use_gpu:
            self.cuda()
            self.weight = self.weight.cuda()

    def encode(self, inputs, hidden=None, is_training=False):
        """
        encode
        """
        # input (Pack) {'src':( tensor([[],[] batch_size个]),tensor([] 每个length) )
        #               'tgt':( tensor([[],[] ]),tensor([]) )
        #               'cue':( tensor([ [ [],[]...个数a], [ [],[] ]...个数b,  ]), tensor([ [个数a],[],...个数b ]))}
        outputs = Pack()

        # # src
        # enc_inputs = _, lengths = inputs.src[0][:, 1:-1], inputs.src[1]-2   # 1:-1 开头有起始符 结尾有终止符 所以长度-2
        # # enc_inputs[0]: (batch_size,src_max_len)
        # # enc_inputs[1]: (batch_size)
        # enc_outputs, enc_hidden = self.encoder(enc_inputs, hidden)  # 对src进行编码
        # # enc_outputs [batch_size, src最大个数(每个时刻的输出), 800]
        # # enc_hidden [1,batch_size,800]

        # add
        # goal encode
        # goal_inputs = inputs.goal[0][:, 1:-1], inputs.goal[1] - 2
        # _, goal_enc_hidden = self.encoder(goal_inputs, hidden)

        # if self.with_bridge:
        #     enc_hidden = self.bridge(enc_hidden)    # 800->800 再经过一个tanh

            # # add
            # goal_enc_hidden = self.bridge(goal_enc_hidden)

        # src
        batch_size, src_num, src = inputs.src[0].size()
        src_len = inputs.src[1]
        src_len[src_len > 0] -= 2
        # src_current_len = src_len[:,0]
        src_inputs = inputs.src[0].view(-1, src)[:, 1:-1], src_len.view(-1)
        # size([batch_size * sent_num, sent])
        # size([batch_size * sent_num])
        src_enc_outputs, src_enc_hidden = self.src_encoder(src_inputs, hidden)
        # (batch_size*src_num, src-2, 800)
        # (1,batch_size*src_num,800)

        # current_src_output = src_enc_outputs.view(batch_size,src_num,-1, self.hidden_size)[:,0].squeeze(1)
        # (batch_size, src-2, 800)

        src_outputs = src_enc_hidden[-1].view(batch_size, src_num, -1)
        # [batch_size, src_num, 800]
        # current_src_hidden = src_outputs[:,0].unsqueeze(0)
        # (batch_size, 1, 800)
        src_mask = inputs.src[1].eq(0)

        # src_2
        src_len2 = src_enc_outputs.new_full(
            size=(batch_size,),
            fill_value = src_mask.size(1),
        dtype=torch.int64)
        src_len2 = src_len2 - src_mask.sum(dim=-1)
        src_inputs2 = src_outputs, src_len2
        #2021.5.18delete src_enc_outputs2, src_enc_hidden2 = self.src_encoder2(src_inputs2,hidden)

        # (batch, src_max_len, hidden)
        # (1, batch,hidden)

        if self.is_current_src:
            # current_src
            src_current_input = src_inputs[0].view(batch_size,src_num,-1)
            #print(src_current_input)
            try:
                src_current_input = src_current_input.gather(1, (src_len2-1).view(-1,1,1).repeat(1, 1, src_current_input.size(-1)))
            except:
                print(src_current_input)
            src_current_len = src_inputs[1].view(batch_size,-1)
            src_current_len = src_current_len.gather(1, (src_len2-1).unsqueeze(1))
            src_current_inputs =  src_current_input.squeeze(1), src_current_len.squeeze(1)
            # src_current_hidden1 = src_outputs.gather(1, (src_len2-1).view(-1,1,1).repeat(1, 1, src_outputs.size(-1)))
            # (batch, 1, hidden)
            _, src_current_hidden = self.src_current_encoder(src_current_inputs, hidden)
            # (1,batch_size,800)


        # knowledge
        batch_size, sent_num, sent = inputs.cue[0].size()   # (batch_size, sent_num, sent) sent_num 最多个knowledge个数  sent knowledge最大的长度
        tmp_len = inputs.cue[1]     # (batch_size, sent_num))每个knowledge实际长度
        tmp_len[tmp_len > 0] -= 2   # 大于0的减2
        cue_inputs = inputs.cue[0].view(-1, sent)[:, 1:-1], tmp_len.view(-1)    # 改变维度 输入到knowledge_encoder对每个知识编码
        # cue_input[0]: tensor([[],[],..]) size([batch_size*sent_num, sent])  个数长度不足补0
        # cue_input[1]: tensor([])  size([batch_size*sent_num])
        ###################2020.11.8测试
        # c = cue_inputs[1].cuda().data.cpu().numpy()
        # if 0 in c:
        #     print(inputs.cue[0])
        ###################
        # 对cue进行编码
        cue_enc_outputs, cue_enc_hidden = self.knowledge_encoder(cue_inputs, hidden)
        # cue_enc_outputs [batch_size*sent_num, sent-2, 800] batch_size*sent_num 就是一个batch中补全后知识的总个数(包括[0,0,0..]) sent-2就是一个知识长度(token个数)
        # cue_enc_hidden [1,batch_size*sent_num,800]
        #cue_enc_outputs_attention = cue_enc_outputs.view(batch_size,-1,600)

        cue_outputs = cue_enc_hidden[-1].view(batch_size, sent_num, -1) #处理成batchsize大小
        # [batch_size, sent_num, 800]
        cue_mask = inputs.cue[1].eq(0)

        # cue_2
        cue_len2  = cue_enc_outputs.new_full(
            size=(batch_size,),
            fill_value=cue_mask.size(1),
            dtype=torch.int64)
        cue_len2 = cue_len2 -cue_mask.sum(dim=-1)
        cue_inputs2 = cue_outputs, cue_len2
        cue_enc_outputs2, cue_enc_hidden2 = self.knowledge_encoder2(cue_inputs2, hidden)
        # (batch, cue_max_len, hidden)
        # (1, batch,hidden)

        #2021.5.13
        #kque = cue_inputs[0].view(batch_size, 1, -1).squeeze(1)
        kque = cue_inputs[0].unsqueeze(1).reshape(batch_size, 1, -1).squeeze(1)
        qsrc = src_inputs[0].view(batch_size, 1, -1).squeeze(1)
        attn_mask = padding_mask(kque,qsrc)
        src_crossattention_withcue_outputs = self.tattention(src_enc_outputs, cue_enc_outputs.view(batch_size,-1,600), cue_enc_outputs.view(batch_size,-1,600),attn_mask=attn_mask)
        src_enc_outputs2, src_enc_hidden2 = self.src_encoder2(src_crossattention_withcue_outputs, hidden)
        #2020.12.16 src和知识做一次attention
        # src_enc_hidden2 = self.tattention(src_enc_hidden2.transpose(0, 1), cue_enc_hidden2.transpose(0, 1), cue_enc_hidden2.transpose(0, 1))
        # src_enc_hidden2=src_enc_hidden2.transpose(0, 1)

        if self.is_current_src:
            # Attention
            max_len = cue_enc_outputs2.size(1)
            cue_mask2 = sequence_mask(cue_len2, max_len).eq(0)

            # weighted_cue, cue_attn = self.prior_attention(query=enc_hidden[-1].unsqueeze(1),
            #                                               memory=cue_outputs,
            #                                               mask=inputs.cue[1].eq(0))
            # enc_hidden (1,batch_size,800)->enc_hidden[-1] (batch_size,800) -> unsqueeze(1) (batch_size,1,800)
            # query (batch_size,1,800)
            # memory (batch_size, cue_max_num, 800)   sent_num 是一个batch最大知识个数
            # mask (batch_size, cue_max_num) 每个知识的长度 eq(0) 等于0的值为1，其他为0 也就是0的个数就是知识的个数
            # weighted_cue (batch_size, 1, 800)
            # cue_attn (batch_size, 1, sent_num)



        # # add
        # weighted_cue, cue_attn = self.prior_attention(query=goal_enc_hidden[-1].unsqueeze(1),
        #                                               memory=cue_outputs,
        #                                               mask=inputs.cue[1].eq(0))

        if self.with_bridge:
            dec_hidden = self.bridge(torch.cat([src_enc_hidden2,cue_enc_hidden2], dim=-1))
            # hidden = self.bridge(torch.cat([src_enc_hidden2,cue_enc_hidden2], dim=-1))    # 800->800 再经过一个tanh
            # if self.is_current_src:
            #    dec_hidden = self.bridge(torch.cat([weighted_src,weighted_cue], dim=-1))    # 800->800 再经过一个tanh
            #    dec_hidden = dec_hidden.transpose(0,1)
            # else:
            #    dec_hidden = self.bridge(torch.cat([src_enc_hidden2, cue_enc_hidden2], dim=-1))
            # hidden = self.bridge(torch.cat([src_enc_hidden2,weighted_cue.transpose(0,1)], dim=-1))    # 800->800 再经过一个tanh

        knowledge = cue_enc_outputs2

        # if self.use_bow:
        #     bow_logits = self.bow_output_layer(knowledge)
        #     outputs.add(bow_logits=bow_logits)

        if self.use_kd:
            knowledge = self.knowledge_dropout(knowledge)


        # add
        # new_knowledge = self.fc(knowledge.transpose(1,0))
        # x_k_outputs, x_k_hidden = self.enc_k_encoder(enc_inputs,  new_knowledge.repeat(2, 1, 1))

        dec_init_state = self.decoder.initialize_state(
            hidden=dec_hidden,  #所有源语句编码和所有知识编码
            # hidden=x_k_hidden,
            # hidden=enc_hidden,
            # attn_memory=enc_outputs if self.attn_mode else None,
            # attn_memory=x_k_outputs if self.attn_mode else None,
            # memory_lengths=lengths if self.attn_mode else None,
            knowledge=knowledge,
            cue_outputs=cue_outputs, #没有mask的知识编码向量
            cue_mask=cue_mask,
            cue_enc_outputs=cue_enc_outputs.view(batch_size,sent_num,-1,self.hidden_size),
            cue_enc_inputs=cue_inputs[0].view(batch_size, sent_num, -1),
            src_outputs2=src_enc_outputs2,
            # current_src_output=current_src_output,
            # src_current_len=src_current_len,
            src_len2=src_len2,
            cue_outputs2=cue_enc_outputs2,  #mask后的知识编码向量
            cue_len2=cue_len2
        )
        # enc_hidden (1,batch_size,800)
        # enc_outputs (batch_size, src最大长度(每个时刻的输出), 800)
        # length (batch_size) 每个src长度
        # cue_mask(batch_size, cue_max_num)  知识的个数
        # cue_enc_outputs(batch_size*sent_num, sent-2, 800)->(batch_size, sent_num, sent-2,800)
        # cue_enc_inputs (batch_size, sent_num, sent-2)
        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

        # log_prob, state, output, _= self.decoder.decode(input, state)
        # return log_prob, state, output

    def forward(self, enc_inputs, dec_inputs, hidden=None, is_training=False):
        """
        forward
        """
        outputs, dec_init_state = self.encode(
                enc_inputs, hidden, is_training=is_training)
        log_probs, _ = self.decoder(dec_inputs, dec_init_state)
        # log_probs, _, log_probs_k = self.decoder(dec_inputs, dec_init_state)
        # if torch.isnan(log_probs):
        #     1
        outputs.add(logits=log_probs)
        # outputs.add(log_probs_k=log_probs_k)
        return outputs

    def collect_metrics(self, outputs, target, cue, epoch=-1):
        """
        collect_metrics
        """
        # target (batch_size, len) 去掉BOS
        num_samples = target.size(0)    # (batch_size)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        # test begin
        # nll = self.nll(torch.log(outputs.posterior_attn+1e-10), outputs.attn_index)
        # loss += nll
        # attn_acc = attn_accuracy(outputs.posterior_attn, outputs.attn_index)
        # metrics.add(attn_acc=attn_acc)
        # metrics.add(loss=loss)
        # return metrics
        # test end


        logits = outputs.logits
        scores = -self.nll_loss(logits, target, reduction=False)
        nll_loss = self.nll_loss(logits, target)
        num_words = target.ne(self.padding_idx).sum().item()
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(nll=(nll_loss, num_words), acc=acc)

        if  self.use_bow:
            bow_logits=outputs.logits
            #bow_logits = self.bow_output_layer(bow_logits)
            bow_lables=cue
            bow_logits = bow_logits[:, :bow_lables.shape[1] + 1,]
            #bow_logits = bow_logits.repeat(1,bow_lables.size(-1),1)
            bow = self.nll_loss(bow_logits, bow_lables)
            loss += bow
            metrics.add(bow=bow)

        # add
        # log_probs_k = outputs.log_probs_k
        # nll_loss_k = self.nll_loss(log_probs_k, target)
        # acc_k = accuracy(log_probs_k, target, padding_idx=self.padding_idx)
        # metrics.add(nll_k=(nll_loss_k, num_words), acc=acc_k)


        loss += nll_loss

            # loss += nll_loss_k
        # if torch.isnan(loss):
        #     print('c')
        metrics.add(loss=loss)
        return metrics, scores

    def iterate(self, inputs, optimizer=None, grad_clip=None, is_training=False, epoch=-1):
        """
        iterate
        """
        # input.src: [0] (batch_size, src_max_len)  [1] (batch_size)
        # input.tgt: [0] (batch_size, tgt_max_len)  [1] (batch_size)
        # input.cue: [0] (batch_size, cue_max_num, cue_max_len) [1] (batch_size, cue_max_num)
        enc_inputs = inputs
        dec_inputs = inputs.tgt[0][:, :-1], inputs.tgt[1] - 1  # 去除结尾符EOS
        target = inputs.tgt[0][:, 1:]   # 去除开头符BOS 目标值
        cue = inputs.cue[0][:, 0, 1:target.shape[1]+1]
        outputs = self.forward(enc_inputs, dec_inputs, is_training=is_training)
        # prior_attn, posterior_attn, bow_logists, indexs, logits
        metrics, scores = self.collect_metrics(outputs, target, cue, epoch=epoch)

        loss = metrics.loss
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        if is_training:
            if self.use_pg:
                self.baseline = 0.99 * self.baseline + 0.01 * metrics.reward.item()
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()
        return metrics, scores
