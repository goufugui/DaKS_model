import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import src.args as args


# 完整模型
# class KnowledgeSelector(nn.Module):
#     def __init__(self):
#         super(KnowledgeSelector, self).__init__()
#         self.bert_model = BertModel.from_pretrained(args.bert_pretrained_model_name)
#         # self.cls_classifier = nn.Linear(args.bert_hidden_size, args.num_labels)
#         self.gru = nn.GRU(args.bert_hidden_size,
#                           args.bert_hidden_size,
#                           num_layers=args.num_gru_layers,
#                           bidirectional=args.num_gru_directions==2)
#         self.joint_classifier = nn.Linear(args.bert_hidden_size + args.bert_hidden_size * 2 * args.num_gru_layers,
#                                                  args.num_labels)
#         for p in self.bert_model.parameters():
#             p.requires_grad = False
#
#     def forward(self, input_ids, token_type_ids, attention_mask):
#         bert_outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#         bert_last_hidden_states = bert_outputs.last_hidden_state.transpose(0, 1)  # [seq_len, batch_size, bert_hidden_size]
#         bert_cls_outputs = bert_last_hidden_states[0].unsqueeze(0)  # [1, batch_size, bert_hidden_size]
#         gru_outputs, gru_hidden = self.gru(bert_last_hidden_states[1:])  # output:[seq_len - 1, batch_size, num_directions * bert_hidden_size]  gru_hidden:[num_gru_layers * num_directions, batch_size, bert_hidden_size]
#         final_states = torch.cat((bert_cls_outputs, gru_hidden), 0).transpose(0, 1)  # [batch_size, 1 + num_gru_layers * num_gru_directions, bert_hidden_size]
#         final_states = final_states.reshape(-1, (1 + args.num_gru_layers * args.num_gru_directions) * args.bert_hidden_size)  # [batch_size, (1 + num_gru_layers * num_gru_directions) * bert_hidden_size]
#         return self.joint_classifier(final_states)  # [batch_size, num_labels]
#         # cls_states = last_hidden_states.transpose(0, 1)[0]  # [batch_size, args.bert_hidden_size]
#         # return self.cls_classifier(cls_states)  # [batch_size, args.num_labels]


# 没有句子级编码
# class KnowledgeSelector(nn.Module):
#     def __init__(self):
#         super(KnowledgeSelector, self).__init__()
#         self.bert_model = BertModel.from_pretrained(args.bert_pretrained_model_name)
#         # self.cls_classifier = nn.Linear(args.bert_hidden_size, args.num_labels)
#         self.gru = nn.GRU(args.bert_hidden_size,
#                           args.bert_hidden_size,
#                           num_layers=args.num_gru_layers,
#                           bidirectional=args.num_gru_directions==2)
#         self.joint_classifier = nn.Linear(args.bert_hidden_size * 2 * args.num_gru_layers,
#                                                  args.num_labels)
#         for p in self.bert_model.parameters():
#             p.requires_grad = False
#
#     def forward(self, input_ids, token_type_ids, attention_mask):
#         bert_outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#         bert_last_hidden_states = bert_outputs.last_hidden_state.transpose(0, 1)  # [seq_len, batch_size, bert_hidden_size]
#         gru_outputs, gru_hidden = self.gru(bert_last_hidden_states[1:])  # output:[seq_len - 1, batch_size, num_directions * bert_hidden_size]  gru_hidden:[num_gru_layers * num_directions, batch_size, bert_hidden_size]
#         final_states = gru_hidden.transpose(0, 1)  # [batch_size, num_gru_layers * num_gru_directions, bert_hidden_size]
#         final_states = final_states.reshape(-1, (args.num_gru_layers * args.num_gru_directions) * args.bert_hidden_size)  # [batch_size, (num_gru_layers * num_gru_directions) * bert_hidden_size]
#         return self.joint_classifier(final_states)  # [batch_size, num_labels]


# 没有Token级编码
class KnowledgeSelector(nn.Module):
    def __init__(self):
        super(KnowledgeSelector, self).__init__()
        self.bert_model = BertModel.from_pretrained(args.bert_pretrained_model_name)
        self.cls_classifier = nn.Linear(args.bert_hidden_size, args.num_labels)
        # self.gru = nn.GRU(args.bert_hidden_size,
        #                   args.bert_hidden_size,
        #                   num_layers=args.num_gru_layers,
        #                   bidirectional=args.num_gru_directions==2)
        # self.joint_classifier = nn.Linear(args.bert_hidden_size + args.bert_hidden_size * 2 * args.num_gru_layers,
        #                                          args.num_labels)
        for p in self.bert_model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_last_hidden_states = bert_outputs.last_hidden_state.transpose(0, 1)  # [seq_len, batch_size, bert_hidden_size]
        # bert_cls_outputs = bert_last_hidden_states[0].unsqueeze(0)  # [1, batch_size, bert_hidden_size]
        # gru_outputs, gru_hidden = self.gru(bert_last_hidden_states[1:])  # output:[seq_len - 1, batch_size, num_directions * bert_hidden_size]  gru_hidden:[num_gru_layers * num_directions, batch_size, bert_hidden_size]
        # final_states = torch.cat((bert_cls_outputs, gru_hidden), 0).transpose(0, 1)  # [batch_size, 1 + num_gru_layers * num_gru_directions, bert_hidden_size]
        # final_states = final_states.reshape(-1, (1 + args.num_gru_layers * args.num_gru_directions) * args.bert_hidden_size)  # [batch_size, (1 + num_gru_layers * num_gru_directions) * bert_hidden_size]
        # return self.joint_classifier(final_states)  # [batch_size, num_labels]
        cls_states = bert_last_hidden_states[0]  # [batch_size, args.bert_hidden_size]
        return self.cls_classifier(cls_states)  # [batch_size, args.num_labels]
