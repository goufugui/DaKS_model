num_labels = 2
dataset_dir = '../KdConv/knowledge_select_dataset/film/'
ckpt_dir = '../model/'
batch_size = 32
num_epochs = 5
num_bert_fix_epochs = 2
learning_rate = 0.00005
clip = 1.0
log_batch_interval = 100
num_gru_layers = 2
num_gru_directions = 2

# BERT 相关超参
# bert_pretrained_model_name = 'hfl/chinese-bert-wwm-ext'
bert_pretrained_model_name = 'hfl/chinese-bert-wwm'
bert_hidden_size = 768  # 需要与 bert_pretrained_model_name 对应
bert_max_len = 512
