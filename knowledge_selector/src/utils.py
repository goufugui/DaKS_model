import torch
from transformers import BertTokenizer
import csv
import random
import src.args as args

bert_tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_model_name)


def strings2ids(sentence1, sentence2, label):
    ids = bert_tokenizer(sentence1, sentence2)
    ids['label'] = int(label)
    # if random.random() > 0.5:
    #     ids = bert_tokenizer(sentence1, sentence2)
    #     # ids['label'] = int(label)
    #     ids['label'] = 0
    # else:
    #     ids = bert_tokenizer(sentence1, sentence1)
    #     ids['label'] = 1
    return ids


def get_dataset(filename):
    dataset = []
    with open(filename, 'r', encoding='utf-8') as in_file:
        csv_data = csv.reader(in_file)
        for line in csv_data:
            dataset.append(strings2ids(line[0], line[1], line[2]))
    return dataset


def get_datasets():
    train_dataset = get_dataset(args.dataset_dir + 'train_1_20.csv')
    dev_dataset = get_dataset(args.dataset_dir + '/dev_1_20.csv')
    test_dataset = get_dataset(args.dataset_dir + '/test_1_20.csv')
    return train_dataset, dev_dataset, test_dataset


def get_batch_data(dataset, batch_size=args.batch_size, shuffle=False):
    batch_data = []
    if shuffle:
        random.shuffle(dataset)
    for l in range(0, len(dataset), batch_size):
        r = min(l + batch_size, len(dataset))
        max_len = 0
        for i in range(l, r):
            max_len = max(max_len, len(dataset[i]['input_ids']))
        max_len = min(max_len, args.bert_max_len)
        batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_label = [], [], [], []
        for i in range(l, r):
            batch_input_ids.append(dataset[i]['input_ids'])
            batch_token_type_ids.append(dataset[i]['token_type_ids'])
            batch_attention_mask.append(dataset[i]['attention_mask'])
            batch_label.append(dataset[i]['label'])
            if len(dataset[i]['input_ids']) > args.bert_max_len:
                batch_input_ids[-1] = batch_input_ids[-1][: args.bert_max_len - 1]
                batch_input_ids[-1].append(bert_tokenizer.sep_token_id)
                batch_token_type_ids[-1] = batch_token_type_ids[-1][: args.bert_max_len]
                batch_attention_mask[-1] = batch_attention_mask[-1][: args.bert_max_len]
                continue
            for _ in range(max_len - len(dataset[i]['input_ids'])):
                batch_input_ids[-1].append(bert_tokenizer.pad_token_id)
                batch_token_type_ids[-1].append(1)
                batch_attention_mask[-1].append(0)
        batch_data.append({'input_ids': torch.tensor(batch_input_ids),
                           'token_type_ids': torch.tensor(batch_token_type_ids),
                           'attention_mask': torch.tensor(batch_attention_mask),
                           'label': torch.tensor(batch_label)})
    return batch_data


if __name__ == '__main__':
    print(get_datasets())
