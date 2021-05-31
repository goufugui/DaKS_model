# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

import src.args as args
from src.model import KnowledgeSelector
import src.utils as utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = KnowledgeSelector().to(device)
criterion = nn.CrossEntropyLoss()
train_dataset, dev_dataset, test_dataset = utils.get_datasets()
dev_batch_dataset = utils.get_batch_data(dev_dataset, shuffle=False)
test_batch_dataset = utils.get_batch_data(test_dataset, batch_size=1, shuffle=False)
optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=(len(train_dataset)+args.batch_size-1)//args.batch_size*args.num_epochs)

all_train_losses = []
all_dev_losses = []


def train_epoch():
    model.train()
    total_loss = 0.
    start_time = time.time()
    train_batch_dataset = utils.get_batch_data(train_dataset, shuffle=True)
    for batch_id in range(1, len(train_batch_dataset)):
        optimizer.zero_grad()
        batch_data = train_batch_dataset[batch_id]
        output = model(input_ids=batch_data['input_ids'].to(device),
                       token_type_ids=batch_data['token_type_ids'].to(device),
                       attention_mask=batch_data['attention_mask'].to(device))
        loss = criterion(output, batch_data['label'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if batch_id % args.log_batch_interval == 0:
            batch_loss = total_loss / args.log_batch_interval
            all_train_losses.append(batch_loss)
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:f} | ms/batch {:5.2f} | loss {:5.5f}'.format(
                epoch_id, batch_id, len(train_batch_dataset), scheduler.get_last_lr()[0],
                (time.time() - start_time) * 1000 / args.log_batch_interval, batch_loss))
            total_loss = 0
            start_time = time.time()


def evaluate_epoch():
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_data in dev_batch_dataset:
            output = model(input_ids=batch_data['input_ids'].to(device),
                           token_type_ids=batch_data['token_type_ids'].to(device)
                           , attention_mask=batch_data['attention_mask'].to(device))
            loss = criterion(output, batch_data['label'].to(device))
            total_loss += loss.item()
    return total_loss / len(dev_batch_dataset)


def test():
    model.eval()
    num_success, num_fail = 0, 0
    with torch.no_grad():
        for batch_data in test_batch_dataset:
            output = model(input_ids=batch_data['input_ids'].to(device),
                           token_type_ids=batch_data['token_type_ids'].to(device)
                           , attention_mask=batch_data['attention_mask'].to(device))
            if abs(F.softmax(output.squeeze(0), dim=0)[1].item() - batch_data['label'][0].item()) < 0.5:
                num_success += 1
            else:
                num_fail += 1
            # print(abs(F.softmax(output.squeeze(0), dim=0)[1].item() - batch_data['label'][0].item()))
        print('num_success:', num_success)
        print('num_fail:', num_fail)
        print('success_rate', num_success / (num_success + num_fail))


if __name__ == '__main__':
    print('初始化完成，开始训练')
    best_dev_loss = float('inf')
    best_model = None

    for epoch_id in range(1, args.num_epochs + 1):
        epoch_start_time = time.time()
        if epoch_id == args.num_bert_fix_epochs + 1:
            for p in model.bert_model.parameters():
                p.requires_grad = True
        train_epoch()
        dev_loss = evaluate_epoch()
        all_dev_losses.append(dev_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time {:5.2f}s | dev loss {:5.5f}'.format(epoch_id, time.time() - epoch_start_time, dev_loss))
        print('-' * 89)

        torch.save(model.state_dict(), args.ckpt_dir + 'model_epoch_' + epoch_id.__str__() + '.ckpt')
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_model = model
    torch.save(best_model.state_dict(), args.ckpt_dir + 'best_model.ckpt')
    plt.figure()
    plt.plot([i for i in range(1, args.num_epochs * (len(train_dataset) // args.batch_size // args.log_batch_interval) + 1)], all_train_losses)
    plt.plot([i * (len(train_dataset) // args.batch_size // args.log_batch_interval) for i in range(1, args.num_epochs + 1)], all_dev_losses)
    plt.show()

    test()
