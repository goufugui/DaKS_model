import random

input_file = open('../data/KnowledgeSelectionCorpus.csv', 'r', encoding='utf-8')  # 需要划分的语料
# train_file = open('../data/train.csv', 'w', encoding='utf-8')
# dev_file = open('../data/dev.csv', 'w', encoding='utf-8')
test_file = open('../data/test.csv', 'w', encoding='utf-8')

corpus_save = []

for line in input_file:
    corpus_save.append(line)
random.shuffle(corpus_save)
corpus_n = len(corpus_save)
for i in range(corpus_n):
    # if i < corpus_n / 100:
    # dev_file.write(corpus_save[i])
    # elif i < 2 * corpus_n / 100:
    test_file.write(corpus_save[i])
    # else:
    # train_file.write(corpus_save[i])
