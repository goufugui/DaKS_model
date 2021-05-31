import torch
from transformers import BertModel, BertTokenizer, BertConfig
from src.model import KnowledgeSelector
from src.utils import bert_tokenizer

if __name__ == '__main__':
    knowledge_selector = KnowledgeSelector()
    inputs = bert_tokenizer("你好", '我很好', return_tensors='pt')
    for batch in inputs.input_ids:
        for id in batch:
            print(bert_tokenizer.ids_to_tokens[id.item()])
    print('inputs:', inputs)
    print('output:', knowledge_selector(**inputs))
