import pickle
f = open('corpus.pkl', 'rb')

info = pickle.load(f)


print(info)




#添加pkl数据
#data = {0: {'gold_tuple': ('<莫妮卡·贝鲁奇>', '<代表作品>'), 'gold_relations': ['<代表作品>'], 'question': '莫妮卡·贝鲁奇的代表作？', 'sql': '{ <莫妮卡·贝鲁奇> <代表作品> ?x. }', 'answer': ['<西西里的美丽传说>'], 'gold_entitys': ['<莫妮卡·贝鲁奇>']}}
#pickle.dump(data, f2)