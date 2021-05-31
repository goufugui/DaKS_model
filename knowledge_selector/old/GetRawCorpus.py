from py2neo import Graph
import xlrd
from src.Triple2Sentence import triple2sentence
import random
import json

graph = Graph('http://localhost:7474', username='neo4j')
template_table = xlrd.open_workbook('./data/Templates.xlsx').sheets()[0]
generation_corpus = open('./data/GenerationCorpus.txt', 'w', encoding='utf-8')
# generation_corpus2 = open('./data/GenerationCorpus2.txt', 'w', encoding='utf-8')
NER_corpus = open('./data/NERCorpus.txt', 'w', encoding='utf-8')
knowledge_selection_corpus = open('./data/KnowledgeSelectionCorpus.txt', 'w', encoding='utf-8')
num_false = 2
err_cnt = 0

# 保存生成的语料，用来生成负例
# question_save = []
# entity_save = []
# knowledge_save = []


def get_template_type(s):
    relation, entity = '', ''
    in_relation = False
    for c in s:
        if c == '-' and not in_relation:
            in_relation = True
            continue
        if c == '-' and in_relation:
            break
        if in_relation:
            relation += c
        elif c != '<':
            entity += c
    return entity, relation


def get_corpus(question_templates, entity_type, answer_templates, relationship):  # TODO: 换成根据图谱生成负例，效果应该会更好
    print('entity_type:', entity_type, 'relationship:', relationship)
    if len(question_templates) == 0 and len(answer_templates) == 0:
        return
    # 生成数据文件
    # print('sql:', 'MATCH ()-[r:`' + relationship + '`]->() RETURN r')
    for sql_res in graph.run('MATCH ()-[r:`' + relationship + '`]->() RETURN r'):
        start_node = [sql_res[0].start_node.labels.__str__()[1:], sql_res[0].start_node['name'].replace(',', '，').replace('\n', '').replace(' ', '')]  # 去除知识图谱中的特殊符号
        end_node = [sql_res[0].end_node.labels.__str__()[1:], sql_res[0].end_node['name'].replace(',', '，').replace('\n', '').replace(' ', '')]
        if end_node[0] == '简介':
            end_node[1] = end_node[1][:20]
        knowledge_sentence = triple2sentence(start_node[1], relationship, end_node[1])
        # 填充模板，生成语料
        for question_template in question_templates:
            question = question_template.replace('<' + start_node[0] + '>', start_node[1]).replace('<' + end_node[0] + '>', end_node[1])  # TODO: 之后可以考虑加入实体的变形
            # question_save.append(question)
            entity = ''  # TODO: 目前只处理问句中只有一个命名实体的情况
            if '<' + start_node[0] + '>' in question_template:
                entity = start_node[1]
            if '<' + end_node[0] + '>' in question_template:
                entity = end_node[1]
            NER_corpus.write(question + ',' + entity + ',1\n')
            # entity_save.append(entity)
            # knowledge_save.append(knowledge_sentence)
            if random.randint(0, 1):
                knowledge_selection_corpus.write(question + ',' + knowledge_sentence + ',1\n')
            for answer_template in answer_templates:
                answer = answer_template.replace('<' + end_node[0] + '>', end_node[1]).replace('<' + start_node[0] + '>', start_node[1])  # 这个replace顺序保证了上下句问题能正确处理
                generation_corpus.write(question + '\t' + answer + '\t' + knowledge_sentence + '\n')
                # generation_corpus2.write(question + '\t' + answer + '\t' + start_node[1] + '\t' + relationship + '\t' + end_node[1] + '\n')
        # 生成知识选择负例
        neg_centre_node = start_node if start_node[0] == entity_type else end_node
        neg_num = 0
        # if relationship == '包括':
        #     print('neg_sql:', "MATCH p=(n{name:'" + neg_centre_node[1] + "'})-[r]-() RETURN p")
        try:
            for _ in graph.run("MATCH p=(n{name:'" + neg_centre_node[1] + "'})-[r]-() RETURN p"):
                neg_num += 1
            for neg_sql_res in graph.run("MATCH p=(n{name:'" + neg_centre_node[1] + "'})-[r]->() RETURN p"):    # TODO: problem is in this line!!!
                # print('neg_sql_res:', neg_sql_res)
                neg_start_node = [neg_sql_res[0].start_node.labels.__str__()[1:],
                                  neg_sql_res[0].start_node['name'].replace(',', '，').replace('\n', '').replace(' ', '')]
                neg_end_node = [neg_sql_res[0].end_node.labels.__str__()[1:],
                                neg_sql_res[0].end_node['name'].replace(',', '，').replace('\n', '').replace(' ', '')]
                # print('start_node:', start_node[1], 'neg_start_node:', neg_start_node[1], 'end_node::', end_node, 'neg_end_node:', neg_end_node)
                in_relationship = False
                neg_relationship = ''
                # if relationship == '包括':
                #     print('neg_sql_res[0].__str__():', neg_sql_res[0])
                for c in neg_sql_res[0].__str__():  # TODO: problem is in this line!!!
                    if c == ':' and not in_relationship:
                        in_relationship = True
                        continue
                    if in_relationship:
                        if c == ' ':
                            break
                        neg_relationship += c
                neg_knowledge = triple2sentence(neg_start_node[1], neg_relationship, neg_end_node[1])
                if neg_knowledge == knowledge_sentence:
                    continue
                # if relationship == '包括':
                #     print('neg_start:', neg_start_node, 'neg_end:', neg_end_node, 'neg_relationship:', neg_relationship)
                for question_template in question_templates:
                    if random.randint(1, neg_num) <= 2:
                        question = question_template.replace('<' + start_node[0] + '>', start_node[1]).replace('<' + end_node[0] + '>', end_node[1])
                        knowledge_selection_corpus.write(question + ',' + neg_knowledge + ',0\n')
            for neg_sql_res in graph.run("MATCH p=(n{name:'" + neg_centre_node[1] + "'})<-[r]-() RETURN p"):  # 除了边的方向，其它都和上面一样
                neg_start_node = [neg_sql_res[0].start_node.labels.__str__()[1:],
                                  neg_sql_res[0].start_node['name'].replace(',', '，').replace('\n', '').replace(' ', '')]
                neg_end_node = [neg_sql_res[0].end_node.labels.__str__()[1:],
                                neg_sql_res[0].end_node['name'].replace(',', '，').replace('\n', '').replace(' ', '')]
                # if neg_start_node[1] == start_node[1] and neg_end_node[1] == end_node[1]:
                #     continue
                in_relationship = False
                neg_relationship = ''
                for c in neg_sql_res[0].__str__():
                    if c == ':' and not in_relationship:
                        in_relationship = True
                        continue
                    if in_relationship:
                        if c == ' ':
                            break
                        neg_relationship += c
                neg_knowledge = triple2sentence(neg_end_node[1], neg_relationship, neg_start_node[1])
                if neg_knowledge == knowledge_sentence:
                    continue
                # print('neg_start:', neg_start_node, 'neg_end:', neg_end_node, 'neg_relationship:', neg_relationship)
                for question_template in question_templates:
                    if random.randint(1, neg_num) <= 2:
                        question = question_template.replace('<' + start_node[0] + '>', start_node[1]).replace('<' + end_node[0] + '>', end_node[1])
                        knowledge_selection_corpus.write(question + ',' + neg_knowledge + ',0\n')
        except AttributeError:
            global err_cnt
            err_cnt += 1
            print('error!', err_cnt)


def get_pkubase_data(path):
    res = []
    f_film = open(path, 'r', encoding='utf-8')
    data = json.load(f_film)
    knowledge_save = []
    for topic in data:
        last_message = ''
        for i in range(len(topic['messages'])):
            line = topic['messages'][i]
            if 'attrs' in line:
                for attr in line['attrs']:
                    if ',' not in last_message and ',' not in attr['name'] and ',' not in attr['attrname'] and ',' not in attr['attrvalue']:  # 控制数据格式
                        res.append(last_message + ',' + attr['name'] + '的' + attr['attrname'] + '是' + attr['attrvalue'] + ',1\n')
                        knowledge_save.append(attr['name'] + '的' + attr['attrname'] + '是' + attr['attrvalue'])
            last_message = line['message']
    last_ask_knowledge = []
    last_ask = ''
    false_res = []
    for i in res:
        if last_ask and i.split(',')[0] != last_ask:
            false_num = 5
            while false_num:
                false_knowledge = knowledge_save[random.randint(0, len(knowledge_save)-1)]
                if false_knowledge in last_ask_knowledge:
                    continue
                false_res.append(last_ask + ',' + false_knowledge + ',0\n')  # 控制数据格式
                false_num -= 1
            last_ask_knowledge.clear()
        last_ask = i.split(',')[0]  # 要根据数据格式修改
        last_ask_knowledge.append(i.split(',')[1])  # 要根据数据格式修改
    return false_res + res


if __name__ == '__main__':
    # question_templates, answer_templates = [], []
    # template_type_save = '', ''
    # for row_num in range(template_table.nrows):
    #     if template_table.cell_value(row_num, 0):
    #         get_corpus(question_templates, template_type_save[0], answer_templates, template_type_save[1])
    #         question_templates.clear()
    #         answer_templates.clear()
    #         template_type_save = get_template_type(template_table.cell_value(row_num, 0))
    #     if template_table.cell_value(row_num, 1):
    #         question_templates.append(template_table.cell_value(row_num, 1))
    #         question_templates.append(template_table.cell_value(row_num, 1)[:-1])
    #     if template_table.cell_value(row_num, 2):
    #         answer_templates.append(template_table.cell_value(row_num, 2))
    #         answer_templates.append(template_table.cell_value(row_num, 2)[:-1])
    # get_corpus(question_templates, template_type_save[0], answer_templates, template_type_save[1])
    # print('正例生成完毕')

    # 知识选择添加pku_bask数据
    for i in get_pkubase_data('data/film/train.json'):
        knowledge_selection_corpus.write(i)
    for i in get_pkubase_data('data/music/train.json'):
        knowledge_selection_corpus.write(i)
    for i in get_pkubase_data('data/travel/train.json'):
        knowledge_selection_corpus.write(i)
    # print('pku_base数据添加完毕')

    # 关闭文件
    NER_corpus.close()
    knowledge_selection_corpus.close()
    generation_corpus.close()
