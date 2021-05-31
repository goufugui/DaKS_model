from py2neo import Graph
import random
import json
import csv
import time

graph = Graph('http://localhost:7474', username='neo4j')
knowledge_selection_corpus = open('../data/KnowledgeSelectionCorpus.csv', 'w', encoding='utf-8', newline='')
knowledge_selection_corpus_csv = csv.writer(knowledge_selection_corpus)
neg_num = 5
center_knowledges_save = {}


def sql_res2knowledge_sentence(sql_res):
    r = ''
    in_r = False
    reverse = False
    for c in sql_res.__str__():
        if c == '<':
            reverse = True
        if c == ':' and not in_r:
            in_r = True
            continue
        if in_r:
            if c == ' ':
                break
            else:
                r += c
    if reverse:
        return sql_res.end_node['name'] + '的' + r + '是' + sql_res.start_node['name']
    return sql_res.start_node['name'] + '的' + r + '是' + sql_res.end_node['name']


def get_pkubase_data(path):
    f_film = open(path, 'r', encoding='utf-8')
    data = json.load(f_film)
    for topic in data:
        if random.randint(0, 2) == 0:
            time.sleep(10)
        last_message = ''
        sub_knowledges = []
        if topic['name'] in center_knowledges_save:
            sub_knowledges = center_knowledges_save[topic['name']]
        else:
            for sql_res in graph.run("match (a)-[r]-() where a.name='" + topic['name'].replace('\\', '/').replace("'", "\'") + "' return DISTINCT r"):
                sub_knowledges.append(sql_res2knowledge_sentence(sql_res[0]))
            for sql_res in graph.run("match (a)-[r1]-()-[r2]-() where a.name='" + topic['name'].replace('\\', '/').replace("'", "\'") + "' return DISTINCT r2"):
                sub_knowledges.append(sql_res2knowledge_sentence(sql_res[0]))
            center_knowledges_save[topic['name']] = sub_knowledges
        for i in range(len(topic['messages'])):
            line = topic['messages'][i]
            # print(line)
            pos_knowledges, new_sub_knowledges = [], []
            if 'attrs' in line:
                for attr in line['attrs']:
                    pos_knowledge_sentence = attr['name'] + '的' + attr['attrname'] + '是' + attr['attrvalue']
                    pos_knowledges.append(pos_knowledge_sentence)
                    if pos_knowledge_sentence not in sub_knowledges:
                        print('last_message:', last_message, 'pos_knowledge_sentence:', pos_knowledge_sentence, 'sub_knowledges:', sub_knowledges)
                    if last_message:
                        knowledge_selection_corpus_csv.writerow((last_message.replace(',', '，'), pos_knowledge_sentence.replace(',', '，'), '1'))
                        # knowledge_selection_corpus.write(last_message + ',' + pos_knowledge_sentence + ',1\n')
                    if attr['name'] in center_knowledges_save:
                        new_sub_knowledges += center_knowledges_save[attr['name']]
                    else:
                        tmp = []
                        for sql_res in graph.run("match (a)-[r]-() where a.name='" + attr['name'].replace('\\', '/').replace("'", r"\'") + "' return DISTINCT r"):
                            tmp_sentence = sql_res2knowledge_sentence(sql_res[0])
                            new_sub_knowledges.append(tmp_sentence)
                            tmp.append(tmp_sentence)
                        for sql_res in graph.run("match (a)-[r1]-()-[r2]-() where a.name='" + attr['name'].replace('\\', '/').replace("'", r"\'") + "' return DISTINCT r2"):
                            tmp_sentence = sql_res2knowledge_sentence(sql_res[0])
                            new_sub_knowledges.append(tmp_sentence)
                            tmp.append(tmp_sentence)
                        center_knowledges_save[attr['name']] = tmp
                    if attr['attrvalue'] in center_knowledges_save:
                        new_sub_knowledges += center_knowledges_save[attr['attrvalue']]
                    else:
                        tmp = []
                        for sql_res in graph.run("match (a)-[r]-() where a.name='" + attr['attrvalue'].replace('\\', '/').replace("'", r"\'") + "' return DISTINCT r"):
                            tmp_sentence = sql_res2knowledge_sentence(sql_res[0])
                            new_sub_knowledges.append(tmp_sentence)
                            tmp.append(tmp_sentence)
                        for sql_res in graph.run("match (a)-[r1]-()-[r2]-() where a.name='" + attr['attrvalue'].replace('\\', '/').replace("'", r"\'") + "' return DISTINCT r2"):
                            tmp_sentence = sql_res2knowledge_sentence(sql_res[0])
                            new_sub_knowledges.append(tmp_sentence)
                            tmp.append(tmp_sentence)
                        center_knowledges_save[attr['attrvalue']] = tmp
            if last_message and sub_knowledges:
                neg_n = neg_num
                cnt = 0
                while neg_n:
                    cnt += 1
                    if cnt > neg_num * 10:
                        print('干扰知识不足')
                        break
                    rand_id = random.randint(0, len(sub_knowledges)-1)
                    neg_knowledge_sentence = sub_knowledges[rand_id]
                    if neg_knowledge_sentence in pos_knowledges:
                        continue
                    knowledge_selection_corpus_csv.writerow((last_message.replace(',', '，'), neg_knowledge_sentence.replace(',', '，'), '0'))
                    # knowledge_selection_corpus.write(last_message + ',' + neg_knowledge_sentence + ',0\n')
                    neg_n -= 1
            # last_message = line['message']
            last_message += line['message']
            if len(last_message) > 55:
                """ 根据模型最大接受输入的最大长度调整 """
                last_message = last_message[-55:]
            if 'attrs' in line:
                sub_knowledges = list(set(new_sub_knowledges))


if __name__ == '__main__':
    # 知识选择添加pku_bask数据
    # get_pkubase_data('../知识图谱/film/test.json')
    # get_pkubase_data('../知识图谱/music/test.json')
    get_pkubase_data('../知识图谱/travel/test.json')
    print('pku_base数据添加完毕')

    # 关闭文件
    knowledge_selection_corpus.close()
