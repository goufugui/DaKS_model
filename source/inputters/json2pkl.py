import json
import pickle
import re
import jieba


def trans_json():
    total_list_train,total_list_dev,total_list_test = [],[],[]
    i=0
    with open('data/新建文本文档.txt', 'r', encoding='utf-8') as f:
        for line in f:
            one = re.split('\t', line)
            one_dic,two_dic1,two_dic2,atr_dic,atr_list,alist ={},{},{},{},[],[]
            two_dic1["message"] = one[0]
            two_dic2["message"] = one[1]
            atr_dic["name"] = one[2]
            atr_dic["attrname"] = one[3]
            atr_dic["attrvalue"] = re.split('\n', one[4])[0]
            atr_list.append(atr_dic)
            two_dic2["attrs"]  = atr_list
            alist.append(two_dic1)
            alist.append(two_dic2)
            one_dic["messages"]=alist
            one_dic["name"]="李白"
            i += 1
            if i%99==0:
                total_list_dev.append(one_dic)
                continue
            if i%100==0:
                total_list_test.append(one_dic)
                continue
            total_list_train.append(one_dic)

    print(i)
    train_str = json.dumps(total_list_train)
    test_str = json.dumps(total_list_test)
    dev_str = json.dumps(total_list_dev)
    with open('data/train.json', 'w',encoding='utf-8') as json_file:
        json_file.write(train_str)
    with open('data/dev.json', 'w',encoding='utf-8') as json_file:
        json_file.write(dev_str)
    with open('data/test.json', 'w',encoding='utf-8') as json_file:
        json_file.write(test_str)


if __name__ == '__main__':
     trans_json()


