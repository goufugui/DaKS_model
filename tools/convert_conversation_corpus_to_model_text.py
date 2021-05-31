#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: convert_conversation_corpus_to_model_text.py
"""

import sys
import json
import collections


def preprocessing_for_one_conversation(text,
                                       topic_generalization=False,
                                       for_predict=False):
    """
    preprocessing_for_one_conversation
    """
    conversation = json.loads(text.strip(), encoding="utf-8", \
                              object_pairs_hook=collections.OrderedDict)

    goal = conversation["goal"]
    knowledge = conversation["knowledge"]
    history = conversation["history"]
    if not for_predict:
        response = conversation["response"] if "response" in conversation else "null"
        # if "response" in conversation:
        #     response = conversation["response"]
        # else:   # 测试集goal中的knowledge作为tgt
        #     response = ' '.join([' '.join(spo) for spo in goal[1:len(goal)]])

    topic_a = goal[0][1]
    topic_b = goal[0][2]
    for i, [s, p, o] in enumerate(knowledge):
        if u"领域" == p:
            if topic_a == s:
                domain_a = o
            elif topic_b == s:
                domain_b = o

    topic_dict = {}
    if u"电影" == domain_a:
        topic_dict["video_topic_a"] = topic_a
    else:
        topic_dict["person_topic_a"] = topic_a

    if u"电影" == domain_b:
        topic_dict["video_topic_b"] = topic_b
    else:
        topic_dict["person_topic_b"] = topic_b

    # chat_path_str = ' '.join([' '.join(spo) for spo in goal])
    # knowledge_str1 = ' '.join([' '.join(spo) for spo in knowledge])
    # knowledge_str2 = '\1'.join([' '.join(spo) for spo in knowledge])
    # history_str = ' '.join(history)

    chat_path_str = [' '.join([' '.join(spo) for spo in goal])]
    knowledge_str1 = ' '.join([' '.join(spo) for spo in knowledge])
    knowledge_str2 = '\1'.join([' '.join(spo) for spo in knowledge])

    history = chat_path_str + conversation["history"]
    history_str = '\1'.join(history)
    # history_str = ' '.join(history)

    src = history_str
    if not for_predict:
        model_text = '\t'.join([src, response, knowledge_str2])
    else:
        model_text = '\t'.join([src, knowledge_str2])

    if topic_generalization:
        topic_list = sorted(topic_dict.items(), key=lambda item: len(item[1]), reverse=True)
        for key, value in topic_list:
            model_text = model_text.replace(value, key)

    return model_text, topic_dict


def convert_conversation_corpus_to_model_text(corpus_file, text_file, topic_file, \
                                              topic_generalization=False):
    """
    convert_conversation_corpus_to_model_text
    """
    fout_text = open(text_file, 'w', encoding="utf-8")
    fout_topic = open(topic_file, 'w', encoding="utf-8")
    with open(corpus_file, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            model_text, topic_dict = preprocessing_for_one_conversation(
                line.strip(), topic_generalization=topic_generalization)

            topic_dict = json.dumps(topic_dict, ensure_ascii=False)

            fout_text.write(model_text + "\n")
            fout_topic.write(topic_dict + "\n")

            # session = json.loads(line.strip(), encoding="utf-8", object_pairs_hook=collections.OrderedDict)
            # for k in session["knowledge"]:
            #     fout_topic.write(topic_dict + "\n")

    fout_text.close()
    fout_topic.close()


def main():
    """
    main
    """
    convert_conversation_corpus_to_model_text(sys.argv[1],
                                              sys.argv[2],
                                              sys.argv[3],
                                              int(sys.argv[4]) > 0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
