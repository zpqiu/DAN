# encoding: utf-8
"""
Author: zhaopeng qiu
Date: 9 Feb, 2019
"""


import os
import json
import jieba
import argparse

stopwords = None


def word_segment(text):
    words = jieba.lcut(text)
    return [word for word in words if word not in stopwords]


def is_similar(docs, doc_a):
    """
    如果两个list的Jaccard系数大于0.9, 则返回相似, 否则为不相似
    """
    for doc in docs:
        intersection = set(doc_a).intersection(set(doc))
        if len(intersection) / len(set(doc_a+doc)) > 0.9:
            return True
    return False


def parse_drop_duplicate(es_list, es_count=10):
    docs = []
    for es in es_list:
        if len(docs) == es_count:
            break
        doc = word_segment(es)
        if not is_similar(docs, doc):
            docs.append(doc)

    docs = [" ".join(doc) for doc in docs]
    return docs


def build():
    global stopwords
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--f", default="../datasets/medical/tmp_ES.txt", type=str,
                        help="Path of the ES result file.")

    args = parser.parse_args()

    stopwords = open("./stopword_punctuation.txt", "r", encoding="utf8").readlines()
    stopwords = set([word.strip() for word in stopwords])

    fr = open(args.f, "r", encoding="utf-8")

    output_file_name = ".".join(os.path.basename(args.f).split(".")[:-1]) + ".corpus"
    output_path = os.path.join(os.path.dirname(os.path.abspath(args.f)), output_file_name)

    with open(output_path, "w", encoding="utf8") as fw:
        for i, line in enumerate(fr):
            if i % 100 == 0:
                print("Has processed: {0}".format(i))
            j = json.loads(line)

            q = word_segment(j["question"])
            fw.write(" ".join(q) + "\n")

            ess = []
            for op in "ABCDE":
                op_text = word_segment(j[op])
                fw.write(" ".join(op_text) + "\n")

                es_list = [fact for fact in j["es_research_facts"]["Q+" + op]]
                ess += es_list

            docs = parse_drop_duplicate(ess)
            for doc in docs:
                fw.write(doc + "\n")


if __name__ == "__main__":
    build()
