# encoding: utf-8
"""
Author: zhaopeng qiu
Date: 9 Feb, 2019
      update at Apr. 16, 2019
"""


import os
import json
import jieba
import argparse


stopwords = None


def word_segment(text):
    words = jieba.lcut(text)
    return [word for word in words if word not in stopwords and word != " "]


def is_similar(docs, doc_a):
    """
    如果两个list的Jaccard系数大于0.9, 则返回相似, 否则为不相似
    """
    for doc in docs:
        intersection = set(doc_a).intersection(set(doc))
        if len(intersection) / len(set(doc_a+doc)) > 0.9:
            return True
    return False


def parse_drop_duplicate(es_list, es_count, max_len):
    """
    取前es_count个unique ES的结果作为相关文档
    :param es_list: see ES's format
    :param es_count: int, 为每个选项解析多少个相关文档
    :return: List[str],
    """
    docs = []
    for es in es_list:
        if len(docs) == es_count:
            break
        doc = word_segment(es)
        if not is_similar(docs, doc):
            docs.append(doc)
    if len(docs) < es_count:
        docs += [["PAD"], ] * (es_count-len(docs))
    docs = [" ".join(doc[:max_len]) for doc in docs]
    return docs


def parse_single_line(args, text):
    """
    Format of text: 
    {
        "id": XXX,
        "difficulty": 0.XXX,
        "question": "TEXT OF QUESTION",
        "answer": "B",
        "A": "OPTION A", "B": "OPTION B", "C": "OPTION C", "D": "OPTION D", "E": "OPTION E",
        "es_research_facts": {
            "Q+A": ["XXXX", "XXXX", ...],
            "Q+B": ["XXXX", "XXXX", ...],
            "Q+C": ["XXXX", "XXXX", ...],
            "Q+D": ["XXXX", "XXXX", ...],
            "Q+E": ["XXXX", "XXXX", ...],
        }
    }
    """
    j = json.loads(text)
    qid = j["id"]
    ques = j["question"]
    q = word_segment(ques)

    all_options = "ABCDE"
    q_options = []
    for option in all_options:
        option_text = word_segment(j[option])
        q_options.append(" ".join(truncate_seq_pair(q, option_text, 100)))

    es_count = min(args.e, 10)
    max_len = min(args.l, 100)

    option_docs = []
    for option in all_options:
        try:
            op_doc = ["[PAD]", ] * es_count
            es_list = [fact for fact in j["es_research_facts"]["Q+" + option]]
            op_doc = parse_drop_duplicate(es_list,  es_count, max_len)
            option_docs.append(op_doc)
        except Exception as ex:
            print("Error line", qid)

    ans_index = "ABCDE".index(j["answer"])
    qa = q_options[ans_index]
    other_qa = [q_options[i] for i in range(5) if i != ans_index]
    doc_set = option_docs[ans_index]
    other_doc_set = [option_docs[i] for i in range(5) if i != ans_index]

    result = {
        "qa": qa,
        "other_qa": other_qa,
        "doc_set": doc_set,
        "other_doc_set": other_doc_set,
        "qid": qid,
        "difficulty": j["difficulty"]
    }

    return json.dumps(result, ensure_ascii=False)


def truncate_seq_pair(q, option, max_length):
    if len(option) > 50:
        option = option[:50]
    truncated_q_seq = q[-1 * (max_length - len(option)):]
    return truncated_q_seq + option


def parse(args, f, output_file_name):
    with open(output_file_name, "w", encoding="utf-8") as fw:
        for i, line in enumerate(f):
            if i % 100 == 0:
                print("Has parsed: {0}".format(i))

            result = parse_single_line(args, line)
            fw.write(result + "\n")


def build():
    global stopwords
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--f", default="../datasets/medical/tmp_ES.txt", type=str,
                        help="Path of the ES result file.")
    parser.add_argument("--o", default="", type=str,
                        help="Path of the output file.")
    parser.add_argument("--e", default=1, type=int,
                        help="The count of ES of each QA pair. The Maximum is 10")
    parser.add_argument("--l", default=100, type=int,
                        help="The max length of sequences")

    args = parser.parse_args()

    stopwords = open("./stopword_punctuation.txt", "r", encoding="utf8").readlines()
    stopwords = set([word.strip() for word in stopwords])

    fr = open(args.f, "r", encoding="utf-8")

    if args.o is None:
        output_file_name = ".".join(os.path.basename(args.f).split(".")[:-1]) + ".input"
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.f)), output_file_name)
    else:
        output_path = args.o

    parse(args, fr, output_path)


if __name__ == "__main__":
    build()
