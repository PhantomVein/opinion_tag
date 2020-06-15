from Metric import *
import pandas as pd


def evaluate(gold_set_list, predict_set_list):
    metric = Metric()

    predict_num = 0
    correct_num = 0
    gold_num = 0

    for gold_set, predict_set in zip(gold_set_list, predict_set_list):
        predict_num += len(predict_set)
        gold_num += len(gold_set)
        correct_num += len(predict_set & gold_set)

    metric.correct_label_count = correct_num
    metric.predicated_label_count = predict_num
    metric.overall_label_count = gold_num

    return metric


def seg_chunks(text: str, sep):
    return set(text.split(sep))


def eval_yn():
    df = pd.read_excel('业务产品现模型与人工结果对比_v1.0_1023.xlsx')
    gold_set_list = []
    predict_set_list = []
    for row in df.itertuples():
        gold_set_list.append(seg_chunks(row[2], ","))
        predict_set_list.append(seg_chunks(row[3], ","))
    result = evaluate(gold_set_list, predict_set_list)
    result.print()


def eval_from_file(test_path):
    df = pd.read_excel('业务产品现模型与人工结果对比_v1.0_1023.xlsx')
    gold_set_list = []
    predict_set_list = []
    for row in df.itertuples():
        gold_set_list.append(seg_chunks(row[2], ","))
        file_name = row[1]
        file_name += ".txt.ner"
        file_path = test_path + file_name
        content = ''
        with open(file_path, mode='r', encoding='utf8') as file:
            content = file.read()
        predict_set_list.append(seg_chunks(content, " "))
    result = evaluate(gold_set_list, predict_set_list)
    result.print()


eval_from_file("../test_data/test_doc/")
