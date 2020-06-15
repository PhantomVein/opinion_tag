from data.Metric import *
from data.Dataloader import *


def get_ent(labels):
    idx = 0
    idy = 0
    endpos = -1
    ent = []
    while (idx < len(labels)):
        if (is_start_label(labels[idx])):
            idy = idx
            endpos = -1
            while (idy < len(labels)):
                if not is_continue_label(labels[idy], labels[idx], idy - idx):
                    endpos = idy - 1
                    break
                endpos = idy
                idy += 1
            ent.append(clean_label(labels[idx]) + '[' + str(idx) + ',' + str(endpos) + ']')
            idx = endpos
        idx += 1
    return ent


def clean_label(label):
    start = ['B', 'b', 'M', 'm', 'E', 'e', 'S', 's', 'I', 'i']
    if len(label) > 2 and label[1] == '-':
        if label[0] in start:
            return label[2:]
    return label


def is_continue_label(label, startLabel, distance):
    if distance == 0:
        return True
    if len(label) < 3:
        return False
    if distance != 0 and is_start_label(label):
        return False
    if (startLabel[0] == 's' or startLabel[0] == 'S') and startLabel[1] == '-':
        return False
    if clean_label(label) != clean_label(startLabel):
        return False
    return True


def is_start_label(label):
    start = ['b', 'B', 's', 'S']
    if (len(label) < 3):
        return False
    else:
        return (label[0] in start) and label[1] == '-'


def segprf(onebatch, outputs):
    predict_num = 0
    correct_num = 0
    gold_num = 0
    assert len(onebatch) == len(outputs)
    for idx, inst in enumerate(onebatch):
        # check(inst.words, outputs[idx])

        gold_set = set(get_ent(inst.words))
        predict_set = set(get_ent(outputs[idx]))

        predict_num += len(predict_set)
        gold_num += len(gold_set)

        correct_num += len(predict_set & gold_set)
    return correct_num, predict_num, gold_num


def check(inst1, inst2):
    assert len(inst1.chars) == len(inst2.chars)
    for c1, c2 in zip(inst1.chars, inst2.chars):
        assert c1 == c2


def evaluate(gold_file, predict_file):
    metric = Metric()

    gold_insts = read_corpus(gold_file)
    predict_insts = read_corpus(predict_file)

    predict_num = 0
    correct_num = 0
    gold_num = 0

    for g_inst, p_inst in zip(gold_insts, predict_insts):
        check(g_inst, p_inst)

        gold_set = set(get_ent(g_inst.gold_labels))
        predict_set = set(get_ent(p_inst.gold_labels))

        predict_num += len(predict_set)
        gold_num += len(gold_set)
        correct_num += len(predict_set & gold_set)

    metric.correct_label_count = correct_num
    metric.predicated_label_count = predict_num
    metric.overall_label_count = gold_num

    return metric


def get_loose_ent(labels):
    idx = 0
    idy = 0
    endpos = -1
    loose_ent = []
    while (idx < len(labels)):
        if (is_start_label(labels[idx])):
            idy = idx
            endpos = -1
            while (idy < len(labels)):
                if not is_continue_label(labels[idy], labels[idx], idy - idx):
                    endpos = idy - 1
                    break
                endpos = idy
                idy += 1
            loose_ent.append((clean_label(labels[idx]), idx, endpos))
            idx = endpos
        idx += 1
    return loose_ent


def loose_evaluate(gold_file, predict_file):
    metric = Metric()

    gold_insts = read_corpus(gold_file)
    predict_insts = read_corpus(predict_file)

    predict_num = 0
    correct_num = 0
    gold_num = 0

    for g_inst, p_inst in zip(gold_insts, predict_insts):
        check(g_inst, p_inst)

        gold_set = set(get_loose_ent(g_inst.gold_labels))
        predict_set = set(get_loose_ent(p_inst.gold_labels))

        predict_num += len(predict_set)
        gold_num += len(gold_set)
        # correct_num += len(predict_set & gold_set)
        for loose_predict_ent in predict_set:
            for loose_gold_ent in gold_set:
                if loose_predict_ent[0] == loose_gold_ent[0] and loose_predict_ent[1] >= loose_gold_ent[1] and \
                        loose_predict_ent[2] <= loose_gold_ent[2]:
                    correct_num += 1

    metric.correct_label_count = correct_num
    metric.predicated_label_count = predict_num
    metric.overall_label_count = gold_num

    return metric


def each_class_evaluate(label, gold_file, predict_file):
    metric = Metric()

    gold_insts = read_corpus(gold_file)
    predict_insts = read_corpus(predict_file)

    predict_num = 0
    correct_num = 0
    gold_num = 0

    for g_inst, p_inst in zip(gold_insts, predict_insts):
        check(g_inst, p_inst)

        gold_set = set(get_loose_ent(g_inst.gold_labels))
        predict_set = set(get_loose_ent(p_inst.gold_labels))

        # correct_num += len(predict_set & gold_set)
        for loose_predict_ent in predict_set:
            if loose_predict_ent[0] == label:
                predict_num += 1
        for loose_gold_ent in gold_set:
            if loose_gold_ent[0] == label:
                gold_num += 1
        for loose_predict_ent in predict_set:
            for loose_gold_ent in gold_set:
                if loose_predict_ent[0] == label and loose_predict_ent[1] == loose_gold_ent[1] and loose_predict_ent[2] == loose_gold_ent[2]:
                    correct_num += 1

    metric.correct_label_count = correct_num
    metric.predicated_label_count = predict_num
    metric.overall_label_count = gold_num

    return metric


def acc_evaluate(gold_file, predict_file):
    metric = Metric()

    gold_insts = read_corpus(gold_file)
    predict_insts = read_corpus(predict_file)

    predict_num = 0
    correct_num = 0
    gold_num = 0

    for g_inst, p_inst in zip(gold_insts, predict_insts):
        check(g_inst, p_inst)

        gold_labels = g_inst.gold_labels
        pred_labels = p_inst.gold_labels

        predict_num += len(gold_labels)
        gold_num += len(pred_labels)
        correct_num += sum([gold_label == pred_label for gold_label, pred_label in zip(gold_labels, pred_labels)])

    metric.correct_label_count = correct_num
    metric.predicated_label_count = predict_num
    metric.overall_label_count = gold_num

    return metric
