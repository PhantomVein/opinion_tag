from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable
from data.Dataloader import *
from data.Instance import *
import re
import random

def read_conll(infile):
    sentence = []
    for line in infile:
        tok = line.strip()
        if tok == '':
            yield sentence
            sentence = []
        else:
            sentence.append(tok)

def parse_conll(info):
    chars = []
    gold_labels = []
    for line in info:
        id, c, l = line.split("\t")
        chars.append(c)
        gold_labels.append(l)

    bichars = []
    char_len = len(chars)
    for idx in range(char_len):
        if idx == 0:
            bichar = '-NULL-' + chars[idx]
        else:
            bichar = chars[idx - 1] + chars[idx]
        bichars.append(bichar)

    inst = Instance()
    inst.chars = chars
    inst.gold_labels = gold_labels
    return inst

def read_corpus(file_path, max_char_len=-1):
    data = []
    with open(file_path, mode='r', encoding='utf8') as infile:
        for info in read_conll(infile):
            inst = parse_conll(info)
            if max_char_len == -1:
                data.append(inst)
            else:
                if len(inst.chars) < max_char_len:
                    data.append(inst)
    return data

def parse_sentence(line):
    str_len = len(line)

    chars = []

    for idx in range(str_len):
        chars.append(line[idx])

    bichars = []
    for idx in range(str_len):
        if idx == 0:
            bichar = '-NULL-' + chars[idx]
        else:
            bichar = chars[idx - 1] + chars[idx]
        bichars.append(bichar)

    inst = Instance()
    inst.chars = chars
    inst.bichars = bichars
    return inst

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

def read_raw_corpus(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf8') as infile:
        for line in infile.readlines():
            line = line.strip()
            sentences = cut_sent(line)
            for sent in sentences:
                sent = re.sub(r'\s+', '', sent)
                inst = parse_sentence(sent)
                data.append(inst)
    return data

def get_gold_label(data):
    for inst in data:
        inst.gold_labels = []
        for word in inst.words:
            for index, c in enumerate(word):
                if index == 0:
                    inst.gold_labels.append('b')
                else:
                    inst.gold_labels.append('i')
        assert len(inst.gold_labels) == len(inst.chars)

def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def actions_variable(batch, vocab):
    batch_feats = []
    batch_actions = []
    batch_action_indexes = []
    batch_candidate = []
    for data in batch:
        feat = data[1]
        batch_feats.append(feat)
    for data in batch:
        actions = data[2]
        action_indexes = np.zeros(len(actions), dtype=np.int32)
        batch_actions.append(actions)
        for idx in range(len(actions)):
            ac = actions[idx]
            index = vocab.ac2id(ac)
            action_indexes[idx] = index
        batch_action_indexes.append(action_indexes)
    for data in batch:
        candidate = data[3]
        batch_candidate.append(candidate)
    return batch_feats, batch_actions, batch_action_indexes, batch_candidate

def batch_data_variable(batch, vocab, config):
    batch_size = len(batch)
    max_edu_len = -1
    max_edu_num = -1
    for data in batch:
        EDUs = data[0].EDUs
        edu_num = len(EDUs)
        if edu_num > max_edu_num: max_edu_num = edu_num
        for edu in EDUs:
            EDU_len = edu.end - edu.start + 1
            if EDU_len > max_edu_len:max_edu_len = EDU_len

    if max_edu_len > config.max_edu_len: max_edu_len = config.max_edu_len

    #edu_words = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    #edu_extwords = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    #edu_tags = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    #word_mask = Variable(torch.Tensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    #word_denominator = Variable(torch.ones(batch_size, max_edu_num).type(torch.FloatTensor) * -1, requires_grad=False)
    #edu_mask = Variable(torch.Tensor(batch_size, max_edu_num).zero_(), requires_grad=False)
    #edu_types = Variable(torch.LongTensor(batch_size, max_edu_num).zero_(), requires_grad=False)

    edu_words = np.zeros((batch_size, max_edu_num, max_edu_len), dtype=int)
    edu_extwords = np.zeros((batch_size, max_edu_num, max_edu_len), dtype=int)
    edu_tags = np.zeros((batch_size, max_edu_num, max_edu_len), dtype=int)
    word_mask = np.zeros((batch_size, max_edu_num, max_edu_len), dtype=int)
    word_denominator = np.ones((batch_size, max_edu_num), dtype=int) * -1
    edu_mask = np.zeros((batch_size, max_edu_num), dtype=int)
    edu_types = np.zeros((batch_size, max_edu_num), dtype=int)

    for idx in range(batch_size):
        doc = batch[idx][0]
        EDUs = doc.EDUs
        edu_num = len(EDUs)
        for idy in range(edu_num):
            edu = EDUs[idy]
            edu_types[idx, idy] = vocab.EDUtype2id(edu.type)
            edu_len = len(edu.words)
            edu_mask[idx, idy] = 1
            word_denominator[idx, idy] = edu_len
            assert edu_len == len(edu.tags)
            for idz in range(edu_len):
                if idz >= max_edu_len:
                    break
                word = edu.words[idz]
                tag = edu.tags[idz]
                edu_words[idx, idy, idz] = vocab.word2id(word)
                edu_extwords[idx, idy, idz] = vocab.extword2id(word)
                tag_id = vocab.tag2id(tag)
                edu_tags[idx, idy, idz] = tag_id
                word_mask[idx, idy, idz] = 1

    edu_words = torch.from_numpy(edu_words).type(torch.LongTensor)
    edu_extwords = torch.from_numpy(edu_extwords).type(torch.LongTensor)
    edu_tags = torch.from_numpy(edu_tags).type(torch.LongTensor)
    word_mask = torch.from_numpy(word_mask).type(torch.FloatTensor)
    word_denominator = torch.from_numpy(word_denominator).type(torch.FloatTensor)
    edu_mask = torch.from_numpy(edu_mask).type(torch.FloatTensor)
    edu_types = torch.from_numpy(edu_types).type(torch.LongTensor)

    return edu_words, edu_extwords, edu_tags, word_mask, edu_mask, word_denominator, edu_types

def load_pretrained_embs(embfile):
    embedding_dim = -1
    word_count = 0
    with open(embfile, encoding='utf-8') as f:
        for line in f.readlines():
            if word_count < 1:
                values = line.split()
                embedding_dim = len(values) - 1
            word_count += 1
    print('Total words: ' + str(word_count) + '\n')
    print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')
    id2elem = ['<pad>', '<unk>']
    index = len(id2elem)
    embeddings = np.zeros((word_count + index, embedding_dim))
    with open(embfile, encoding='utf-8') as f:
        for line in f.readlines():
            values = line.split()
            id2elem.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[Vocab.UNK] += vector
            embeddings[index] = vector
            index += 1

    embeddings[Vocab.UNK] = embeddings[Vocab.UNK] / word_count
    embeddings = embeddings / np.std(embeddings)

    reverse = lambda x: dict(zip(x, range(len(x))))
    elem2id = reverse(id2elem)

    if len(elem2id) != len(id2elem):
        print("serious bug: extern words dumplicated, please check!")

    return embeddings, id2elem, elem2id

def label_variable(onebatch, vocab):
    batch_size = len(onebatch)
    lengths = []
    for inst in onebatch:
        lengths.append(len(inst.chars))
    max_len = max(lengths)

    batch_gold_labels = np.ones((batch_size, max_len), dtype=int)

    for idx in range(batch_size):
        gold_label_strs = onebatch[idx].gold_labels
        gold_label_indexes = vocab.label2id(gold_label_strs)
        idy = 0
        for label_index in gold_label_indexes:
            batch_gold_labels[idx][idy] = label_index
            idy += 1
    batch_gold_labels = torch.tensor(batch_gold_labels, dtype=torch.long)
    return batch_gold_labels


def data_variable(onebatch, vocab):
    batch_size = len(onebatch)
    char_lengths = []
    for inst in onebatch:
        char_lengths.append(len(inst.chars))

    max_char_len = max(char_lengths)

    batch_chars = []
    char_mask = np.zeros((batch_size, max_char_len), dtype=float)

    for idx in range(batch_size):
        batch_chars.append(torch.from_numpy(onebatch[idx].embeddings))
        for idy, char_index in enumerate(onebatch[idx].ids):
            char_mask[idx][idy] = 1

    batch_chars = torch.nn.utils.rnn.pad_sequence(batch_chars)
    batch_chars = torch.transpose(batch_chars, 0, 1)

    char_mask = torch.tensor(char_mask, dtype=torch.float)

    label_mask = char_mask.type(torch.long)

    return batch_chars, char_mask, label_mask

def path2labels(paths, vocab):
    if isinstance(paths, list):
        return [vocab.id2label(x) for x in paths]
    return vocab.id2label(paths)

def labels2output(onebatch, labels):
    outputs = []
    for idx, inst in enumerate(onebatch):
        predict_labels = labels[idx]
        assert len(predict_labels) == len(inst.chars)

        tmp = ''
        predict_sent = []
        for idy, label in enumerate(predict_labels):
            if label == 'b':
                if idy > 0:
                    predict_sent.append(tmp)
                tmp = inst.chars[idy]

            if label == 'i':
                tmp += inst.chars[idy]
        if tmp is not '':
            predict_sent.append(tmp)
        outputs.append(predict_sent)

    return outputs

def load_word_from_dic(file_path):
    word_list = []
    with open(file_path, mode='r', encoding='utf8') as infile:
        for word in infile.readlines():
            word = word.strip()
            word_list.append(word)
    return word_list
