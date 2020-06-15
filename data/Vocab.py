from collections import Counter
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.utils.rnn import pad_sequence
import os
import pickle


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def bert_pretraining(dataset, config):
    bert_tokenizer = BertTokenizer('./bert-base-chinese' + '/vocab.txt')
    model = BertModel.from_pretrained('./bert-base-chinese')
    model.eval()
    model.to(config.device)

    for batch in batch_slice(dataset, config.train_batch_size):
        tokens_tensor = []

        for instance in batch:
            instance.ids = bert_tokenizer.convert_tokens_to_ids(instance.chars)
            tokens_tensor.append(torch.tensor(instance.ids))

        tokens_tensor = pad_sequence(tokens_tensor).T
        attention_mask = torch.ne(tokens_tensor, torch.zeros_like(tokens_tensor))

        tokens_tensor = tokens_tensor.to(config.device)
        attention_mask = attention_mask.to(config.device)

        with torch.no_grad():
            outputs = model(tokens_tensor, attention_mask=attention_mask)
            encoded_layers = outputs[0]

        for index, instance in enumerate(batch):
            instance.embeddings = encoded_layers[index, 0:len(instance.ids), :].cpu().numpy()


class Vocab(object):
    PAD, UNK = 0, 1

    def __init__(self):
        return

    def create_label(self, train_data):
        label_counter = Counter()
        for inst in train_data:
            for label in inst.gold_labels:
                label_counter[label] += 1
        self._id2label = []

        for label, count in label_counter.most_common():
            self._id2label.append(label)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._label2id = reverse(self._id2label)
        if len(self._label2id) != len(self._id2label):
            print("serious bug: label dumplicated, please check!")

    def create_pretrained_embs(self, embfile):
        return

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.UNK) for x in xs]
        return self._label2id.get(xs, self.UNK)

    def id2label(self, xs):
        if isinstance(xs, list):
            return [self._id2label[x] for x in xs]
        return self._id2label[xs]

    @property
    def label_size(self):
        return len(self._id2label)


def creatVocab(train_data, dev_data, test_data, config):
    if os.path.exists(config.load_vocab_path):
        print('loading bert pre-trained embedding...')
        with open(config.load_vocab_path, "rb") as f:
            train_data, dev_data, test_data, vocab = pickle.load(f)
        print('load bert-pretrained embedding successfully')
    else:
        print('generating bert pre-trained embedding...')
        vocab = Vocab()
        vocab.create_label(train_data)
        bert_pretraining(train_data + dev_data + test_data, config)
        with open(config.load_vocab_path, "wb") as f:
            pickle.dump((train_data, dev_data, test_data, vocab), f)
        print('generate and save bert pre-trained embedding successfully')
    vocab = Vocab()
    vocab.create_label(train_data)
    return train_data, dev_data, test_data, vocab


