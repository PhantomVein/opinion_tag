import sys
sys.path.extend(["../../","../","./"])
import random
import argparse
from data.Dataloader import *
from driver.Config import *
from data.Vocab import *
from modules.TaggerModel import LSTMscorer
from modules.Tagger import *
from data.Evaluate import *
import time
import itertools
import torch.nn as nn
import pickle
from modules.Optimizer import *

def train(train_data, dev_data, test_data, vocab, config, tagger):

    model_param = filter(lambda p: p.requires_grad,
                         itertools.chain(
                             tagger.lstm_scorer.parameters(),
                             tagger.crf.parameters()
                         )
                         )

    model_optimizer = Optimizer(model_param, config)

    batch_num = int(np.ceil(len(train_data) / float(config.train_batch_size)))
    global_step = 0
    best_F = 0

    print(print(tagger.lstm_scorer.parameters))
    print(print(tagger.crf.parameters))
    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_correct,  overall_total = 0, 0
        for onebatch in data_iter(train_data, config.train_batch_size, True):
            batch_gold_labels = label_variable(onebatch, vocab)

            batch_chars, char_mask, label_mask = \
                data_variable(onebatch, vocab)
            tagger.train()

            tagger.forward(batch_chars, char_mask)
            loss = tagger.compute_loss(batch_gold_labels, label_mask)

            total, correct = tagger.compute_acc(batch_gold_labels, label_mask)
            overall_total += total
            overall_correct += correct
            acc = overall_correct / overall_total
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            during_time = float(time.time() - start_time)
            print("Step:%d, Iter:%d, batch:%d, time:%.2f, acc:%.4f, loss:%.4f"
                  %(global_step, iter, batch_iter,  during_time, acc, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(model_param, max_norm=config.clip)

                model_optimizer.step()
                model_optimizer.zero_grad()

                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                labeling(dev_data, tagger, vocab, config, config.dev_file + '.' + str(global_step))
                dev_seg_eval = evaluate(config.dev_file, config.dev_file + '.' + str(global_step))

                print("Dev:")
                dev_seg_eval.print()

                labeling(test_data, tagger, vocab, config, config.test_file + '.' + str(global_step))
                test_seg_eval = evaluate(config.test_file, config.test_file + '.' + str(global_step))
                fac_class_seg_eval = each_class_evaluate('f', config.test_file, config.test_file + '.' + str(global_step))
                sug_class_seg_eval = each_class_evaluate('s', config.test_file,
                                                         config.test_file + '.' + str(global_step))
                con_class_seg_eval = each_class_evaluate('c', config.test_file,
                                                         config.test_file + '.' + str(global_step))
                loose_test_seg_eval = loose_evaluate(config.test_file, config.test_file + '.' + str(global_step))
                acc_test_seg_eval = acc_evaluate(config.test_file, config.test_file + '.' + str(global_step))

                print("Test:")
                test_seg_eval.print()
                print("fac class Test:")
                fac_class_seg_eval.print()
                print("sug class Test:")
                sug_class_seg_eval.print()
                print("con class Test:")
                con_class_seg_eval.print()
                print("Loose Test:")
                loose_test_seg_eval.print()
                print("Acc Test:")
                acc_test_seg_eval.print()

                dev_F = dev_seg_eval.getAccuracy()
                if best_F < dev_F:
                    print("Exceed best Full F-score: history = %.4f, current = %.4f" % (best_F, dev_F))
                    best_F = dev_F

                    if config.save_after >= 0 and iter >= config.save_after:
                        print("Save model")
                        tagger_model = {'lstm': tagger.lstm_scorer.state_dict(),
                                        'crf': tagger.crf.state_dict()}
                        torch.save(tagger_model, config.save_model_path + "." + str(global_step))


def labeling(data, tagger, vocab, config, outputFile, split_str=' '):
    start = time.time()
    outf = open(outputFile, mode='w', encoding='utf8')
    key_words = []
    for onebatch in data_iter(data, config.test_batch_size, False):
        batch_chars, char_mask, label_mask = \
            data_variable(onebatch, vocab)
        tagger.eval()

        b = len(onebatch)

        seg = False
        for idx in range(b):
            if len(onebatch[idx].chars) > 0:
                seg = True
                break

        if seg:
            tagger.forward(batch_chars, char_mask)
            best_paths = tagger.viterbi_decode(label_mask)

            labels = path2labels(best_paths, vocab)
            for idx in range(b):
                chars = onebatch[idx].chars
                label = labels[idx]
                key_words += get_key_words(chars, label)

                char_len = len(labels[idx])
                for idy in range(char_len):
                    outf.write(str(idy + 1) + "\t" + chars[idy] + "\t" + label[idy] + "\n")
                outf.write('\n')
    #outf.write(' '.join(set(key_words)) + '\n')
    during_time = float(time.time() - start)
    outf.close()
    print("sentence num: %d,  labeling time = %.2f " % (len(data), during_time))


def get_key_words(chars, labels):
    char_len = len(chars)
    assert char_len == len(labels)
    key_words = []
    idx = 0
    while True:
        if idx >= char_len:
            break

        if labels[idx] == 'b-NER':
            tmp_word = chars[idx]

            offset = 1
            while True:
                index = idx + offset
                if index >= char_len:
                    break

                if labels[index] != 'i-NER':
                    break

                if labels[index] == 'i-NER':
                    tmp_word += chars[index]
                offset += 1

            key_words.append(tmp_word)

            idx += len(tmp_word)
        else:
            idx += 1
    return key_words



if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='config.keyword.cfg')
    argparser.add_argument('--model', default='BaseSegment')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    train_data = read_corpus(config.train_file, config.max_train_inst_len)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)

    torch.set_num_threads(args.thread)
    config.use_cuda = False
    config.device = config.device = torch.device('cpu')
    if gpu and args.use_cuda:
        config.use_cuda = True
        config.device = torch.device('cuda')

    train_data, dev_data, test_data, vocab = creatVocab(train_data, dev_data, test_data, config)


    print("\nGPU using status: ", config.use_cuda)

    print("train num: ", len(train_data))
    print("dev num: ", len(dev_data))
    print("test num: ", len(test_data))

    # pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    crf = CRF(num_tags=vocab.label_size,
              constraints=None,
              include_start_end_transitions=False)

    lstm_scorer = LSTMscorer(vocab, config)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True

        lstm_scorer = lstm_scorer.cuda()
        crf = crf.cuda()

    tagger = Tagger(lstm_scorer, crf, vocab, config)
    train(train_data, dev_data, test_data, vocab, config, tagger)

