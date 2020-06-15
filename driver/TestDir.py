import sys
sys.path.extend(["../../","../","./"])
import random
from data.Dataloader import *
import argparse
from driver.Config import *
from modules.TaggerModel import LSTMscorer
from modules.Tagger import *
import pickle
from driver.TrainTest import *
from data.Evaluate import *

def labeling(data, tagger, vocab, config, outputFile, split_str=' '):
    start = time.time()
    outf = open(outputFile, mode='w', encoding='utf8')
    key_words = []
    for onebatch in data_iter(data, config.test_batch_size, False):
        batch_chars, batch_extchars, batch_bichars, batch_extbichars, char_mask, label_mask = \
            data_variable(onebatch, vocab)
        tagger.eval()

        b = len(onebatch)

        seg = False
        for idx in range(b):
            if len(onebatch[idx].chars) > 0:
                seg = True
                break

        if seg:
            tagger.forward(batch_chars, batch_extchars, batch_bichars, batch_extbichars, char_mask)
            best_paths = tagger.viterbi_decode(label_mask)

            labels = path2labels(best_paths, vocab)
            for idx in range(b):
                chars = onebatch[idx].chars
                label = labels[idx]
                key_words += get_key_words(chars, label)

            #for idy in range(char_len):
                #outf.write(str(idy + 1) + "\t" + chars[idy] + "\t" + label[idy] + "\n")
            #outf.write('\n')
    outf.write(' '.join(set(key_words)) + '\n')
    during_time = float(time.time() - start)
    outf.close()
    print("sentence num: %d,  labeling time = %.2f " % (len(data), during_time))



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
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model_id', default=1, type=int, help='model id')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--test_dir', default='', help='without evaluation')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    tagger_model = torch.load(config.load_model_path + '.' + str(args.model_id))

    crf = CRF(num_tags=vocab.label_size,
              constraints=None,
              include_start_end_transitions=False)
    lstm_scorer = LSTMscorer(vocab, config)

    lstm_scorer.load_state_dict(tagger_model['lstm'])
    crf.load_state_dict(tagger_model['crf'])

    if config.use_cuda:
        torch.backends.cudnn.enabled = True

        lstm_scorer = lstm_scorer.cuda()
        crf = crf.cuda()

    tagger = Tagger(lstm_scorer, crf, vocab, config)

    if args.test_dir is not '':
        dirs = os.listdir(args.test_dir)

        for dir in dirs:
            c_dir = os.path.join(args.test_dir, dir)
            files = os.listdir(c_dir)
            for file in files:
                file_path = os.path.join(c_dir, file)
                test_data = read_raw_corpus(file_path)
                print('labeling ' + file_path, end='....  ')
                labeling(test_data, tagger, vocab, config, file_path + '.ner', '/')

    print("labeling OK")






