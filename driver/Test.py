import sys
sys.path.extend(["../../","../","./"])
import random
from data.Dataloader import *
import argparse
from driver.Config import *
from modules.TaggerModel import LSTMscorer
from modules.Tagger import *
import pickle
from driver.TrainTest import labeling
from data.Evaluate import *

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
    argparser.add_argument('--test_file', default='', help='without evaluation')

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

    if args.test_file is '':
        test_data = read_corpus(config.test_file)
        get_gold_label(test_data)
        labeling(test_data, tagger, vocab, config, config.test_file + '.out')
        test_seg_eval = evaluate(config.test_file, config.test_file + '.out')
        test_seg_eval.print()
    else:
        test_data = read_corpus(args.test_file)
        labeling(test_data, tagger, vocab, config, args.test_file + '.out')
        test_seg_eval = evaluate(args.test_file, args.test_file + '.out')
        test_seg_eval.print()





