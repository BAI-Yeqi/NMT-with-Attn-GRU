'''
PyTorch Models

author: Bai Yeqi
email: yeqi001@ntu.edu.sg
'''


import torch
import os
import argparse
from dataset import prepareData
from model import EncoderRNN, AttnDecoderRNN, device
from trainer import trainIters
from evaluator import evaluateBLEU, evalAndShowAttns, load_model


def main(args):
    input_lang, output_lang, train_pairs, test_pairs = \
        prepareData('eng', 'fra', True)
    hidden_size = args.word_dim + args.char_dim
    input_use_char = bool(args.char_dim)
    encoder1 = EncoderRNN(
        input_lang.n_words, args.word_dim, 
        args.char_dim, input_lang.n_chars
    ).to(device)
    attn_decoder1 = AttnDecoderRNN(
        hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    if not args.eval_mode:
        trainIters(
            encoder1, attn_decoder1, args.train_steps, train_pairs,
            input_lang, output_lang, print_every=5000,
            input_use_char=input_use_char, output_dir=args.output_dir,
            teacher_forcing_ratio=args.teacher_forcing_ratio)
    else:
        # Skip training, load pre-trained weights
        encoder1, attn_decoder1 = load_model(
            encoder1, attn_decoder1, args.output_dir)
    evaluateBLEU(
        encoder1, attn_decoder1, test_pairs,
        input_lang, output_lang, input_use_char=input_use_char,
        print_utterance=True, beam_size=args.beam_size)
    evalAndShowAttns(
        encoder1, attn_decoder1, args.output_dir,
        input_lang, output_lang, input_use_char=input_use_char,
        beam_size=args.beam_size)


def parse_args():
    parser = argparse.ArgumentParser(description='NMT Model')
    parser.add_argument('--model_name', type=str, default='baseline',
                        help='name of the model')
    parser.add_argument('--char_dim', type=int, default=0,
                        help='dimension of char encoder')
    parser.add_argument('--word_dim', type=int, default=256,
                        help='dimension of word encoder')
    parser.add_argument('--train_steps', type=int, default=75000,
                        help='number of training steps')
    parser.add_argument('--eval_mode', default=False,
                        action='store_true')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='infer with beam search if beam_size > 1')
    parser.add_argument('--output_dir', type=str, default='',
                        help='placeholder, do not modify')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='teacher_forcing_ratio')
    args = parser.parse_args()
    os.makedirs('./output', exist_ok=True)
    args.output_dir = os.path.join('./output', args.model_name)
    return args
    

if __name__ == '__main__':
    args = parse_args()
    main(args) 
