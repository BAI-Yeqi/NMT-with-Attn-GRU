'''
PyTorch Models

author: Bai Yeqi
email: yeqi001@ntu.edu.sg
'''


import torch
from dataset import prepareData
from model import EncoderRNN, AttnDecoderRNN, device
from trainer import trainIters
from evaluator import evaluateRandomly, evaluateAndShowAttention
from utils import demo_french_sentences


def main():
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    hidden_size = 256
    encoder1 = EncoderRNN(
        input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(
        hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    trainIters(
        encoder1, attn_decoder1, 75000, pairs, 
        input_lang, output_lang, print_every=5000
    )
    evaluateRandomly(encoder1, attn_decoder1)
    for sentence in demo_french_sentences:
        evaluateAndShowAttention(sentence, encoder1, attn_decoder1)


if __name__ == '__main__':
    main()
