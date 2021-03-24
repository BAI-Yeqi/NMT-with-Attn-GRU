'''
Evaluator

author: Bai Yeqi
email: yeqi001@ntu.edu.sg
'''


import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
from nltk.translate.bleu_score import sentence_bleu
from dataset import MAX_LENGTH, SOS_token, EOS_token
from trainer import tensorFromSentence
from utils import demo_french_sentences
from model import device


def evaluate(encoder, decoder, sentence, 
             input_lang, output_lang, max_length=MAX_LENGTH,
             input_use_char=False):
    with torch.no_grad():
        if input_use_char:
            input_tensor, input_char_tensor = tensorFromSentence(
                input_lang, sentence, input_use_char)
        else:
            input_tensor = tensorFromSentence(
                input_lang, sentence, input_use_char)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            if input_use_char:
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], input_char_tensor[ei], encoder_hidden)
            else:
                encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, pairs,
                     input_lang, output_lang, n=10,
                     input_use_char=False):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(
            encoder, decoder, pair[0], 
            input_lang, output_lang,
            input_use_char=input_use_char)
        output_sentence = ' '.join(output_words)
        print('output_words:', output_words)
        print('<', output_sentence)
        print('')


def evaluateBLEU(encoder, decoder, pairs,
                 input_lang, output_lang, n=10,
                 input_use_char=False, 
                 print_utterance=False):
    scores = []
    for i, pair in enumerate(pairs):
        gt_words = pair[1].split(' ')
        output_words, attentions = evaluate(
            encoder, decoder, pair[0],
            input_lang, output_lang,
            input_use_char=input_use_char)
        output_words= output_words[0:-1]
        output_sentence = ' '.join(output_words)
        print('gt_words:', gt_words)
        print('output_words:', output_words)
        score = sentence_bleu([gt_words], output_words)
        scores.append(score)
        if print_utterance:
            print('>', pair[0])
            print('=', pair[1])
            print('<', output_sentence)
            print('Instance BLEU Score: {:.2f}'.format(score))
            print('')
    avg_score = np.mean(scores)
    print('Test BLEU Score: {:.2f}'.format(avg_score))
    return avg_score
 

def visAttention():
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, "je suis trop froid .")
    plt.matshow(attentions.numpy())


def showAttention(input_sentence, output_words, attentions, 
                  save_path, show=False):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(save_path)
    if show:
        plt.show()


def evaluateAndShowAttention(input_sentence, encoder, 
                             attn_decoder, save_path,
                             input_lang, output_lang,
                             input_use_char=False):
    output_words, attentions = evaluate(
        encoder, attn_decoder, input_sentence,
        input_lang, output_lang, input_use_char=input_use_char)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions, save_path)
    

def evalAndShowAttns(encoder, attn_decoder, output_dir, 
                     input_lang, output_lang, input_use_char=False):
    os.makedirs(output_dir, exist_ok=True)
    for i, sentence in enumerate(demo_french_sentences):
        save_path = os.path.join(output_dir, '{}.png'.format(i))
        evaluateAndShowAttention(
            sentence, encoder, attn_decoder, save_path,
            input_lang, output_lang, input_use_char=input_use_char)
