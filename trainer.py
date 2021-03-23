'''
Trainer

author: Bai Yeqi
email: yeqi001@ntu.edu.sg
'''


import torch
import torch.nn as nn
from torch import optim
import os
import time
import random
from dataset import MAX_LENGTH, SOS_token, EOS_token
from utils import timeSince, showPlot
from model import device


def charIndexsFromSentence(lang, sentence):
    max_word_len = 0
    seq_char_ids = []
    for word in sentence.split(' '):
        word_len = len(word)
        char_ids = []
        if word_len > max_word_len:
            max_word_len = word_len
        for char in word:
            char_id = lang.char2index[char]
            char_ids.append(char_id)
        seq_char_ids.append(char_ids)
    seq_char_ids.append([lang.char2index['EOS']])
    for i, char_ids in enumerate(seq_char_ids):
        if len(char_ids) < max_word_len:
            pad_len = max_word_len - len(char_ids)
            padding = [lang.char2index['PAD']] * pad_len
            seq_char_ids[i] = seq_char_ids[i] + padding
    return seq_char_ids


def indexesFromSentence(lang, sentence, use_char=False):
    word_ids = [lang.word2index[word] for word in sentence.split(' ')]
    if use_char:
        char_ids = charIndexsFromSentence(lang, sentence)
        return word_ids, char_ids
    else:
        return word_ids


def tensorFromSentence(lang, sentence, use_char=False):
    if use_char:
        word_ids, char_ids = indexesFromSentence(
            lang, sentence, use_char)
        char_tensor = torch.tensor(
            char_ids, dtype=torch.long, device=device)
    else:
        word_ids = indexesFromSentence(lang, sentence)
    word_ids.append(EOS_token)
    word_tensor = torch.tensor(
        word_ids, dtype=torch.long, device=device
    ).view(-1, 1)
    if use_char:
        return word_tensor, char_tensor
    else:
        return word_tensor


def tensorsFromPair(pair, input_lang, output_lang, input_use_char=False):
    if input_use_char:
        input_tensor, input_char_tensor = \
            tensorFromSentence(input_lang, pair[0], input_use_char)
    else:
        input_tensor = tensorFromSentence(input_lang, pair[0], input_use_char)
    target_tensor = tensorFromSentence(output_lang, pair[1])
    if input_use_char:
        return (input_tensor, input_char_tensor, target_tensor)
    else:
        return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, criterion, 
          max_length=MAX_LENGTH, teacher_forcing_ratio=0.5,
          input_char_tensor=None):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        if input_char_tensor is not None:
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], input_char_tensor[ei],
                encoder_hidden)
        else:
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)                
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < \
        teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, pairs,
               input_lang, output_lang, print_every=1000,
               plot_every=100, learning_rate=0.01,
               input_use_char=False, output_dir='output'):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, 
                                      output_lang, input_use_char)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        if not input_use_char:
            input_tensor, target_tensor = training_pair
            input_char_tensor = None
        else:
            input_tensor, input_char_tensor, target_tensor = training_pair

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer,
                     criterion, input_char_tensor=input_char_tensor)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    os.makedirs(output_dir, exist_ok=True)
    showPlot(
        plot_losses, 
        os.path.join(output_dir, 'loss.png')
    )
