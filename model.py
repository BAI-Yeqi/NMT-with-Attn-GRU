'''
PyTorch Models

author: Bai Yeqi
email: yeqi001@ntu.edu.sg
'''


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataset import MAX_LENGTH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN1DBlock(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 batchnorm=False):
        '''
        1D CNN block
        '''
        nn.Module.__init__(self)
        self.batchnorm = batchnorm
        self.layers = []
        self.layers.append(
            nn.Conv1d(channel_in,
                      channel_out,
                      kernel,
                      stride,
                      padding,
                      dilation,
                      bias=False))
        if self.batchnorm:
            self.layers.append(
                nn.BatchNorm1d(channel_out, affine=True))
        self.layers.append(
            nn.ReLU(inplace=True))
        self.model = nn.Sequential(*self.layers)

    def forward(self, seq_in):
        '''
        seq_in: tensor with shape [N, D, L]
        '''
        seq_out = self.model(seq_in)
        return seq_out


class CharEncoderCNN(nn.Module):
    def __init__(self, n_chars, char_dim):
        super(CharEncoderCNN, self).__init__()
        self.char_dim = char_dim

        self.embedding = nn.Embedding(n_chars, char_dim)
        self.cnn = CNN1DBlock(char_dim, char_dim)

    def forward(self, input):
        # [1, char_dim, seq_len]
        embedded = self.embedding(input).transpose(0, 1).unsqueeze(0)
        output = self.cnn(embedded)
        output = torch.mean(output, -1).unsqueeze(1)
        #print(output.shape)
        return output


class EncoderRNN(nn.Module):
    def __init__(self, n_words, word_dim, char_dim=None, n_chars=None):
        super(EncoderRNN, self).__init__()
        self.word_dim = word_dim
        if char_dim:
            self.use_char = True
            self.char_cnn = CharEncoderCNN(n_chars, char_dim)
            print('EncoderRNN: Using Character Encoder')
        else:
            self.use_char = False
            char_dim = 0
            print('EncoderRNN: NOT Using Character Encoder')
        self.hidden_size = word_dim + char_dim

        self.word_embedding = nn.Embedding(n_words, word_dim)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, word_tensor, char_tensor=None, hidden=None):
        #print(self.embedding(word_tensor).shape) # [1, 256]
        word_emb = self.word_embedding(word_tensor).view(1, 1, -1)
        if self.use_char:
            char_emb = self.char_cnn(char_tensor)
            #print(char_emb.shape)
            word_emb = torch.cat([word_emb, char_emb], -1)
        #print(word_emb.shape)
        output, hidden = self.gru(word_emb, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
