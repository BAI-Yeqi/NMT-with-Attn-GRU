'''
Data Processing and Data Loading

author: Bai Yeqi
email: yeqi001@ntu.edu.sg
'''


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
from sklearn.model_selection import train_test_split


SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class Lang:
    def __init__(self, name, use_char=True):
        self.name = name
        # Word Utils
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        # Character Utils
        self.use_char = use_char
        if self.use_char:
            self.char2index = {"PAD": 0, "EOS": 1, "SOS": 2}
            self.char2count = {"PAD": 0, "EOS": 1, "SOS": 2}
            self.index2char = {0: "PAD", 1: "EOS", 2: "SOS"}
            self.n_chars = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
            if self.use_char:
                for char in word:
                    self.addChar(char)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1


def unicodeToAscii(s):
    '''
    Turn a Unicode string to plain ASCII
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    '''
    Lowercase, trim, and remove non-letter characters
    '''
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    train_pairs, test_pairs = train_test_split(
        pairs, test_size=0.20, random_state=1234)
    print('{} utterances in total'.format(len(pairs)))
    print('{} utterances in training set'.format(len(train_pairs)))
    print('{} utterances in test set'.format(len(test_pairs)))
    return input_lang, output_lang, train_pairs, test_pairs


if __name__ == '__main__':
    # Unit test
    import torch
    from trainer import charIndexsFromSentence, tensorsFromPair

    input_lang, output_lang, train_pairs, test_pairs = prepareData('eng', 'fra', True)
    print('input_lang.char2index:', input_lang.char2index)
    print('output_lang.char2index:', output_lang.char2index)
    pair = random.choice(train_pairs)
    print(pair)
    seq_char_ids = charIndexsFromSentence(input_lang, pair[0])
    #seq_char_ids = np.array(seq_char_ids)
    seq_char_ids = torch.tensor(seq_char_ids, dtype=torch.long)
    print(seq_char_ids, seq_char_ids.shape)
    input_tensor, input_char_tensor, target_tensor = \
        tensorsFromPair(pair, input_lang, output_lang, input_use_char=True)
    print(input_tensor.shape,
          input_char_tensor.shape,
          target_tensor.shape)
