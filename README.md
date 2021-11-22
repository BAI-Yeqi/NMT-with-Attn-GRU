# NMT-with-Attn-GRU
This repo is a clean, readable, and easy-to-use implementation of Neural Machine Translation with Attentive GRU, which supports teacher forcing and Beam Search decoding.

## Environment Setup
1. In a Python 3.6 environment, install PyTorch according to official website: https://pytorch.org/get-started/locally/
2. install required packages:
```
pip install -r requirements.txt
```

## Dataset
Download data (https://www.manythings.org/anki/):
```
wget https://download.pytorch.org/tutorial/data.zip
unzip data.zip
```

## Train & Evaluate
To train a Word + Character Encoding model with 75000 training steps and teacher forcing ratio 0.75:
```
python main.py --char_dim 128 --model_name 'charEnc_tf0.75' \
    --train_steps 75000 --teacher_forcing_ratio 0.75
```
The corresponding encoder and decoder weights are stored in `./output/charEnc_tf0.75`, and evalulation will be  conducted with greedy decoding algorithm automatically. BLEU score will be displayed in your terminal. When `--char_dim` is set to any positive integer, a character encoder with the corrsponding dimension will be built and trained along with the word encoder and decoder.

To further evaluate the trained model with Beam Search algorithm:
```
python main.py --char_dim 128 --model_name 'charEnc_tf0.75' \
    --eval_mode --beam_size 5
```
By specifying the `--eval_mode` option, `main.py` will skip training and load the trained weights from `./output/charEnc_tf0.75`. When `--beam_size > 1`, beam search algorithm will be executed to evaluate the BLEU score.

Similar commands can be used to train and evaluate the baseline model (without Character Encoding):
```
python main.py --char_dim 0 --model_name 'baseline_tf0.75' \
    --train_steps 75000 --teacher_forcing_ratio 0.75

python main.py --char_dim 0 --model_name 'baseline_tf0.75' \
    --eval_mode --beam_size 5
```

At training time, the `--teacher_forcing_ratio` argument can be used to control the probabilty of using teacher forcing.
