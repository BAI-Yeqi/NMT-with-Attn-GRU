# NMT-with-Attn-GRU
Neural Machine Translation with Attentive GRU

## Dataset
Download data (https://www.manythings.org/anki/):
```
wget https://download.pytorch.org/tutorial/data.zip
unzip data.zip
```

## Train & Evaluate
```
CUDA_VISIBLE_DEVICES=1 python main.py --char_dim 128 --model_name 'charEnc_tf0.75' --train_steps 75000 --teacher_forcing_ratio 0.75
```