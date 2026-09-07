#!/bin/bash

# 1. train the tokenizer given the input text
#  the output will be `vocab.json` and `merges.txt`.
python3 train_tokenizer.py --input tinystories

# 2. encode the input text given the tokenizer from step 1
#   the output will be a file of tokens for the given input text

python3 token_encoder.py --vocab vocab.json --merges merges.txt \
    --input train.txt --output train.tokens --dtype uint16

# 3. train the transformer based language model
