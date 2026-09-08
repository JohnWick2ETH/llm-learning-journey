#!/bin/bash

######################################################################
############## OpenWebText dataset ###################################
######################################################################

# 1. train the tokenizer given the input text
#  the output will be `vocab.json` and `merges.txt`.

# profile with py-spy
if [[ -n "${PROFILE:-}" ]]; then
    uv run --with py-spy py-spy record --format raw --full-filenames \
        --subprocesses --output profile.pyspy --rate 5 --threads -- \
        python train_tokenizer.py --input ../data/owt_train.txt \
        --vocab_size 32000 --special_tokens "<|endoftoken|>" \
        --artifacts ../data/
else
    uv run python train_tokenizer.py --input ../data/owt_train.txt \
        --vocab_size 32000 --special_tokens "<|endoftoken|>" \
        --artifacts ../data/
fi


# 2. encode the input text given the tokenizer from step 1
#   the output will be a file of tokens for the given input text

# python3 token_encoder.py --vocab vocab.json --merges merges.txt \
#    --input train.txt --output train.tokens --dtype uint16

# 3. train the transformer based language model
