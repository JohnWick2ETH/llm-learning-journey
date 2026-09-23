#!/bin/bash

######################################################################
############## OpenWebText dataset ###################################
######################################################################

# 1. train the tokenizer given the input text
#  the output will be `vocab.json` and `merges.txt`.

if [ ! -f "../data/owt_vocab.json" ]; then
    # profile with py-spy
    if [[ -n "${PROFILE:-}" ]]; then
        uv run --with py-spy py-spy record --format raw --full-filenames \
            --subprocesses --output profile.pyspy --rate 5 --threads -- \
            python train_tokenizer.py --input ../data/owt_train.txt --data_set owt \
            --vocab_size 32000 --special_tokens "<|endoftext|>" \
            --artifacts ../data/
    else
        uv run python train_tokenizer.py --input ../data/owt_train.txt --data_set owt \
            --vocab_size 32000 --special_tokens "<|endoftext|>" \
            --artifacts ../data/
    fi
fi


# 2. encode the input text given the tokenizer from step 1
#   the output will be a file of tokens for the given input text

uv run python token_encoder.py --vocab ../data/owt_vocab.json --merges ../data/owt_merges.txt \
    --input ../data/owt_train.txt --output ../data/owt_tokens.txt

# 3. train the transformer based language model
