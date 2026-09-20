#!/bin/bash

######################################################################
############## TinyStories dataset ###################################
######################################################################

# 1. train the tokenizer given the input text
#  the output will be `vocab.json` and `merges.txt`.

# profile with py-spy
if [ ! -f "../data/tinystories_vocab.json" ]; then
    if [[ -n "${PROFILE:-}" ]]; then
        uv run --with py-spy py-spy record --format raw --full-filenames \
            --subprocesses --output profile.pyspy --rate 5 --threads -- \
            python train_tokenizer.py --input ../data/TinyStoriesV2-GPT4-train.txt \
            --data_set tinystories --vocab_size 10000 --special_tokens "<|endoftext|>" \
            --artifacts ../data/
    else
        uv run python train_tokenizer.py --input ../data/TinyStoriesV2-GPT4-train.txt \
            --data_set tinystories --vocab_size 10000 --special_tokens "<|endoftext|>" \
            --artifacts ../data/
    fi
fi


# 2. encode the input text given the tokenizer from step 1
#   the output will be a file of tokens for the given input text
if [ ! -f '../data/tinystories_token.txt' ]; then
    uv run python token_encoder.py --vocab ../data/tinystories_vocab.json --merges ../data/tinystories_merges.txt \
        --input ../data/TinyStoriesV2-GPT4-train.txt --output ../data/tinystories_token.txt
fi

# 3. encode the validation text to get validation token
if [ ! -f '../data/tinystories_valid_token.txt' ]; then
    uv run python token_encoder.py --vocab ../data/tinystories_vocab.json --merges ../data/tinystories_merges.txt \
        --input ../data/TinyStoriesV2-GPT4-valid.txt --output ../data/tinystories_valid_token.txt
fi

# 4. train the transformer based language model
uv run python train_model.py \
    --tokens ../data/tinystories_token.txt \
    --valid_tokens ../data/tinystories_valid_token.txt \
    --model_config ./model_config.json \
    --vocab_size 10000 \
    --training_config ./training_config.json 
