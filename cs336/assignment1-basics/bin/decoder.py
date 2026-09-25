import argparse
import json
import torch
from base64 import b64decode
from cs336_basics.training import load_model_from_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.utils import try_gpu, softmax


# decode from transformer based model
def main(args):
    with open(args.model_config) as f:
        mc = json.load(f)

    with open(args.vocab) as f:
        vocab = json.load(f)
        # recover token id -> token string
        token_id_map = dict()
        token_map = dict()
        for string, id in vocab.items():
            t = b64decode(string)
            token_id_map[id] = t
            token_map[t] = id

    device = try_gpu()
    # 1. load model from transformer training checkpoint
    model = TransformerLM(
        vocab_size=mc["vocab_size"],
        context_length=mc["context_length"],
        d_model=mc["d_model"],
        num_layers=mc["num_layers"],
        num_heads=mc["num_heads"],
        d_ff=mc["d_ff"],
        rope_theta=mc["rope_theta"],
        device=device,
    )

    load_model_from_checkpoint(args.model_checkpoint, model)

    # 2. generate tokens repeatedly
    #   sample from next token prediction distribution
    # todo: load user prompt from command line
    user_prompt = [
        [
            token_map[b"I"],
            token_map[b"'m"],
            token_map[b" "],
            token_map[b"happy"],
        ]
    ]

    running_token_seq = torch.Tensor(user_prompt).to(torch.long).to(device)
    model.eval()
    for _ in range(128):
        # logits tensor is of shape (batch_size, seq_len, vocab_size)
        with torch.no_grad():
            logits = model(running_token_seq)

        # option 1: low-temperature sampling
        tau = 0.1
        adjusted_logits = logits / tau
        probs = softmax(adjusted_logits, dim_i=2)

        next_token_id = torch.multinomial(probs[:, -1, :], num_samples=1).item()
        next_token = torch.tensor([[next_token_id]], device=running_token_seq.device)
        running_token_seq = torch.cat([running_token_seq, next_token], dim=1)

    # 3. decode tokens back to text
    return "".join(
        token_id_map[id.item()].decode("utf-8") for id in running_token_seq[0, :]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("decoder")

    parser.add_argument("--model_config", required=True)
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--vocab", required=True)

    args = parser.parse_args()

    text = main(args)
    print(text)
