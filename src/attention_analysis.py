import os
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def average_attention_distance(attentions, attention_mask):
    # attentions: tuple of layers, each [batch, heads, seq, seq]
    layer_distances = []

    mask = attention_mask.cpu().numpy()

    for layer_attn in attentions:
        attn = layer_attn.detach().cpu().numpy()
        batch_size, num_heads, seq_len, _ = attn.shape

        head_distances = []

        for b in range(batch_size):
            valid_len = int(mask[b].sum())
            if valid_len < 2:
                continue

            for h in range(num_heads):
                a = attn[b, h, :valid_len, :valid_len]
                distances = np.abs(
                    np.arange(valid_len)[:, None] - np.arange(valid_len)[None, :]
                )
                weighted_distance = (a * distances).sum() / (a.sum() + 1e-12)
                head_distances.append(weighted_distance)

        layer_distances.append(float(np.mean(head_distances)))

    return layer_distances


def main(max_length=128, num_samples=100):
    device = get_device()
    model_dir = f"results/imdb_bert_len{max_length}/checkpoint-2500"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, output_attentions=True)
    model.to(device)
    model.eval()

    dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(num_samples))

    all_layer_distances = []

    with torch.no_grad():
        for sample in dataset:
            enc = tokenizer(
                sample["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

            layer_distances = average_attention_distance(outputs.attentions, attention_mask)
            all_layer_distances.append(layer_distances)

    mean_distances = np.mean(np.array(all_layer_distances), axis=0).tolist()

    output_path = f"results/imdb_bert_len{max_length}/attention_distance.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "max_length": max_length,
                "mean_attention_distance_per_layer": mean_distances,
            },
            f,
            indent=2,
        )

    print({
        "max_length": max_length,
        "mean_attention_distance_per_layer": mean_distances,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    main(max_length=args.max_length, num_samples=args.num_samples)
