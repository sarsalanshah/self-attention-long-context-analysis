import math
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(context_length=128, max_samples=500):
    device = get_device()
    model_name = "gpt2"
    output_dir = f"results/wikitext_gpt2_len{context_length}"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    texts = [x["text"] for x in dataset if x["text"].strip()]
    texts = texts[:max_samples]

    losses = []

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=context_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()
            losses.append(loss)

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)

    metrics = {
        "context_length": context_length,
        "avg_loss": avg_loss,
        "perplexity": ppl,
        "num_samples": len(losses),
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    main(context_length=args.context_length, max_samples=args.max_samples)
