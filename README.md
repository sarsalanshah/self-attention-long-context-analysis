# Self-Attention and Long-Context Utilization in Transformer Models

This repository accompanies the term paper:

**"An Empirical Study of Context Scaling, Diminishing Returns, and Attention Locality in Transformers"**

---

## 📌 Overview

This project investigates how Transformer-based language models utilize long input contexts. While modern models support increasingly large context windows, we analyze whether this translates into meaningful performance gains.

The study focuses on three key questions:

- Do longer context windows improve model performance proportionally?
- Do Transformers exhibit biases toward certain token positions?
- What are the computational trade-offs of scaling context length?

---

## 🧪 Experiments

The project includes three main experimental components:

### 1. IMDB Sentiment Classification (BERT)
- Fine-tuned BERT on subsets of the IMDB dataset
- Evaluated performance across context lengths: 64, 128, 256 tokens

### 2. GPT-2 Perplexity Analysis (WikiText-2)
- Measured perplexity across varying context lengths
- Evaluated diminishing returns in language modeling

### 3. Attention Pattern Analysis
- Extracted attention weights from BERT
- Visualized positional bias and locality patterns

---

## 📊 Results

Key findings:

- Performance gains diminish beyond moderate context lengths
- Models show strong bias toward early and late tokens ("lost-in-the-middle")
- Longer contexts significantly increase computational cost without proportional benefit

---


---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sarsalanshah/self-attention-long-context-analysis.git
cd self-attention-long-context-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```
▶️ Running Experiments

1. Train BERT on IMDB
   ```bash
    python src/train_imdb_bert.py
   ```
2. Evaluate GPT-2 Perplexity
   ```bash
    python src/eval_gpt2_wikitext.py
   ```
3. Analyze Attention Patterns
    ```bash
    python src/attention_analysis.py
    ```

## 📚 Requirements

See requirements.txt for full dependency list.

## 📝 Notes
- Experiments were conducted on limited compute (local machine)
- Dataset sizes were subsampled for efficiency
- Results are intended for analysis, not state-of-the-art benchmarking



## 📧 Author

Arsalan Shah
📧 arsalanshah0402@gmail.com

## 📜 License

This project is for academic purposes.

