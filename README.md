<div align="center">

# 🧠 Bigram Language Model — From Zero to PyTorch

**A hands-on, educational repository that teaches you how language models work by building one from scratch.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

*Learn the fundamentals that power GPT, LLaMA, and every modern language model — starting with the simplest possible model: the Bigram.*

</div>

---

## 📖 Table of Contents

- [What is a Bigram Model?](#-what-is-a-bigram-model)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Counting vs. Gradient-Based Approach](#-counting-vs-gradient-based-approach)
- [Key Takeaway](#-key-takeaway)
- [References & Further Reading](#-references--further-reading)

---

## 🎓 What is a Bigram Model?

A **Bigram Language Model** predicts the next character (or word) based **only** on the immediately preceding one. It's the simplest non-trivial language model and the perfect starting point for understanding how all language models work.

### The Core Equation

$$P(w_t \mid w_{t-1})$$

> *"The probability of character $w_t$ given that the previous character was $w_{t-1}$."*

For an entire sequence $w_1, w_2, \ldots, w_N$, the model assumes:

$$P(w_1, w_2, \ldots, w_N) = \prod_{t=2}^{N} P(w_t \mid w_{t-1})$$

This is called the **Markov assumption** — the future depends only on the present, not the past.

### Example

Given the training text `"hello"`, the bigram counts are:

| Context ($w_{t-1}$) | Next ($w_t$) | Count | Probability |
|:---:|:---:|:---:|:---:|
| `h` | `e` | 1 | 1.00 |
| `e` | `l` | 1 | 1.00 |
| `l` | `l` | 1 | 0.50 |
| `l` | `o` | 1 | 0.50 |

After seeing `l`, the model assigns equal probability to `l` and `o`.

### Evaluating the Model — Negative Log-Likelihood (NLL)

We measure how "surprised" the model is by a text using **NLL**:

$$\text{NLL} = -\frac{1}{N-1} \sum_{t=2}^{N} \log P(w_t \mid w_{t-1})$$

- **Lower NLL** → the model finds the text plausible (good fit).
- **Higher NLL** → the model is "surprised" (poor fit).

---

## 📁 Repository Structure

```
biagram-models/
│
├── bigram_scratch.py    # 🐍 Pure Python — Counting approach (no frameworks)
├── bigram_nn.py         # 🔥 PyTorch    — Neural network approach
└── README.md            # 📖 You are here
```

| File | Approach | Dependencies | Key Concept |
|:---|:---|:---|:---|
| `bigram_scratch.py` | Counting & Normalisation | Python stdlib only | Maximum Likelihood Estimation |
| `bigram_nn.py` | Gradient Descent | PyTorch | Embeddings, Cross-Entropy Loss |

---

## 🚀 How to Run

### Prerequisites

- **Python 3.8+** (any recent version works)
- **PyTorch** (only for the neural network version)

```bash
# Install PyTorch (if you haven't already)
pip install torch
```

### Run the Pure Python Version

```bash
python bigram_scratch.py
```

This will:
1. Train the bigram model by counting character pairs
2. Print the Negative Log-Likelihood (NLL) score
3. Generate 200 characters of new "Shakespeare-like" text
4. Display the learned probability distributions

### Run the PyTorch Version

```bash
python bigram_nn.py
```

This will:
1. Train a single-layer neural network for 200 epochs
2. Print the loss decreasing over time
3. Generate 200 characters of new text
4. **Compare** the neural network's learned weights with the counting method

> 💡 **No data files needed!** Both scripts include a Shakespeare snippet directly in the code.

---

## ⚖️ Counting vs. Gradient-Based Approach

This repository implements the **exact same model** in two fundamentally different ways. Here's why that matters:

### Counting Approach (`bigram_scratch.py`)

```
Text → Count Pairs → Normalise → Probability Table → Done!
```

- **How:** Scan the text, count every `(char_a, char_b)` pair, divide by totals.
- **Pros:** Fast, exact, easy to understand.
- **Cons:** Doesn't scale — you can't "count" your way to GPT.

### Gradient-Based Approach (`bigram_nn.py`)

```
Text → Neural Net → Loss → Backprop → Update Weights → Repeat → Done!
```

- **How:** A neural network starts with random weights and iteratively adjusts them to minimise CrossEntropyLoss (which is mathematically equivalent to NLL).
- **Pros:** Scales to billions of parameters (GPT, LLaMA, etc.).
- **Cons:** Slower, approximate (requires many iterations to converge).

### Side-by-Side Comparison

| Aspect | Counting | Neural Network |
|:---|:---|:---|
| **Speed** | ⚡ Instant | 🐢 Iterative (many epochs) |
| **Accuracy** | ✅ Exact MLE | ≈ Approximate (converges) |
| **Scalability** | ❌ Bigrams only | ✅ Scales to any model size |
| **Interpretability** | ✅ Direct counts | 🔍 Weights need softmax |
| **Foundation for** | Traditional NLP | Modern Deep Learning |

---

## 💡 Key Takeaway

> **Gradient descent on CrossEntropyLoss discovers the same probability table that simple counting gives you.**

This is the "Aha!" moment of the repository. Run both scripts and compare the outputs — the neural network's learned weights (after softmax) will closely match the counting model's probability table.

This means that when we scale up to GPT-sized models with billions of parameters, the training process is conceptually doing the same thing — just over much more complex patterns than simple bigrams.

---

## 📚 References & Further Reading

| Resource | Description |
|:---|:---|
| [Andrej Karpathy — "makemore"](https://github.com/karpathy/makemore) | The inspiration for this repository. Karpathy's brilliant series building language models from scratch. |
| [PyTorch Documentation](https://pytorch.org/docs/stable/) | Official docs for all the PyTorch modules used here. |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | When you're ready to go beyond Bigrams and into modern Transformers. |

---

<div align="center">

**If this repo helped you understand language models, consider giving it a ⭐!**

Made with ❤️ for the ML learning community.

</div>
