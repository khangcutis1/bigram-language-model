"""
=============================================================================
  bigram_nn.py  --  Bigram Language Model (PyTorch / Neural Network)
=============================================================================

  Author : Arslan Haroon
  Purpose: Re-implement the exact same Bigram model from bigram_scratch.py,
           but this time using PyTorch and Deep Learning concepts.

  Why re-implement with a Neural Network?
  ---------------------------------------
  The "counting" approach in bigram_scratch.py is simple and fast, but it
  doesn't scale.  Real language models (GPT, LLaMA, etc.) cannot simply
  count -- they learn complex patterns via gradient descent.

  This file bridges the gap:
      - We show that a single-layer neural network trained with gradient
        descent converges to the EXACT SAME probability table that the
        counting method produces.
      - This proves that "learning" via backpropagation is just a
        general-purpose way to discover statistical patterns.

  Architecture (intentionally minimal)
  ------------------------------------
      Input (one-hot character)
          |
          v
      nn.Embedding  <-- acts like a lookup table of learnable weights
          |
          v
      Linear (no hidden layers!)
          |
          v
      Softmax -> Probability distribution over next character

  Because there are NO hidden layers, this network can only capture
  bigram-level dependencies -- exactly the same as the counting model.
=============================================================================
"""

# -- PyTorch & standard-library imports ---------------------------------------
import torch                        # The core tensor library
import torch.nn as nn               # Neural network building blocks
import torch.nn.functional as F     # Functional API (softmax, etc.)
from torch.utils.data import (      # Utilities for batching data
    Dataset, DataLoader
)
import random                       # For setting seeds
import math                         # For log comparisons


# =============================================================================
#   TRAINING DATA  (same Shakespeare snippet as bigram_scratch.py)
# =============================================================================

TRAINING_TEXT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""


# =============================================================================
#   VOCABULARY BUILDING
# -----------------------------------------------------------------------------
#   Neural networks work with NUMBERS, not characters.  So we need to
#   create a mapping:   character <-> integer index
#
#   'a' -> 0,  'b' -> 1,  ...  exactly like a dictionary.
# =============================================================================

# sorted() gives us a deterministic ordering every time we run
chars = sorted(list(set(TRAINING_TEXT)))
VOCAB_SIZE = len(chars)

# stoi = "string to integer"  --  maps each character to a unique index
stoi = {ch: i for i, ch in enumerate(chars)}

# itos = "integer to string"  --  the reverse mapping
itos = {i: ch for i, ch in enumerate(chars)}

print("[VOCAB] {} unique characters".format(VOCAB_SIZE))
print("[VOCAB] Example: 'a' -> {},  'e' -> {},  ' ' -> {}\n".format(
    stoi.get('a', '?'), stoi.get('e', '?'), stoi.get(' ', '?')))


# =============================================================================
#   DATASET CLASS
# -----------------------------------------------------------------------------
#   PyTorch's DataLoader needs a Dataset object.  Our dataset is simply
#   all consecutive character pairs (bigrams) from the training text.
#
#   Each sample is:  (x, y)  where x = index of current char,
#                                  y = index of next char.
# =============================================================================

class BigramDataset(Dataset):
    """
    A PyTorch Dataset that yields (input_index, target_index) pairs from
    consecutive characters in the training text.

    Why a Dataset class?
    --------------------
    PyTorch's DataLoader can then automatically:
      - Shuffle the data each epoch (reduces bias).
      - Split it into mini-batches (faster, more stable training).
      - Optionally load data in parallel with multiple workers.
    """

    def __init__(self, text: str):
        # Convert each character to its integer index
        self.data = [stoi[ch] for ch in text]

    def __len__(self):
        # Number of bigrams = number of characters minus 1
        # (because the last character has no successor)
        return len(self.data) - 1

    def __getitem__(self, idx):
        # Return (current_char_index, next_char_index) as tensors
        # dtype=torch.long because PyTorch embeddings require integer indices
        x = torch.tensor(self.data[idx],     dtype=torch.long)
        y = torch.tensor(self.data[idx + 1], dtype=torch.long)
        return x, y


# =============================================================================
#   NEURAL NETWORK MODEL
# =============================================================================

class BigramNeuralNet(nn.Module):
    """
    A single-layer neural network that learns bigram probabilities.

    Architecture
    ------------
    1. nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)
       - Think of this as a VOCAB_SIZE x VOCAB_SIZE matrix of learnable
         weights.
       - When we feed in the index of character 'h' (say index 7), the
         embedding layer returns ROW 7 of this matrix.
       - That row IS the model's learned "logits" (unnormalised scores)
         for every possible next character.

    Why Embedding and not Linear?
    -----------------------------
    Mathematically, embedding lookup on a one-hot vector is IDENTICAL to
    a matrix-vector multiplication (Linear layer).  But nn.Embedding is:
      - More memory-efficient (no need to materialise the one-hot vector).
      - Conceptually clearer: "look up the row for this character".

    Why no hidden layers?
    ---------------------
    With no hidden layers, each input character maps DIRECTLY to an
    output distribution -- the model can only capture pair-wise (bigram)
    statistics.  Adding hidden layers would let it capture longer
    patterns (trigrams, etc.), but we intentionally keep it simple.
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        # -- The Embedding layer ------------------------------------------
        # Shape: (vocab_size, vocab_size)
        # Row i contains the logits for "what comes after character i".
        #
        # Initially these weights are random -- the training loop will
        # adjust them via gradient descent until they match the true
        # bigram probabilities observed in the data.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,   # How many characters exist
            embedding_dim=vocab_size     # Output size = vocab (one logit per char)
        )

    def forward(self, x):
        """
        Forward pass: character index -> logits for next character.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size,)
            Batch of input character indices.

        Returns
        -------
        logits : torch.Tensor, shape (batch_size, vocab_size)
            Unnormalised scores for each possible next character.
            To get probabilities, apply softmax (done inside the loss
            function for numerical stability).
        """
        # Look up the embedding row for each input character
        # Shape: (batch_size,) -> (batch_size, vocab_size)
        logits = self.embedding(x)
        return logits


# =============================================================================
#   TRAINING LOOP
# =============================================================================

def train_model(model, dataloader, epochs=200, lr=0.05):
    """
    Train the BigramNeuralNet using gradient descent.

    Training loop anatomy (every epoch)
    ------------------------------------
    1. FORWARD PASS  -- Feed input through the network to get predictions.
    2. LOSS          -- Compare predictions with ground truth using
                        CrossEntropyLoss (= Softmax + NLL Loss combined).
    3. BACKWARD PASS -- Compute gradients (dLoss/dweights) via
                        backpropagation.
    4. OPTIMIZER STEP -- Update weights:  w = w - lr * gradient

    Why CrossEntropyLoss?
    ---------------------
    CrossEntropyLoss = Softmax + Negative-Log-Likelihood in one step.
    This is numerically more stable than doing softmax first and then
    taking log, because it avoids potential underflow/overflow.

    It directly optimises the same NLL metric we computed by hand in
    bigram_scratch.py, so the two approaches are truly equivalent.

    Parameters
    ----------
    model : BigramNeuralNet
    dataloader : DataLoader
    epochs : int
    lr : float   (learning rate -- how big each gradient step is)
    """

    # -- Loss function --------------------------------------------------------
    # CrossEntropyLoss expects:
    #   input  = logits of shape (N, C)   N=batch, C=classes
    #   target = class indices of shape (N,)
    criterion = nn.CrossEntropyLoss()

    # -- Optimiser ------------------------------------------------------------
    # SGD = Stochastic Gradient Descent, the simplest optimiser.
    # lr (learning rate) controls step size.  Too high -> overshooting,
    # too low -> slow convergence.  0.05 is a reasonable start.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print("[TRAIN] Starting training for {} epochs, lr={}\n".format(epochs, lr))

    for epoch in range(epochs):

        epoch_loss = 0.0  # Track total loss for this epoch
        n_batches = 0

        for x_batch, y_batch in dataloader:

            # -- Step 1: Forward pass -----------------------------------------
            # Pass the input character indices through the network to get
            # the predicted logits (unnormalised scores) for the next char.
            logits = model(x_batch)    # shape: (batch_size, VOCAB_SIZE)

            # -- Step 2: Compute loss -----------------------------------------
            # CrossEntropyLoss internally:
            #   a) applies softmax to convert logits -> probabilities
            #   b) takes the negative log of the probability at the
            #      correct target index -> this is the NLL
            #   c) averages over the batch
            loss = criterion(logits, y_batch)

            # -- Step 3: Backward pass (backpropagation) ----------------------
            # Zero out old gradients -- PyTorch ACCUMULATES gradients by
            # default (useful for some techniques, but usually we want
            # fresh gradients each step).
            optimizer.zero_grad()

            # Compute dLoss/dw for every learnable weight in the model.
            # This single call traverses the entire computation graph
            # backwards and fills in the .grad attribute of each parameter.
            loss.backward()

            # -- Step 4: Update weights ---------------------------------------
            # Apply the classic SGD rule:  w_new = w_old  -  lr * grad
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # -- Logging ----------------------------------------------------------
        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print("  Epoch {:>4d}/{}  |  Avg Loss: {:.4f}".format(
                epoch + 1, epochs, avg_loss))

    print("\n[TRAIN] Training complete [OK]\n")


# =============================================================================
#   TEXT GENERATION
# =============================================================================

def generate(model, start_char: str, length: int = 100) -> str:
    """
    Generate text character-by-character from the trained neural net.

    Process for each character
    --------------------------
    1. Feed the current character's index into the model -> logits.
    2. Apply softmax to convert logits -> probabilities.
    3. Sample from the probability distribution using
       torch.multinomial (the PyTorch equivalent of random.choices).
    4. The sampled index becomes the next input.

    Parameters
    ----------
    model : BigramNeuralNet
    start_char : str
    length : int

    Returns
    -------
    str
    """
    model.eval()  # Switch to evaluation mode (disables dropout, etc.)

    # Convert the seed character to its integer index
    current_idx = torch.tensor([stoi[start_char]], dtype=torch.long)
    generated = [start_char]

    # torch.no_grad() tells PyTorch we don't need gradients during
    # generation -- this saves memory and speeds things up.
    with torch.no_grad():
        for _ in range(length - 1):
            # Forward pass -> logits
            logits = model(current_idx)           # shape: (1, VOCAB_SIZE)

            # Convert logits to probabilities with softmax
            # dim=-1 means "apply softmax along the last dimension"
            probs = F.softmax(logits, dim=-1)     # shape: (1, VOCAB_SIZE)

            # Sample from the distribution
            # multinomial draws 1 sample according to the probabilities
            next_idx = torch.multinomial(probs, num_samples=1)  # shape: (1, 1)

            # Decode index -> character and append
            generated.append(itos[next_idx.item()])

            # The sampled character becomes the new input
            current_idx = next_idx.squeeze(0)     # shape: (1,)

    model.train()  # Switch back to training mode
    return "".join(generated)


# =============================================================================
#   WEIGHT INSPECTION -- Proving equivalence to the counting method
# =============================================================================

def show_learned_probabilities(model, sample_chars=None):
    """
    Extract the probability table from the neural network's weights and
    display it.

    Why this matters
    ----------------
    After training, the embedding matrix holds LOGITS.  If we apply
    softmax to each row, we get the learned probability distribution
    P(next_char | current_char) -- which should closely match the raw
    counts from bigram_scratch.py.

    This is the "Aha!" moment:
        Gradient descent on CrossEntropyLoss discovers the same
        probabilities that simple counting gives you.
    """
    if sample_chars is None:
        sample_chars = ['t', 'e', ' ']

    # Grab the embedding weight matrix -- shape: (VOCAB_SIZE, VOCAB_SIZE)
    # .detach() disconnects it from the computation graph (no gradients).
    weights = model.embedding.weight.detach()

    # Apply softmax to each row -> probabilities
    probs = F.softmax(weights, dim=-1)

    print("[WEIGHTS] Learned probability distributions (from NN weights):\n")

    for ch in sample_chars:
        idx = stoi[ch]
        row = probs[idx]  # Probability distribution for this character

        # Get top-5 most likely next characters
        top5_values, top5_indices = torch.topk(row, k=5)

        display_char = repr(ch)
        print("  After {}:".format(display_char))
        for val, next_idx in zip(top5_values, top5_indices):
            next_ch = itos[next_idx.item()]
            p = val.item()
            bar = '#' * int(p * 30)
            print("    -> {:5s}  {:.4f}  ({})".format(
                repr(next_ch), p, bar))
        print()


# =============================================================================
#   COMPARISON -- Side-by-side with the counting method
# =============================================================================

def compare_with_counting():
    """
    Import the pure-Python model, train it on the same data, and compare
    its probability table with the neural network's learned weights.

    This function prints a side-by-side comparison for a few characters
    so students can see the convergence.
    """
    try:
        from bigram_scratch import BigramModelPython
    except ImportError:
        print("[COMPARE] Could not import bigram_scratch.py -- skipping "
              "comparison.\n"
              "          Make sure both files are in the same directory.\n")
        return

    # Train the counting model on the identical text
    print("[COMPARE] Training the counting model for comparison...\n")
    counting_model = BigramModelPython()
    counting_model.train(TRAINING_TEXT)

    # Get the NN's probability matrix
    weights = model.embedding.weight.detach()
    nn_probs = F.softmax(weights, dim=-1)

    # Compare for a few characters
    sample_chars = ['t', 'e', ' ', 'a']
    print("[COMPARE] Side-by-side: Counting vs Neural Network\n")
    print("  {:<6} {:<6} {:>10} {:>10} {:>10}".format(
        'Char', 'Next', 'Counting', 'NN', 'Diff'))
    print("  {:<6} {:<6} {:>10} {:>10} {:>10}".format(
        '-' * 6, '-' * 6, '-' * 10, '-' * 10, '-' * 10))

    for ch in sample_chars:
        if ch not in counting_model.bigram_probs:
            continue

        idx = stoi[ch]
        counting_dist = counting_model.bigram_probs[ch]

        # Get top-3 from counting model
        sorted_counting = sorted(
            counting_dist.items(), key=lambda x: x[1], reverse=True
        )[:3]

        for next_ch, count_p in sorted_counting:
            if next_ch not in stoi:
                continue
            next_idx = stoi[next_ch]
            nn_p = nn_probs[idx][next_idx].item()
            diff = abs(count_p - nn_p)
            print("  {:<6} {:<6} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                repr(ch), repr(next_ch), count_p, nn_p, diff))

    print("\n  (Small differences are expected -- the NN uses gradient")
    print("   descent which converges approximately, not exactly.)\n")


# =============================================================================
#   MAIN -- Orchestrate the full pipeline
# =============================================================================

if __name__ == "__main__":

    # -- Reproducibility ------------------------------------------------------
    # Setting seeds ensures you get the same results every run, which is
    # crucial for debugging and teaching.
    torch.manual_seed(42)
    random.seed(42)

    print("=" * 65)
    print("  Bigram Language Model -- PyTorch (Gradient-Based Approach)")
    print("=" * 65, "\n")

    # -- 1. Prepare the dataset & dataloader ----------------------------------
    dataset = BigramDataset(TRAINING_TEXT)
    print("[DATA]  Total bigram samples: {}\n".format(len(dataset)))

    # batch_size=64 means we process 64 bigram pairs per gradient step.
    # shuffle=True randomises the order each epoch to prevent the model
    # from memorising the sequence of training examples.
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # -- 2. Create the model --------------------------------------------------
    model = BigramNeuralNet(vocab_size=VOCAB_SIZE)
    print("[MODEL] Parameters: {:,}".format(
        sum(p.numel() for p in model.parameters())))
    print("[MODEL] Architecture:\n{}\n".format(model))

    # -- 3. Train -------------------------------------------------------------
    train_model(model, dataloader, epochs=200, lr=0.05)

    # -- 4. Generate text -----------------------------------------------------
    seed_char = "t"
    gen_length = 200
    print("[GEN]   Generating {} chars starting with '{}':\n".format(
        gen_length, seed_char))
    generated_text = generate(model, start_char=seed_char, length=gen_length)
    print("--- BEGIN GENERATED TEXT ---")
    print(generated_text)
    print("--- END GENERATED TEXT ---\n")

    # -- 5. Inspect the learned weights ---------------------------------------
    show_learned_probabilities(model)

    # -- 6. Compare with the counting approach --------------------------------
    compare_with_counting()

    print("=" * 65)
    print("  Done!  Compare the output above with bigram_scratch.py.")
    print("  The probabilities should be very similar -- proof that")
    print("  gradient descent discovers the same patterns as counting!")
    print("=" * 65)
