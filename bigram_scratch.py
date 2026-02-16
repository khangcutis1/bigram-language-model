"""
=============================================================================
  bigram_scratch.py  —  Bigram Language Model (Pure Python, No Frameworks)
=============================================================================

  Author : Arslan Haroon
  Purpose: Teach how a Bigram Language Model works from first principles,
           using ONLY Python's standard library.

  What is a Bigram Model?
  -----------------------
  A Bigram model predicts the next character (or word) based SOLELY on the
  previous one.  Formally, it models:

      P(w_t | w_{t-1})

  For example, if we've seen that after the letter 'h' the letter 'e' appears
  60% of the time, the model captures exactly that probability.

  This file implements the "Counting / Frequency" approach:
      1.  Scan the training text and COUNT how often each character follows
          every other character.
      2.  NORMALIZE those counts into probabilities (so they sum to 1).
      3.  Use the resulting lookup table to GENERATE new text and to
          EVALUATE the model with Negative Log-Likelihood (NLL).

  No external data files are needed — a small Shakespeare snippet is
  embedded directly in this script so it runs out-of-the-box.
=============================================================================
"""

# ── Standard-library imports (NO third-party packages!) ─────────────────────
from collections import defaultdict  # For conveniently counting pairs
import random                        # For sampling from distributions
import math                          # For log() used in NLL calculation


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TRAINING DATA                                                          ║
# ╠═══════════════════════════════════════════════════════════════════════════╣
# ║  We embed a small Shakespeare excerpt directly so the script can be     ║
# ║  run immediately — no downloading, no file paths, no setup.             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  BigramModelPython  —  The core model class                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class BigramModelPython:
    """
    A character-level Bigram Language Model built with pure Python.

    Internal data structures
    ------------------------
    self.bigram_counts : dict[str, dict[str, int]]
        Raw frequency counts.  bigram_counts['h']['e'] = 12 means the pair
        ('h' → 'e') appeared 12 times in training.

    self.bigram_probs : dict[str, dict[str, float]]
        Normalised probabilities derived from the counts.
        bigram_probs['h']['e'] = 0.23 means that after 'h', 'e' follows
        with probability 0.23.

    self.vocab : set[str]
        The full set of unique characters observed in the training text.
    """

    def __init__(self):
        # ── Counts table ────────────────────────────────────────────────
        # We use a defaultdict of defaultdicts so that accessing a key that
        # hasn't been seen yet automatically initialises its count to 0.
        # This avoids tedious "if key not in dict" checks everywhere.
        self.bigram_counts = defaultdict(lambda: defaultdict(int))

        # ── Probability table (filled after training) ───────────────────
        self.bigram_probs = {}

        # ── Vocabulary (set of all unique characters) ───────────────────
        self.vocab = set()

    # ─────────────────────────────────────────────────────────────────────
    #  train(text)
    # ─────────────────────────────────────────────────────────────────────
    def train(self, text: str) -> None:
        """
        Build the bigram probability table from raw text.

        Algorithm (two passes):
          Pass 1 — Count:   Slide a window of size 2 across the text and
                            increment bigram_counts[char_i][char_i+1].
          Pass 2 — Normalise: For each "context" character, divide every
                            successor count by the total count so the
                            probabilities sum to 1.0.

        Parameters
        ----------
        text : str
            The corpus to learn from (any string of characters).
        """

        # ── Step 1: Build the vocabulary ────────────────────────────────
        # We need the full set of characters so we know the "alphabet" of
        # our model.  In a word-level model this would be the vocabulary
        # of words; here each unique character is a "token".
        self.vocab = set(text)
        print(f"[TRAIN] Vocabulary size: {len(self.vocab)} unique characters")

        # ── Step 2: Count bigrams ───────────────────────────────────────
        # A "bigram" is simply a pair of consecutive characters.
        # We iterate through the text with a sliding window of size 2.
        #
        #   text:  "hello"
        #   pairs: ('h','e'), ('e','l'), ('l','l'), ('l','o')
        #
        # For each pair we increment the count:
        #   bigram_counts['h']['e'] += 1
        #   bigram_counts['e']['l'] += 1
        #   ... and so on.
        for i in range(len(text) - 1):
            current_char = text[i]      # w_{t-1}  (the "context")
            next_char    = text[i + 1]  # w_t      (the "target")
            self.bigram_counts[current_char][next_char] += 1

        # ── Step 3: Normalise counts → probabilities ────────────────────
        # For each context character, we want:
        #
        #   P(next_char | current_char) = count(current, next) / Σ count(current, *)
        #
        # This is a simple Maximum Likelihood Estimate (MLE).
        #
        # Example: if after 'h' we saw  e:12, a:5, i:3  (total=20)
        #   P(e|h) = 12/20 = 0.60
        #   P(a|h) =  5/20 = 0.25
        #   P(i|h) =  3/20 = 0.15
        #
        # These three probabilities sum to 1.0 — a valid distribution!
        for current_char, next_chars in self.bigram_counts.items():

            # Total times this character appeared as a "context"
            total = sum(next_chars.values())

            # Build the probability sub-dictionary for this context
            self.bigram_probs[current_char] = {
                next_c: count / total
                for next_c, count in next_chars.items()
            }

        # ── Recap ───────────────────────────────────────────────────────
        total_bigrams = sum(
            sum(nc.values()) for nc in self.bigram_counts.values()
        )
        print(f"[TRAIN] Total bigrams counted: {total_bigrams}")
        print(f"[TRAIN] Unique bigram types:   "
              f"{sum(len(v) for v in self.bigram_counts.values())}")
        print("[TRAIN] Training complete [OK]\n")

    # ─────────────────────────────────────────────────────────────────────
    #  generate(start_char, length)
    # ─────────────────────────────────────────────────────────────────────
    def generate(self, start_char: str, length: int = 100) -> str:
        """
        Generate new text one character at a time using the learned
        probability distribution.

        How sampling works
        ------------------
        Given the current character, we look up its probability distribution
        (the row in our table) and sample the next character according to
        those probabilities.

        Python's `random.choices` accepts a list of items and a matching
        list of weights/probabilities and returns a weighted random pick.

        Parameters
        ----------
        start_char : str
            The character to begin generation with.
        length : int
            How many characters to generate (default 100).

        Returns
        -------
        str
            The generated text string.
        """

        # Validate that the start character was seen during training
        if start_char not in self.bigram_probs:
            raise ValueError(
                f"Start character '{start_char}' was never seen during "
                f"training.  Available characters: "
                f"{sorted(self.bigram_probs.keys())}"
            )

        # Begin the output with the seed character
        current_char = start_char
        generated = [current_char]

        for _ in range(length - 1):
            # ── Look up the distribution for the current character ──────
            # If the current character has no recorded successors (very
            # unlikely with enough data, but let's be safe), we pick a
            # random character from the vocabulary as a fallback.
            if current_char not in self.bigram_probs:
                current_char = random.choice(list(self.vocab))
                generated.append(current_char)
                continue

            probs = self.bigram_probs[current_char]

            # ── Weighted random sampling ────────────────────────────────
            # `random.choices` returns a LIST of k samples; we ask for 1.
            # `population` = the possible next characters
            # `weights`    = their corresponding probabilities
            next_char = random.choices(
                population=list(probs.keys()),
                weights=list(probs.values()),
                k=1                               # we want exactly 1 sample
            )[0]                                   # extract the single item

            generated.append(next_char)
            current_char = next_char               # shift the window forward

        return "".join(generated)

    # ─────────────────────────────────────────────────────────────────────
    #  negative_log_likelihood(text)
    # ─────────────────────────────────────────────────────────────────────
    def negative_log_likelihood(self, text: str) -> float:
        """
        Compute the Negative Log-Likelihood (NLL) of the given text under
        the trained Bigram model.

        Why NLL?
        --------
        NLL tells us "how surprised the model is" by the text.

        •  A LOW  NLL means the model finds the text very plausible
           (it assigns high probabilities to the observed transitions).
        •  A HIGH NLL means the model is "surprised" — the text contains
           transitions it considers unlikely.

        Mathematical definition
        -----------------------
        Given a text  w_1, w_2, ..., w_N  the likelihood is:

            L = ∏_{t=2}^{N}  P(w_t | w_{t-1})

        We take the log (product → sum) for numerical stability:

            log L = Σ_{t=2}^{N}  log P(w_t | w_{t-1})

        Then negate and average to get the NLL:

            NLL = - (1 / (N-1)) · Σ_{t=2}^{N}  log P(w_t | w_{t-1})

        Lower is better.  A perfect model on its own training data would
        still not reach 0 because natural language has inherent randomness.

        Smoothing
        ---------
        If a bigram was NEVER seen during training, P = 0, and log(0)
        is undefined (−∞).  We apply a tiny floor value (1e-10) to avoid
        this.  In production you'd use Laplace smoothing or back-off, but
        for educational clarity we keep it simple.

        Parameters
        ----------
        text : str
            The text to evaluate.

        Returns
        -------
        float
            The average negative log-likelihood (lower is better).
        """

        log_likelihood_sum = 0.0  # Running sum of log-probabilities
        n = 0                     # Count of bigrams evaluated

        for i in range(len(text) - 1):
            current_char = text[i]
            next_char    = text[i + 1]

            # ── Retrieve P(next_char | current_char) ────────────────────
            # If the context character was seen, look up the probability
            # of the specific successor.  Otherwise, fall back to 1e-10.
            if (current_char in self.bigram_probs
                    and next_char in self.bigram_probs[current_char]):
                prob = self.bigram_probs[current_char][next_char]
            else:
                # Unseen bigram → assign a tiny probability instead of 0
                # to prevent log(0) = -infinity.
                prob = 1e-10

            # ── Accumulate log-probability ──────────────────────────────
            # math.log computes the NATURAL logarithm (base e).
            # Since 0 < prob <= 1, log(prob) is always ≤ 0.
            log_likelihood_sum += math.log(prob)
            n += 1

        # ── Average and negate ──────────────────────────────────────────
        # We divide by n (number of bigrams) to make the metric
        # comparable across texts of different lengths.
        # Negation converts the (negative) log-likelihood into a positive
        # number where LOWER = BETTER.
        nll = -log_likelihood_sum / n
        return nll


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN — Run the full pipeline                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    print("=" * 65)
    print("  Bigram Language Model — Pure Python (Counting Approach)")
    print("=" * 65, "\n")

    # ── 1. Instantiate the model ────────────────────────────────────────
    model = BigramModelPython()

    # ── 2. Train on the embedded Shakespeare snippet ────────────────────
    model.train(TRAINING_TEXT)

    # ── 3. Evaluate: Compute NLL on training data ───────────────────────
    #    (In a real project you'd use a held-out test set, but for this
    #     demo we evaluate on the training data itself.)
    nll = model.negative_log_likelihood(TRAINING_TEXT)
    print(f"[EVAL]  Negative Log-Likelihood (NLL): {nll:.4f}")
    print(f"        (Lower is better — means the model is less 'surprised')\n")

    # ── 4. Generate new text ────────────────────────────────────────────
    #    We pick a common starting character so generation is coherent.
    seed_char = "t"
    gen_length = 200
    print(f"[GEN]   Generating {gen_length} chars starting with '{seed_char}':\n")
    generated_text = model.generate(start_char=seed_char, length=gen_length)
    print("--- BEGIN GENERATED TEXT ---")
    print(generated_text)
    print("--- END GENERATED TEXT ---\n")

    # ── 5. Display a sample of the probability table ────────────────────
    #    Let's peek at the distribution after a few interesting characters
    #    so students can see what the model actually learned.
    sample_chars = ['t', 'e', ' ']
    print("[TABLE] Sample probability distributions:\n")
    for ch in sample_chars:
        if ch in model.bigram_probs:
            display_char = repr(ch)  # Makes spaces & newlines visible
            probs = model.bigram_probs[ch]

            # Sort by probability (highest first) for readability
            sorted_probs = sorted(
                probs.items(), key=lambda x: x[1], reverse=True
            )

            print(f"  After {display_char}:")
            for next_c, p in sorted_probs[:5]:  # Show top 5
                bar = '#' * int(p * 30)
                print(f"    -> {repr(next_c):5s}  {p:.4f}  ({bar})")
            print()

    print("=" * 65)
    print("  Done!  Try changing the seed character or training text.")
    print("=" * 65)
