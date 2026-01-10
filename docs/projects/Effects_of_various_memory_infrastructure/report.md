---
title: Effects of Various Memory Infrastructure
author: Amani Agrawal
date: 6 Jan 2026
---

## 1. Problem and Motivation

This project investigates what role the hidden state in a Recurrent Neural Network (RNN) actually plays in sequence prediction.

In class, we learned that RNNs pass a hidden state forward through time, allowing the model to remember past information. Intuitively, this hidden state is said to encode context. However, it is not immediately obvious whether this mechanism is fundamentally different from simply passing the last n tokens explicitly, as done in n-gram models.

Both approaches provide information about the past to predict the next token:
- n-gram models use explicit tokens with a fixed context window,
- RNNs use a learned, fixed-size hidden representation.

The central question is whether the RNN hidden state captures something qualitatively different from an explicit n-gram context, or whether it effectively behaves like a learned n-gram model.

---

## 2. Hypothesis

**Hypothesis:**  
For short-range dependency tasks, a standard RNN behaves similarly to an n-gram-style model, since only recent tokens are needed for accurate prediction. As the required dependency range increases, the RNN’s hidden state becomes increasingly advantageous, because it can compress long-range context into a fixed-size representation, while an n-gram model must explicitly expand its context window.

**Support:**  
RNN performance remains high as dependency length increases, while n-gram performance degrades.

**Refutation:**  
If n-gram models match RNN performance even for long-range dependencies.

---

## 3. Experimental Design

### Models
- A vanilla RNN trained to predict the next token in a sequence.
- An n-gram-style feedforward model that predicts using only the previous *n* tokens.
- A LSTM trained to predict the next token in a sequence.
- A RNN model with attention trained to predict the next token in a sequence.
- A Bidirectional RNN trained to predict the next token in a sequence.


### Dataset
Synthetic sequence prediction tasks are constructed where the model predicts the next token, which is a copy of an earlier token in the sequence. This allows the dependency length to be precisely controlled.

The input is an incomplete palindromic sequence, and the model must predict the next token. Sequence lengths are chosen from `{5, 7, 9, 11, 13}`. The forward part of the sequence is always more than half the total length to avoid trivial symmetry. Each valid length and forward-length combination is equally represented in the dataset.

### Tasks
- **Task 1:** Alphabetical sequences with wraparound (A → B → … → Z → A).
- **Task 2:** Fully random sequences with no alphabetical structure.

### Variables
- **Varied:** Sequence length (dependency range), model type.
- **Held Fixed:** Dataset size, training procedure, and prediction objective.

This design isolates the effect of the memory mechanism while keeping all other factors constant.

---

## 4. Results

### Short-Range Task

All models achieve perfect performance on the short-range dependency task:

- **N-gram accuracy:** 100%
- **RNN accuracy:** 100%
- **LSTM accuracy:** 100%
- **RNN with attention accuracy:** 100%
- **Bidirectional RNN accuracy:** 100%

This indicates that when the correct prediction depends only on recent context, all models—regardless of their memory mechanism—are sufficient.

---

### Long-Range Task

Performance diverges significantly as the dependency range increases:

- **N-gram accuracy:** 18%
- **RNN accuracy:** 30%
- **LSTM accuracy:** 37%
- **RNN with attention accuracy:** 23%
- **Bidirectional RNN accuracy:** 30%

While all recurrent models outperform the n-gram baseline, the gap between architectures becomes visible, with LSTMs achieving the highest accuracy.

Only aggregate accuracy values are reported; raw training logs and intermediate losses are omitted for clarity.

---

## 5. Analysis

The results confirm the central hypothesis of the project.

For the **short-range task**, all models perform equally well because the prediction depends only on local context. In this regime, explicitly passing recent tokens (as in an n-gram model) is sufficient, and the additional representational power of a recurrent hidden state provides no measurable benefit.

In contrast, the **long-range task** highlights the importance of memory mechanisms. The n-gram model performs poorly because it lacks access to tokens outside its fixed context window. Recurrent models consistently outperform it, demonstrating that hidden states allow information from earlier in the sequence to influence predictions.

Among recurrent models, the **LSTM performs best.**

Interestingly, the **RNN with attention underperforms the vanilla RNN**, despite achieving similar training loss (as shown in the notebook). A likely explanation is that Task 2 requires remembering a single specific earlier token. Vanilla RNNs may more effectively compress and preserve this information in their hidden state as compared to RNN with attention, which might get flooded with too much information at each step.

Bidirectional RNNs show modest improvement over the vanilla RNN but remain constrained.

Overall, these findings reinforce that RNN hidden states provide a genuine advantage over explicit token passing, especially as dependency length increases, and that architectural choices matter depending on the nature of the memory required.

---

## 6. Conclusion

This project demonstrates that the hidden state in an RNN provides a meaningful and functional advantage over explicitly passing recent tokens, particularly as the required dependency range increases.

The results highlight that architectural choices matter. LSTMs outperform vanilla RNNs by better preserving relevant information over time, while attention mechanisms do not necessarily improve performance in tasks that require remembering a single specific token rather than dynamically remembering tokens.

RNN's hidden states act as adaptive memory representations whose benefits become evident as task complexity increases. When memory demands exceed the capacity of a fixed-size hidden state, this naturally motivates more advanced architectures such as LSTM, bidirectional, and attention-based models.
