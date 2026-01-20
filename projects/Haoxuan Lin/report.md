# Effects of Various Memory Infrastructure

**Author:** Amani Agrawal  
**Course:** NYU – Introduction to Deep Learning (DLFL25U)  
**Date:** January 6, 2026  

---

## 1. Problem and Motivation

This project investigates the functional role of the hidden state in Recurrent Neural Networks (RNNs) for sequence prediction tasks.

In class, we learned that RNNs propagate a hidden state forward through time, allowing the model to retain information about previous inputs. This hidden state is commonly described as encoding *context*. However, it is not immediately clear whether this mechanism is fundamentally different from explicitly providing recent tokens, as done in classical n-gram models.

Both approaches expose past information to predict the next token:

- **n-gram models** explicitly pass a fixed window of previous tokens.
- **RNNs** compress past information into a learned, fixed-size hidden representation.

The central question of this project is whether the RNN hidden state captures something qualitatively different from an explicit n-gram context, or whether it effectively behaves like a learned n-gram model.

---

## 2. Hypothesis

**Hypothesis**

For short-range dependency tasks, a standard RNN behaves similarly to an n-gram-style model, since only recent tokens are required for accurate prediction. As the required dependency range increases, the RNN hidden state becomes increasingly advantageous, because it can compress long-range context into a fixed-size representation, while an n-gram model must explicitly expand its context window.

**Support**

- RNN performance remains high as dependency length increases.
- N-gram performance degrades as dependency length increases.

**Refutation**

- If n-gram models match RNN performance even for long-range dependencies.

---

## 3. Experimental Design

### Models

The following models are compared under identical training conditions:

- A vanilla RNN trained for next-token prediction
- An n-gram-style feedforward model using the previous *n* tokens
- A Long Short-Term Memory (LSTM) network
- An RNN augmented with attention
- A Bidirectional RNN

### Dataset

Synthetic sequence prediction tasks are constructed such that the correct next token is a copy of an earlier token in the sequence. This allows the dependency length to be precisely controlled.

Each input sequence is an incomplete palindromic structure. The forward portion of the sequence always exceeds half of the total length to avoid trivial symmetry. Sequence lengths are drawn from `{5, 7, 9, 11, 13}`, and all valid length–structure combinations are equally represented.

### Tasks

Two task variants are used:

- **Task 1:** Alphabetical sequences with wraparound (A → B → … → Z → A)
- **Task 2:** Fully random sequences with no exploitable alphabetical structure

### Variables

- **Varied:** sequence length (dependency range), model architecture  
- **Held fixed:** dataset size, training procedure, prediction objective  

This design isolates the effect of the memory mechanism itself.

---

## 4. Results

### Short-Range Dependency Task

All models achieve perfect accuracy when dependencies are short:

- N-gram: 100%
- RNN: 100%
- LSTM: 100%
- RNN with attention: 100%
- Bidirectional RNN: 100%

This indicates that when correct predictions depend only on recent context, all models are sufficient regardless of their memory mechanism.

---

### Long-Range Dependency Task

Performance diverges significantly as dependency length increases:

- N-gram: 18%
- RNN: 30%
- LSTM: 37%
- RNN with attention: 23%
- Bidirectional RNN: 30%

All recurrent models outperform the n-gram baseline, with LSTMs achieving the highest accuracy. Only aggregate accuracy values are reported for clarity.

---

## 5. Analysis

The results support the central hypothesis of this project.

For short-range tasks, explicitly passing recent tokens is sufficient, and the hidden state of an RNN provides no measurable advantage. In contrast, long-range tasks expose the limitations of fixed-window models. The n-gram model fails because it cannot access tokens outside its predefined context window.

Recurrent models outperform n-gram baselines by maintaining a hidden state that allows information from earlier positions in the sequence to influence predictions.

Among recurrent architectures, LSTMs perform best, reflecting their ability to preserve relevant information over longer time spans. Interestingly, the RNN with attention underperforms the vanilla RNN despite similar training loss. A likely explanation is that the task requires recalling a single specific token rather than dynamically aggregating multiple contextual cues, making attention less effective in this setting.

Bidirectional RNNs show modest improvement but remain constrained by the fixed hidden-state representation.

---

## 6. Conclusion

This project demonstrates that the hidden state in an RNN provides a meaningful functional advantage over explicitly passing recent tokens, particularly as dependency length increases.

While n-gram models are sufficient for short-range tasks, recurrent architectures become essential for long-range dependencies. Architectural choices matter: LSTMs outperform vanilla RNNs by better preserving relevant information, while attention mechanisms do not universally improve performance in tasks that require remembering a single specific token.

Overall, RNN hidden states act as adaptive memory representations whose benefits emerge as task complexity increases, motivating the use of more structured memory mechanisms when long-range dependencies are present.
