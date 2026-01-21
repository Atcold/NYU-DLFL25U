# RNN Memory Capacity

## 1. Conceptual question
When the distance between critical information and the prediction target increases, does the memory capacity of recurrent neural networks (RNNs) gradually decay or suddenly collapse?

Specifically: as the time interval (K) between critical contextual information and the prediction target in a sequence gradually increases, does the RNN's prediction performance gradually decline, or does a significant performance drop occur after reaching a certain critical distance K*?

---

## 2. Hypothesis

Recurrent Neural Networks (RNNs) have a limited effective memory capacity in their hidden states. When the time interval (denoted as K) between crucial contextual information and the prediction target exceeds a certain threshold K*, the model performance exhibits a non-linear, sudden decline (i.e., the "memory cliff"), rather than a smooth, gradual decay.

---

## 3. Experimental design

**What is being compared?**
- RNN performance at different memory distances K (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25)
- RNN performance with different hidden layer sizes (16, 32, 64, 128, 256)

**What is kept fixed?**
- Model architecture: single-layer RNN with ReLU activation
- Embedding dimension: 32
- Training iterations: 3000 (Experiment 1), 2000 (Experiment 2)
- Learning rate: 0.001
- Task type: Cloze task (predicting a word that appeared K steps earlier)
- Vocabulary and dataset: same cloze tasks for all experiments

**What is varied?**
- Memory distance K: the number of filler words between the key information and the prediction target
- Hidden layer size H: 16, 32, 64, 128, 256

**Why this experiment is sufficient to test the hypothesis:**
- The cloze test naturally measures memory capacity because the task inherently requires long-range memory of words; if the model forgets the keywords, it cannot predict correctly.
- By systematically varying the value of K, we can observe whether performance degrades gradually or collapses suddenly.
- By changing the hidden layer size, we can test whether memory capacity is linearly related to the representation dimension.
- Accuracy below the random baseline indicates complete forgetting, providing a clear threshold for memory failure.


---

## 4. Results
**Experiment 1: Memory Distance vs Accuracy (H=64)**

| K | Accuracy | Status |
|---|----------|--------|
| 0 | 100% | Strong |
| 1 | 100% | Strong |
| 2 | 99.8% | Strong |
| 3 | 97.4% | Strong |
| 4 | 19.8% | Weak |
| 5+ | 0% | Forgotten |

Key observation: Accuracy drops from 97% to 0% within just 2 steps (K=3 → K=5). This is a cliff, not a slope.

**Experiment 2: Hidden Size vs Capacity Limit K\***

| Hidden Size | K* (Capacity Limit) |
|-------------|---------------------|
| 16 | 2 |
| 32 | 4 |
| 64 | 4 |
| 128 | 5 |
| 256 | 5 |

Key observation: Hidden size increases 16× (16→256), but K* only increases from 2 to 5.

## 5. Analysis

**Why the cliff?** From the RNN update equation:
$$h_t = \sigma(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t)$$

Information from step 0 reaching step K is approximately scaled by $\lambda^K$, where $\lambda$ is related to the spectral radius of $W_{hh}$. This is **exponential decay**.

However, the accuracy doesn't decay exponentially, but rather remains at a high level before suddenly dropping. Why is this?

**Threshold Effect:** As long as the signal strength exceeds the discrimination threshold, the model can predict correctly. Once the signal strength falls below the noise level, the accuracy immediately drops to a random level. The "cliff" point is where the signal strength is approximately equal to the noise level.

**Why sublinear scaling?** A larger hidden state can store more information at each step, but the **recurrent structure** still leads to exponential decay. Larger capacity can delay the appearance of the "cliff," but cannot eliminate it.

## 6. Conclusion / Teaching Moment

**What did we learn?**
- The hidden state of an RNN is an information bottleneck with strict capacity limitations.
- Memory decay manifests as a sudden drop (cliff effect) due to threshold effects, even though the underlying signal decay is gradual.
- Increasing the hidden layer size is inefficient for extending memory span.

**What misconceptions does this clarify?**
- "RNNs can handle sequences of arbitrary length"—theoretically yes, but in practice they have a hard memory limit K*.
- "Larger models mean proportionally increased memory capacity"—this is not the case; the performance improvement is not linear.

**Connection to Transformers:** This explains why attention mechanisms were invented. Attention mechanisms bypass the hidden state bottleneck by directly accessing all previous positions (Q·K), thus completely avoiding exponential decay.

## 7. Limitations
Small vocabulary/dataset: Only 8 cloze tasks with 75-word vocabulary. Results may not generalize to natural language scale.