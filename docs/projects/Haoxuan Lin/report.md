# Effects of Activation Functions in Deep CNN–RNN Models

**Author:** Haoxuan Lin  
**Course:** NYU – Introduction to Deep Learning (DLFL25U)  
**Date:** January 6, 2026  

---

## 1. Problem and Motivation

Activation functions introduce nonlinearity into neural networks and play a critical role in determining how information and gradients propagate through layers. Among them, **ReLU** is one of the most widely used activation functions due to its simplicity, computational efficiency, and strong empirical performance. However, ReLU is also known to suffer from the *dying ReLU* problem: when a neuron's pre-activation becomes negative, its output is clamped to zero, and the gradient through that neuron is exactly zero. Once this happens persistently, the neuron effectively stops learning.

A commonly proposed remedy is **Leaky ReLU**, which modifies the negative input region by introducing a small, non-zero slope. In theory, this ensures that gradients never become exactly zero, allowing neurons to recover even if they fall into the negative regime. However, in practice, the negative slope parameter **α is often chosen to be very small** (e.g., α ≪ 1). As a result, the negative branch behaves *almost like zero*, and in shallow networks the behavior of Leaky ReLU can be nearly indistinguishable from that of standard ReLU.

This observation raises a subtle but important question:

> If α is tiny, are Leaky ReLU–style activations truly different from ReLU in practice, or do they only differ in theory? Moreover, how do these small differences interact with increasing network depth?

The key insight motivating this project is that **small effects can accumulate across many layers**. While a single multiplication by a small α may have a negligible impact, repeated attenuation of activations and gradients in the negative region can progressively weaken signal propagation as depth increases. Over many layers, this accumulation may significantly affect optimization dynamics and model performance.

This project therefore aims to study how activation functions that are nearly identical in shallow settings can lead to meaningfully different behaviors in deeper end-to-end models, and to understand when such differences help or hinder training.

---
## 2. Hypothesis

**Hypothesis:**  
For shallow networks, ReLU and small-α variants behave similarly because the negative branch is *nearly zero*. As depth increases, repeated scaling by α increasingly attenuates signal and gradients, worsening optimization and performance—especially for more aggressive negative-slope variants.

**Support:**  
Accuracy decreases as depth increases, and negative-slope variants degrade faster than standard ReLU/Leaky ReLU at intermediate depth.

**Refutation:**  
If small-α variants consistently outperform ReLU as depth increases.


---

## 3. Experimental Design
### Models
- **ReLU** CNN–RNN model (baseline)
- **Leaky ReLU** CNN–RNN model with small negative slope α
- **Negative-slope variant** CNN–RNN model (non-standard variant applied on negative inputs)

All other architecture and training settings are held fixed across experiments.

### Dataset / Task
We use **MNIST** for a digit-to-word prediction task. Given an image of a handwritten digit, the model generates a token sequence for the digit name (e.g., `<b>three<e>`). This combines:
- visual feature extraction (CNN encoder)
- sequential token prediction (RNN decoder)

### Variables
- **Varied:** encoder depth, activation function
- **Held fixed:** dataset, optimizer, training procedure, decoder architecture, objective

---

## 4. Results

### Shallow Models

For shallow networks, all activation functions achieve similar performance:

- **ReLU (depth = 0): 87.35%**

This result indicates that when network depth is limited, the precise behavior of the activation function in the negative input region is largely irrelevant. Because the negative slope parameter α is very small, both Leaky ReLU and negative-slope variants behave almost identically to ReLU near zero. As a result, no meaningful performance difference is observed in the shallow regime.

---

### Deep Models

As network depth increases, accuracy consistently degrades across all activation functions:

- **Depth 6:**  
  - ReLU: 78.42%  
  - Leaky ReLU: 76.70%  
  - Negative-slope: 72.01%

- **Depth 12:**  
  - ReLU: 75.23%  
  - Leaky ReLU: 73.36%  
  - Negative-slope: 60.20%

- **Depth 30:**  
  - All variants collapse to approximately **20% accuracy**

At intermediate depths (6 and 12), negative-slope activations degrade more rapidly than ReLU or Leaky ReLU. This suggests that although gradients are technically non-zero, repeated scaling by a very small α across many layers progressively weakens signal and gradient propagation, making optimization more difficult.

At extreme depth (30), all activation variants fail similarly, indicating that overall optimization barriers dominate and that activation-level modifications alone are insufficient to maintain trainability.


---

## 5. Analysis

These results highlight that small activation differences can accumulate into significant training effects in deep networks.

Because the negative slope α is very small, both Leaky ReLU and negative-slope variants behave almost like ReLU near zero. In shallow networks, this makes positive and negative regions effectively equivalent, and no meaningful performance difference is observed.

However, as depth increases, repeated multiplication by α in the negative activation region leads to progressive attenuation of signal and gradients. Even though gradients are technically non-zero, they become extremely small after passing through many layers, effectively behaving as zero.

For the negative-slope variant, this effect is further amplified. The additional sign inversion or scaling applied to negative activations introduces extra distortion into feature representations. Over many layers, this leads to unstable optimization and worse performance compared to standard ReLU or Leaky ReLU.

At extreme depth (depth = 30), optimization difficulties dominate. Gradient attenuation, signal distortion, and limited training time collectively cause all models to fail, indicating that activation-level fixes alone are insufficient to ensure trainability at this scale.

---

## 6. Conclusion

This project demonstrates that while activation function variants such as Leaky ReLU introduce non-zero gradients in theory, their practical effectiveness depends critically on network depth.

When the negative slope parameter α is very small, positive and negative activations behave almost equivalently in shallow networks. As depth increases, however, repeated scaling by α progressively weakens gradient flow and degrades training performance. In some cases, negative-slope variants can even worsen optimization relative to standard ReLU.

These findings suggest that activation function tweaks alone are not a complete solution for deep network training. Structural mechanisms such as residual connections or normalization strategies are likely required to maintain stable learning at greater depths.