---
title: Using Positional Embedding for classifying similar images with Attention model
author: Amanuel Nigussie Demeke
date: 25 Jan 2026
---

## 1. Problem and Motivation
**CNN** performs well with images due to assumptions the model makes about the image's structure. These assumptions, or **'Inductive bais'**, include locality, stationarity, compositionality, and more. However, if the image's **spatial structure is disturbed**, CNN fails to perform effectively. One such disturbance is the **jigsaw transformation**, where an image is divided into **patches** and the patches are randomly rearranged. This process breaks some meaningful neighborhood relationships. As a result, CNN struggles to classify jigsawed images.

One way to classify jigsawed images is using **attention-based models**. Unlike CNN, attention mechanisms are **permutation invariant**; patches of images are treated as **tokens**, and their order isn't assumed. However, permutation invariance also introduces a limitation. Often, the order of tokens results in **different meanings** (e.g., *"horse eats apple"* vs *"apple eats horse"*). If two images have **similar-looking patches**, attention will struggle to classify them, as those images are different not mostly based on their spatial structure but rather their **geometry**. In such cases, we can experiment using **attention with positional embedding** to understand how patches are arranged in space.


## 2. Hypothesis
When using **attention-based models** for image classification, **positional embeddings** are necessary to distinguish images that share similar sets of patches(tokens). While a permutation-invariant attention model may succeed on classes with clearly **distinct** patch content (e.g., *“1” vs. “8”*), it will fail on classes where the label is primarily determined by **spatial arrangement and geometry** rather than patch appearance (e.g., *“6” vs. “9”*).


## 3. Experimental Design

### Models
We evaluate three models.  
1) A **convolutional neural network**, which serves as a baseline for both normal and jigsawed images.  
2) A **patch-based self-attention model without positional embeddings**. This model treats image patches as an unordered set and is therefore permutation-invariant.  
3) The **same self-attention architecture with positional embeddings**, allowing it to learn the positions of the patches.

### Dataset

We use the **MNIST dataset**, which contains ten classes corresponding to digits from 0 to 9. In addition to the original images, we construct a **jigsawed version** of the dataset by dividing each image into a 7×7 grid of patches (**49 patches**) and randomly rearranging these patches. The visual content of each patch is preserved, while the spatial arrangement is disrupted. For each jigsawed image, the patch permutation is kept constant.

---
All attention models are intentionally minimal. They consist of **single-layer, single-head self-attention** followed by the same **feed-forward classifier** architecture. This design choice ensures that any differences in performance are not due to model depth or capacity.

The primary variable in this experiment is the **presence or absence of positional embeddings** in the attention model. All other factors are held constant, including the dataset, patch size, training procedure, optimizer, number of parameters, and classifier architecture. The CNN baseline is kept fixed for comparison and to illustrate how disturbing inductive biases affects model performance.

The **positional embeddings** used in this setup are implemented as **learnable vectors**. Each of the 49 patches in an image is assigned a unique embedding that acts as a **tag**. This allows the model to distinguish similar patches based on location. These embeddings are added to their corresponding patch feature vectors, associating appearance with position. Unlike fixed positional encodings or explicit coordinate-based representations *(x, y)*, these embeddings do not directly encode geometry; instead, they are learned from data.

This setup directly tests the hypothesis by isolating the role of positional information. Because some MNIST classes share similar patch content, successful classification requires reasoning about geometry rather than appearance alone. Comparing attention models with and without positional embeddings therefore provides a clear and sufficient test of whether positional information is necessary for attention-based image classification.



## 4. Results

### Overall Test Accuracy

| Model | Normal MNIST | Jigsaw MNIST |
|-------|-------------|--------------|
| CNN | 97.90% | 11.18% |
| Attention (No Positional Embedding) | 30.31% | 30.31% |
| Attention (+ Positional Embedding) | 70.90% | 70.90% |

Since a random classifier achieves **10%** accuracy, both the CNN and the attention model with positional embeddings demonstrate strong performance on the original MNIST dataset. However, on the jigsawed dataset, the CNN’s performance **decreases substantially**, while the performance of both attention models **remains the same**.


### Digit Group Accuracy (MNIST)

| Digit Group | Attention (No PE) | Attention (+ PE) |
|-------------|------------------|------------------|
| 1 & 8 | 94.26% | 94.41% |
| 6 & 9 | 65.37% | 92.63% |

Given that a random classifier achieves **50%** accuracy, both attention models perform very well on digit pairs such as **1 and 8**. For digit pairs such as **6 and 9**, the model with positional embeddings shows a significant improvement over the average accuracy of the attention model without positional embeddings.

---


## 5. Analysis

### Jigsaw vs. Normal Dataset

The contrast between the normal and jigsawed datasets is to highlight the importance of inductive bias for CNN and permutation invariance in attention models.

CNN's high accuracy (97.90%) on the original MNIST drops to near-random (11.18%) performance when the dataset is jigsawed. This sharp decline reflects CNN’s reliance on inductive bias. Convolutional filters are designed to detect local patterns under the assumption that nearby pixels are meaningfully related. When the image is divided into patches and randomly rearranged, this spatial structure is destroyed. Even though the pixel content remains unchanged. As a result, CNN fails to extract a meaningful relationship out of the jigsawed image.

In contrast, the attention models show identical performance on both the normal and jigsawed datasets (30.31% accuracy without positional embeddings and 70.90% accuracy with positional embeddings). By design, the jigsawed patches are converted into tokens. As long as each patch can be represented as a token, shuffling the patches does not affect the model’s internal computations, since the model treats image patches as an unordered set of tokens. Rearranging them within the image therefore does not change how they are processed. However, the model’s accuracy depends on how much discriminative information can be extracted from each patch. The more informative the patches, the better the model’s performance.

### Attention With vs. Without Positional Embeddings

This comparison isolates the role of positional information within attention-based models and directly tests our hypothesis.

Without positional embeddings, the attention model achieves 30.31% overall accuracy. Adding learnable positional embeddings increases performance to 70.90%. This substantial improvement indicates that positional information enables the model to encode spatial relationships between patches.

Before adding positional embeddings, the arrangement of patches in the embedding space depends solely on **pixel content**. This may be sufficient if patches from different images are highly distinctive. However, the MNIST dataset consists of black-and-white handwritten digits composed of simple curves and lines. When images are divided into patches, many patches contain very similar visual content within the same digit image and across different digits. For example, the cross in “4” may also appear as part of “7,” and vertical lines may appear in “1,” “4,” or “7.” This similarity is further exaggerated because most of the image content consists of black background, meaning many patches are simply black. When such similar patches are embedded based only on pixel values, their embedding vectors become highly similar, if not identical. As a result, the model struggles to learn meaningful relationships because there is little or no discriminative information.

One way to mitigate this issue is to introduce a logical assumption that gives those patches meaning when they are part of an image, namely their position in the image. A horizontal line patch alone does not provide much information for either us or the model. However, if we know that this patch is located in the upper half of the image, we can more confidently conclude that it belongs to "7" rather than "2". If all patches are associated with a specific location, the model can learn how these patches should be arranged. By simply adding a unique learnable vector (i.e., a positional embedding) to each patch embedding, similar patches in different locations are projected differently in the embedding space. As a result, the attention model can learn more meaningful relationships and structural properties.

The digit-pair analysis further clarifies this effect. For digit pairs such as **1 and 8**, both models perform similarly (~94%). These digits can largely be distinguished using patch-level appearance alone: one is mostly a vertical line, while the other is constructed from multiple curves. However, for digit pairs such as **6 and 9**, the difference is substantial: 65.37% without positional embeddings versus 92.63% with positional embeddings. These digits share highly similar patches, and their difference mainly comes from their orientation and spatial arrangement in the image. The model needs to understand that the circular structure in "6" is located in the lower half of the image, whereas in "9" it appears in the upper half. This becomes possible with the help of positional embeddings.

## 6. Conclusion

This experiment demonstrates that positional embeddings are necessary for attention-based models to classify images that are similar and differ primarily in their spatial arrangement.

We observed that CNNs perform extremely well when spatial structure is preserved, but their performance collapses when that structure is disrupted. On the other hand, attention models remain unaffected by patch shuffling, as long as the patches are converted into tokens.

A common misconception is that attention models automatically “understand” structure. In reality, without positional information, they cannot distinguish different arrangements of the same components. Structure must either be assumed (as in CNNs) or explicitly learned (as in positional embeddings).

Overall, this experiment clarifies that a model's performance depends not only on model size or depth, but also on the assumptions built into the architecture.

## 7. Limitations

The attention architecture used in this study was intentionally minimal (single-layer, single-head). While this design isolates the role of positional embeddings, deeper or multi-head attention models may behave differently and could potentially compensate for missing positional information.

Finally, positional embeddings were implemented as learnable vectors to illustrate how the model can learn positional information and why this property is necessary. However, since spatial position is usually assumed to be known, it could instead be encoded explicitly using sinusoidal positional encodings or a 2D coordinate-based representation.
