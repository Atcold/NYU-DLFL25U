---
title: Do NNs actually need the parameters?
author: Ray Verma
date: 29 Jan 2026
---

## 1. Problem and Motivation 
I take the idea from lectures regarding CNNs: in natural signals, most of the parameters of a Fully Connected Model do not contribute much to the result.
I take this one step further, to see if these models 'waste' parameters on other datasets as well. If so, can we boil the model down into a smaller subset of parameters that may be computationally more efficient.


## 2. Hypothesis (≈ 30 seconds)
A fully connected Neural network is dense, i.e each 'neuron' is connected to _all_ other neurons in a layer. However, in most cases, a given feature in a dataset is not well related to _every_ other feature.  
Due to this, I hypothesize that in arbitrary datasets, a significant proportion of the parameters in an NN are useless, i.e removing them does not effect the performance of the model.  


## 3. Experimental Design (≈ 2 minutes)
For a given dataset and a trained NN, I vary the number of parameters retained, while keeping all other parameters unchanged. I then compare different proportions of retained parameters, and analyze the point at which the performance drops. 
Importantly, I choose the initial architecture of the NN to be the smallest model that achieves reasonable performance. This accounts for the possibility that the NN could be overparameterized to begin with, making the experiment essentially useless. 
This control makes the experiment sufficient to test the hypothesis. 

**Overview**
1. On a given dataset, find the smallest model that achieves reasonable performance. 
    - I limit to 1 layer NNs for clarity and consistency. One can easily check that the results hold for multiple layers as well. 
    - Importantly, we pick the model which is much better than a slightly smaller one.
2. Train this model completely.
3. Start pruning away weights with smallest magnitude ( $|w_i| \approx 0$ )
4. Check the accuracy of this pruned model

Extra:

5. Store the initial (untrained) weights of the complete model. 
6. Re-train the pruned model using these weights.
6. Check accuracy of this model.  

**Datasets**
I test on the following datasets for a diversity of data types
1. `Swiss Roll`: Classic toy 3D dataset from sklearn (spatial data)
2. `Diabetes`: Another classic sklearn dataset that aims to classify whether a sample has diabetes based on given features (real-world data)
3. `MNIST`: For benchmarking, since we already know most params are wasted here (visual data)


## 4. Results
Over 3 datasets: Swiss Roll, Diabetes and MNIST, we consistently have:
1. Chosen an architecture s.t any fewer parameters leads to large decrease in accuracy
2. However, we see that if we prune low magnitude weights from a large model, it performs equally as well (until a certain extent).
3. Importantly, the pruned accuracy is greater than if we trained a model with the same number of weights from scratch.
4. Furthermore, if we train the pruned model with the same initialization, it continues to perform well with larger amount of pruning.
5. Overall, we can remove around 80% of parameters from a given Fully Connected NN, without much loss in performance!


## 5. Analysis and Conclusion
It seems that the density of fully connected NNs is a little overkill, where most parameters end up being wasted. In fact, for our datasets, this is almost 80% of them! 
Real data often has redundant, unnecessary features that do not need to be fed into our models. Vizualizations in the notebook showcase how the pruning process effectively removes some of these issues. 

This means that if we identify the useful ones, we can drastically reduce the size of our model, leading to more efficient models for inference, just like CNNs! 


The main caveat, however, is that we don't know beforehand which parameters will be 'useful'. For an arbitrary dataset, we do not have rules like locality, invariance or heirarchicality. Thus, we can't perform pruning before the training process. Moreover, the set of 'useful' parameters changes based on the initialization of the model weights before training, making this process even harder! 

Upon further research, I found that this is an active area of work called the "Lottery Ticket Hypothesis", where a specific model initialization is called the 'lottery ticket' which determines which parameters 'win' or 'lose'. It was introduced in this 2018 paper: ["The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"](https://arxiv.org/abs/1803.03635). 
Since then there has been quite some work on finding a 'winning ticket' that will maximize the number of useful parameters in our model, thereby allowing us to reduce the overall architecture size. 

Fun! 
