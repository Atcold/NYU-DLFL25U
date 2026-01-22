---
title: Backpropagation Through Depth: A Variation On Backpropagation In RNNs
author: John Quigley
date: 21 Jan 2026
---

## 1. Problem and Motivation
### 1.1 Motivation
In one of my first introductions to deep learning, I heard about the problem of vanishing and exploding gradients. Now, this can be a problem with any deep neural network, but it is particularly bad with RNNs due to their long recursive structure.

Since prevailing methods of optimization are gradient based, this problem obviously makes optimization hard because you would expect that these exponentially large or exponentially small gradients would prevent these models from converging (moving too fast, or moving too slowly). The problem is exacerbated when different layers share parameters. If there was no parameter sharing, updates might be super small, but after enough steps it would find a good minimum. But since RNNs do have parameter sharing, there is nothing preventing the later layers from having dominant gradient steps, and "overruling" any directions that early layers might want to move in.

This line of reasoning motivated my optimization philosophy. Different layers in an RNN might have different "voting power," over the direction the underlying parameters should update. For instance, an earlier layer may have "voting power" that is much too small if the gradient is vanishing. Maybe it would be helpful to give all the layers the same "voting power," so that repeated chain rules no longer deflate/inflate a layer's update direction. This is the idea we explore in this notebook. This also explains the title (which is an homage to Backpropagation Through Time), as we are backpropagating by depth rather than by time.

### 1.2 Related Work

Lots of work was done in the years (2012-2016) on the optimization of RNNs. One paper that particularly stands out in RNN optimization is "On the difficulty of training Recurrent Neural Networks" (Pascanu et. al, 2012), which introduced a clipping strategy to prevent gradients from exploding, and a regularizer to prevent gradients from being too small.

Another paper addressing this problem is "Unitary Evolution Recurrent Neural Networks" (Arjovsky et. al, 2016). This paper suggested that in the equation $h_{t+1} = \sigma(W_{hh}h_t + V_{ih}x_{t+1})$, $W_{hh}$ should be unitary, as this is the term that gets multiplies in the gradient computations. Since unitary matrices preserve vector norm, there will be no exploding or vanishing gradients.

Another paper studied the effects of normalizing gradients for every layer in deep neural networks (not including RNNs), "Block-Normalized Gradient Method" (Wei Yu et. al, 2017). In this paper they consider $\tilde y = f_n(f_{n-1}(...f_1(x)))$, and calculate $g^i_t = \frac{dL}{d\theta_i}$. Then they re-assign $g^i_t \leftarrow \frac{g^i_t}{||g^i_t||_2}$, and then they update $w_{t+1} = w_t - \tau_t \circ g_t$, so $\tau$ essentially controls the magnitude of the gradient, and we always trust the direction of the gradient. This paper takes on the same optimization philosophy that our notebook investigates, that parameter updates shouldn't decay or explode too quickly.

### 1.3 Overview

In the traditional backpropagation with SGD in RNNs, you add up all the cost functions $C(y, \tilde y) = \sum_t C(y_t, \tilde y_t)$, where $\tilde y_t$ is the prediction generated using only inputs from $(x_i){i\leq t}$. In order to understand our optimization scheme, we first consider losses one at a time (this technically isn't how the optimization scheme is implemented exactly, but its a useful simplification for now). 

Also, just for the following paragraph, we will be thinking of an RNN as a deep neural network, rather than a recursive neural network that has shared parameters. This allows us to talk about perturbing parameters in one layer without perturbing that parameter in another layer.

Let $\theta_i$ be the parameters at layer $i << t_0$

If we take the gradient $\frac{dC_{t_0}}{d\theta_i}$, there is a vanishing/exploding gradient problem here, i.e. perturbing parameters from many layers back either have a very large or very small effect on the loss function (the gradient is super small). So if we adopt the idea from Wei Yu et. al and normalize the gradient, then the direction of the step is preserved, but the magnitude is controlled (the magnitude of the update would be around the size of the learning rate). This ensures we always make reasonable progress, not too slow, not too quick, towards the minimum. Tying this back to section 1.1, by making normalizing these magnitudes, we essentially give each layer the same "voting power."

If this were a deep neural network, we would update $\theta_i \leftarrow \theta_i - \eta \frac{dC_{t_0}/d\theta_i}{||dC_{t_0}/d\theta_i||_p}$. However, since RNNs do have parameter sharing, the update we just wrote makes no sense (the parameters in each layer of the RNN are the same, so they must be updated to be the same value). The way the actual update would work would be collect all the directions $arr[i] = \frac{dC_{t_0}/d\theta_i}{||dC_{t_0}/d\theta_i||_p}$, and then update $\theta \leftarrow \theta - \eta * \sum_i arr[i]$. Each term $i$ in the sum has an approximately equal magnitude, cementing the idea that each layer has an equal "vote."

I hypothesize that an approach like this will help RNNs have a longer memory (a longstanding problem of RNNs), because now the memory layer ($i << t_0$) will have an equal say in the parameter updates, whereas before, its "vote" might have been very small.

** Comments to self (ignore if its confusing), maybe it doesn't make sense to add all these steps together, since moving a certain direction in layer 1 doesn't mean you necessarily want to move a certain direction in layer 10 (but you have to because of parameter sharing). Maybe a more nuanced rule would be decide which layer should get the largest update based on some SNR of the gradients calculated. For instance, you could weight the gradient updates instead of summing them, where the weight would increase with higher SNR of gradients (maybe cosine similarity). This is kind of what SGD does, because gradients with higher SNR move in the same direction more times, however SGD still steps with a miniscule step size when the layer is far away, which is what my algorithm would attempt to fix **

## 2. Hypothesis
I expect this new optimization algorithm to run much slower than standard backpropagation -- i.e. Backpropagation Through Depth takes $O(T^2)$ time to backpropagate through a batch, whereas Standard Backpropagation takes $O(T)$ time to backpropagate through a batch, where $T$ is the sequence length. however I think that gradient updates will be much more effective, and perhaps this efficiency (and a more cleverly/efficiently engineered algorithm) will make this a viable optimization algorithm. There also exist shortcuts that significantly cut down run time as long as you have proper domain knowledge of the problem.

## 3. Experimental Design
In order to get more technical, we introduce the Adam optimizer (used to help implement the optimization philosophy) and a mathematical decomposition of RNN gradients (which makes the interpretation of RNNs as a deep neural network more rigorous).

### 3.1 Adam

A recent paper -- "In Search of Adam's Secret Sauce" (Orvieto, Gower, 2025) -- brought to my attention a nice interpretation the Adam update rule, which I'll copy here.

$$g_k = \frac{dL}{dw}$$
$$m_k = \beta_1 * m_{k-1} + (1-\beta_1) * g_k$$
$$v_k = \beta_2 * v_{k-1} + (1-\beta_2) * g_k^2$$
$$w_{k+1} = w_k - \eta \frac{m_k}{\sqrt{v_k} + \epsilon}$$

One thing that you notice in this formulation, is that the update rule is independent of the scale of the gradients. The paper brings up that the update is essentially the same as $\frac{sign(g_k)}{\sqrt{1 + Var(g_k)/E[g_k]^2}}$. Note the signal to noise ratio in the denominator.

We will be using the Adam optimizer to do the "normalization" ($\frac{dC_{t_0}/d\theta_i}{||dC_{t_0}/d\theta_i||_p}$) for us, because it accomplishes the same thing (making updates have a size similar to the learning rate), but we don't use the exact formula in parentheses, because that formula isn't guaranteed to converge to an update of $0$. In other words, using Adam appears more principled at a first glance. More details on this exact implementation in 2.2.

### 3.2 Gradient Decomposition

Another notion I want to bring up is the decomposition of gradients briefly mentioned in Pascanu et. al.

$$h_t = W_{hh}\sigma(h_{t-1}) + W_{ih}x_t + b$$
$$\frac{dL}{d\theta} = \sum_{i=1}^T \frac{dL_t}{d\theta}$$
$$\frac{dL_t}{d\theta} = \sum_{k=1}^t (\frac{dL_t}{dx_t}\frac{dx_t}{dx_k}\frac{d^+x_k}{d\theta})$$
$$\frac{dx_t}{dx_k} = \Pi_{i=k+1}^t \frac{dx_i}{dx_{i-1}} = \Pi_{i=k+1}^t W_{hh}^Tdiag(\sigma'(x_{i-1}))$$

Here, $\frac{d^+x_k}{d\theta}$ is the "immediate" partial derivative of $x_k$ wrt $\theta$, i.e., $\frac{d^+x_k}{d\theta}$ where we pretend $x_{k-1}$ is a constant. This is the more rigorous way to think of RNN gradients, rather than pretending it is a deep neural network.

Revisiting the simplified example in 1.3, where you only consider $\frac{dC_{t_0}}{d\theta}$, $\frac{dC_{t_0}}{d\theta_i}$ would correspond to $\frac{dL_t}{dx_t}\frac{dx_t}{dx_k}\frac{d^+x_k}{d\theta} = \frac{dL_t}{dx_k}\frac{d^+x_k}{d\theta}$.

In this notebook, we will really be emphasizing the decomposition of $$\frac{dL}{d\theta} = \sum_{k\leq t} \frac{dL_t}{d\theta_k}\frac{d^+x_k}{d\theta} = \sum_{\Delta = 0}^T\sum_{k}\frac{dL_{t+\Delta}}{d\theta_k}$$, because this is how you decompose the sum of gradients into parts of equal magnitude (split into powers of $W_{hh}^{\Delta}$). Another way to say this is that we group terms according to how many times they have received an application of the "chain rule."

### 3.3 Backpropagation Through Depth Algorithm

If we only considered one loss $L_t$, then the optimization algorithm would simply be as follows.

**Algorithm: Simplified Update**
```
//Initialize Adam optimizers for different $\Delta$'s
For $k = 1, \dots, t_0$:
    AdamArr[$k$] = torch.Adam()

For input, target in data:
    //Perform one feedforward, and compute loss
    output = model(input)
    loss = criterion(target[$t_0$], output[$t_0$])
    loss.backward()

    //Store gradients
    For $k=1, \dots, t_0$:
        GradArr[$k$] = $\frac{dL_{t_0}}{dx_k}\frac{d^+x_k}{d\theta}$

    For $k=1, \dots, t_0$:
        AdamArr[$k$].zero()
        model.grad = GradArr[$k$]
        AdamArr[$k$].step()
```
If we considered all the losses $(L_t)_{t=1}^T$, then the optimization algorithm would be as follows.

**Algorithm: Full Update**
```
//Initialize Adam optimizers for different $\Delta$'s
For $k = 0, \dots, T-1$:
    AdamArr[$k$] = torch.Adam()

For input, target in data:
    //Perform one feedforward, and compute loss
    output = model(input)
    loss = criterion(target, output)
    loss.backward()

    //Store gradients
    For $t_0=1, \dots, T$:
        For $k=1, \dots, t_0$:
            GradArr[$t_0$][$k$] = $\frac{dL_{t_0}}{dx_k}\frac{d^+x_k}{d\theta}$

    //Reorganize GradArr to partition by $\Delta$
    ReorderedGradArr = []
    For $i=1, \dots, T$:
        For $\Delta = 0, \dots T-i$:
            //We only combine gradients that have similar magnitudes
            ReorderedGradArr[$\Delta$] += GradArr[$i + \Delta$][$i$] 

    For $\Delta=0, \dots, T-1$:
        AdamArr[$\Delta$].zero()
        model.grad = GradArr[$\Delta$]
        AdamArr[$\Delta$].step()
```
It is important to use separate Adam optimizer's for the optimization, because these Adam optimizers contain hidden states ($m_t, v_t$), which depend on the scale of the gradients that are fed in. Thus the gradients we feed to each Adam optimizer should have similar scale, so that they have equal voting power. ** Comment to self, does feeding those gradients into the Adam optimizer make it so that it actually has an equal vote? $$

Furthermore, the algorithmic problem we work with is EchoStep (https://github.com/Atcold/NYU-DLSP20/blob/master/09-echo_data.ipynb, credit to Alfredo Canziani).

We work with a sequence_len = 2000, batch_size = 50, BPTT_T = 20, echo = 3, epochs = 5, lr = 3e-4, betas=(0.95, 0.95)

## 4. Results
![alt text](results.png "Results")

### 4.1 Interpreting the above graph

The purple line and pink line are the test_acc and train_loss respectively, optimized using my algorithm across 5 epochs. This is averaged across 8 different trials (seeds 0-7).

The red line and orange line are the test_acc and train_loss respectively, optimized using a single Adam optimizer across 5 epochs. This is averaged across 8 different trials (seeds 0-7).

As you can see, my optimization algorithm helps reduce test_acc pretty quickly (faster than the vanilla Adam optimizer), but the train_loss never goes very low, and it is apparent that model blows up by the end of the first epoch.

Also, occassionally a benchmark trial will blow up out of nowhere, but then quickly go back to its former performance. It just happens, I don't exactly know why, but that explains the spikes of orange.

## 5. Analysis
There are a few reasons why I think this optimizer doesn't work yet, and some future directions to puruse

### 5.1 You don't want a $\Theta(lr)$ parameter update
Recall we are working with $g_{t, k} = \frac{dL_t}{dh_k}\frac{d^+h_k}{d\theta}$. Now our update for $\theta$ after receiving this gradient has a magnitude $\Theta(lr)$, due to Adam's mechanics. However, consider the case when $g_{t,k}$ is really small, then updating $\theta$ by a $\Theta(lr)$ sized update will only change $L_t$ by roughly $g_{t,k}*lr$. Maybe the "proper" optimization rule is to make $g_{t,k}*lr = \Theta(1)$. A qualitative explanation is that a $\Theta(lr)$ parameter update just isn't well conditioned, and having a $\Theta(lr)$ update can cause very sudden or very miniscule changes to the loss depending on the gradient. 

However, this problem isn't really the main focus of our research (giving different layers equal votes). Rather it seems like a very general research problem which I don't think would yield the largest benefits for the problem at hand (it probably is worthy of studying all by itself though).

### 5.2 There is too much noise in the loss landscape by doing all these updates at once
Consider the plots below. BM stands for the benchmark model, and Diff $\Delta$ stands for the accumulation of gradients $\sum_i \frac{dL_{k+\Delta}}{dx_k}\frac{d^+x_k}{d\theta}$ (I should probably rename these plots). And what is being plotted in grey is the cosine similarity of gradient updates between batch $i$ and batch $i+1$, where the idea is that if the cosine similarity is $1$, then the gradients move in the same direction from batch-to-batch, and there is less noise involved. The red line is a moving average of the past 20 cosine similarities.

In theory, Adam's SNR aspect should be tuning out noisy gradients (and the step sizes when gradients are noisy should be small), but maybe my beta's are too large or $\frac{1}{\sqrt{1 + x^2}}$ doesn't decrease fast enough. You can see this noise, especially in diffs 3, 4, 5, in the first epoch, where the cosine similarity oscillates between -1 and 1, suggesting that it really doesn't know the right direction to step. These moderately sized updates might be confusing the loss landscapes of other optimizers, i.e. by giving every $\Delta$ the same voting power, we are listening to very noisy gradients.


## 6. Conclusion / Teaching Moment / Limitations / Next Steps

### What did we learn?
Giving each depth of the network the same "voting power" is not a viable optimization algorithm. At least not the naive way we did it here. We had a heuristic idea that might work, but had no analytical/mathematical backing, so these are just one of those things you try just in case it works like magic.

### What misconception does this clarify?
Sometimes people talk about vanishing gradients as a big problem, but  they aren't always a problem, sometimes they are just an artifact of the loss landscape and shouldn't be meddled with, without proper analytic reasoning. Amplifying or shrinking gradients may confuse the optimizer or amplify the error of the NN.

### Limitations
- We only experimented with AdamW and also did not perform a thorough hyperparameter search (learning rate, lambda, betas)
- We only experimented with a single toy problem

### Next Steps
The first thing I want to do is try an optimizer that is different from Adam. Maybe SGD with the $L_p$ normalization. The idea is that Adam is not guaranteed to converge, and often pushes for better minima (thanks to Alf for the idea). Asking ChatGPT confirms this, and cites a paper Reddi et. al "On the Convergence of Adam and Beyond." Maybe using AMSGrad, which has convergence guarantees, or an optimization algorithm that thinks very carefully about step sizes would be helpful. I believe that this is my main problem, because as you can see in the appendix, $W_{hh}$ consistently grows, and if I don't use gradient clipping, the gradient magnitudes will grow well into the 100s.

Another idea closely related to 4.2, is to consider each direction a layer or a $\Delta$ wants to move in, and reweight these based on how noisy the past gradients have been. This would avoid the problem of giving super noisy gradients such as diffs 3, 4, and 5 equal votes and making parameter steps noisy. That would be another thing to explore, but fixing the blow up is a prerequisite to exploring this.

## Appendix A
Here are some graphs that show cosine similarity of gradient updates for each algorithm.

Cosine similarity is a popular proxy for the signal-to-noise ratio of a gradient. We take the cosine similarity of gradients in batch($i$) and batch($i+1$) i.e. after every step. If the cosine similarity is 1, then there was a high signal in the last gradient/step we took, because the parameters want to continue moving in that direction. If the cosine similarity is -1, then there was a lot of noise in the last gradient/step we took, because the parameters now want to move in the opposite direction of our last step (want to undo the last update).

Some things to notice in the graphs are periods of time where the cosine similarities sustain a value of 1. I interpret this as a large/good breakthrough in optimization, and the loss should be decreasing quickly during this period. On the other hand, moments where the cosine simliarity is far from 1 may occur because the optimization problem is really hard, or the model has already reached a minimum and is just stepping randomly (no large breakthroughs left).

Some examples are the first 500 batches in the "Standard Adam Optimization Algorithm" graph, if you align this with the loss graph, you see that the period of 1 cosine similarity corresponds with a fast dip in loss.

Lets also consider each of the "Backpropagation Through Depth" graphs. A priori, I'd expect that the Depth 2 graph would have a cosine similarity close to 1 for the longest period of time, because the Depth 2 optimizer is the optimizer that helps the RNN identify what symbol is being echoed (it is the "recall optimizer"). I'd also expect Depth 0, 1 optimizer's to be the next most signal optimizers, for they need to store in the hidden state what the Depth 2 module is recalling. And every other optimizer I would expect the gradients to be pure noise.

This seems to hold true for the first 0.5 epochs, but then I assume it blows up and all signal is lost.

![alt text](standard_optim_algo_cossim.png)

![alt text](BPTD_optim_algo_cossim0.png)
![alt text](BPTD_optim_algo_cossim1.png)
![alt text](BPTD_optim_algo_cossim2.png)
![alt text](BPTD_optim_algo_cossim3.png)
![alt text](BPTD_optim_algo_cossim4.png)
![alt text](BPTD_optim_algo_cossim5.png)
![alt text](BPTD_optim_algo_cossim6.png)

## Appendix B
Here are some graphs showing the evolution of the Adam parameters in the Backpropagation Through Depth algorithm. 

The most important parameters (weight_hh) are the ones that have a steadily increasing gradient. 

This shows that the problem we are encountering in training is an exploding gradient (even when we clip gradients to a norm of 1). 

Even though the Adam steps are of moderate size, the EMA's that Adam tracks become dominated by the exploding gradients.

This is just a note of something to keep in mind if I plan to revisit this algorithm further down the line.

![alt text](BPTD_adam_params.png)