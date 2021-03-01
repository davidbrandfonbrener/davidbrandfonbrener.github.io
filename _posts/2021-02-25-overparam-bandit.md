---
title: What happens when you use big neural nets for offline decision making?
subtitle: A case for action-stable algorithms
date: 2021-02-25

---

Recent results in supervised learning suggest that while overparameterized models have the capacity to overfit, they in fact generalize quite well[^zhang][^belkin]. We wanted to know whether these models also work well in decision making problems. Rather than going straight to the full RL problem that includes temporal credit assignment and exploration, we decided to start with an offline contextual bandit problem since this is most similar to the supervised problem. This lets us isolate the effects of the fact that decision making problems only reveal the outcome of the selected action instead of all possible actions, i.e. we only see *bandit feedback*. 

This post will go through an example problem to introduce some of the main results from our recent paper [Offline Contextual Bandits with Overparameterized Models](https://arxiv.org/abs/2006.15368). Briefly, our main results are the following:

1. We find that policy-based algorithms can struggle with serious overfitting problems when value based algorithms do not.  
2. We introduce the concept of an *action-stable* objective to explain this phenomena. An objective is action-stable at a state if there exists a prediction (action distribution or action-value vector) which optimizes the algorithm's objective no matter which action is observed at that state.

---

## Running example

### Data

For the rest of the post we will work with this simple offline contextual bandit problem with $ d $-dimensional states and 2 actions:

- States/contexts $ s_i \in \R^d $ are sampled iid from an isotropic, zero-mean Gaussian: $ s_i\sim \mathcal{N}(0, I)$
- Actions $ a_i \in$ \{0,1\}​ are chosen uniformly at random by the behavior policy $ \beta$.
- Full reward vectors $ r_i \in \R^2$ are a linear function of state, plus Gaussian noise: $ r_i = \theta^\top s_i + \epsilon_i$ where $ \epsilon_i \sim  \mathcal{N}(0, \epsilon I)$. The data only contains the *observed* rewards $ r_i(a_i)$, which is the reward vector indexed by the selected action.

We will assume that the algorithm has access to the behavior $ \beta$ since issues of estimating $ \beta$ are orthogonal to our results. For our experiments we will sample $ \theta\in \R^{d\times 2}$ uniformly from $[0,1]^{d\times 2}$, we will set $ d=10$, $ \epsilon = 0.1$, and each dataset will have a training set of 100 datapoints and an independent test set of 500 datapoints. 

We can think of this problem as being perhaps the easiest or most natural problem to try out. So any algorithms that struggle on this problem should immediately raise some red flags.

### Algorithms

We will evaluate two algorithms which we call *policy-based* and *value-based*. There are lots of variations of these algorithms and ways to combine them, but these two broad categories capture most of the algorithms that people use and sufficient to illustrate our main results. Below we define these algorithms formally.

*Policy-based* optimizes a policy to maximize an importance weighted estimate of the policy's value:


$$
\hat \pi = \arg\sup_{\pi\in \Pi}\frac{1}{N}\sum_{i=1}^N r_i(a_i) \frac{\pi(a_i\mid s_i)}{\beta(a_i\mid s_i)}
$$

*Value-based* learns a Q function by minimizing the mean squared error and then returns a greedy policy  $ \pi_{\widehat Q}$:

$$
\widehat Q = \arg\inf_{Q \in \mathcal{Q}} \sum_{i=1}^N (Q(s_i, a_i) - r_i(a_i))^2
$$



Both of these algorithms have nice guarantees when we use small model classes[^swam2] [^chen]. But, here we care about what happens when we have really big neural nets as our policies and Q functions, which makes these guarantees vacuous. Practically, in our running example we will use one layer MLPs with width 512 which is more than enough to fit nearly linear functions in 10 dimensions on 100 datapoints.

### Results

Now we can generate some results. Our measure of success will be the *regret* which is defined as


$$
\text{Regret}(\pi) = V(\pi^*) - V(\pi) = \mathbb{E}_{s}\mathbb{E}_{a \sim \pi^* \mid s} \mathbb{E}_{r\mid s}[r(a)] - \mathbb{E}_{s}\mathbb{E}_{a \sim \pi \mid s} \mathbb{E}_{r\mid s}[r(a)]
$$


We run 50 seeds, each corresponding to an independent sample of $s_i, a_i, r_i $ tuples, and plot the results in the figure below. Better policies have lower regret and the optimal policy has zero regret. We include the regret of a random policy to get a sense of scale and find that policy-based algorithms perform much worse than value-based algorithms.

<figure><img src="/assets/img/overparam_bandit/blog_bar.png" width="500"/></figure>

---

## What's going on here?

This contrast is pretty stark. Policy-based algorithms do much worse than value-based and not even so much better than random. But to get a better idea of what's going on we introduce the concept of ***action-stability***.

It's easiest to understand action-stability through a simple thought experiment. Take the dataset $ S $ and construct a perturbed $ \widetilde S $ where we leave all the states $ s_i$ and full reward vectors $ r_i $ the same, but we re-sample the actions from an independent sample from $ \beta$. Since nothing about the environment has changed, we know that the optimal policy remains the same. So we would hope that our learning objective would have the following property: there exists a single model which is optimal (with respect to that objective) on both $S$ and $\widetilde S$. We say that such an objective is *action-stable* because it has an optimal policy which is stable to re-sampling of the actions in the dataset. A more formal definition can be found in the [paper](https://arxiv.org/abs/2006.15368).

<figure><img src="/assets/img/overparam_bandit/toy_stability.png" width="700"/></figure>

The figure shows the results from conducting this thought experiment on our running example problem. We measure the TV distance between pairs of policies trained on datasets with re-sampled actions on a held out test set of states.  We find that over 20 re-samplings of the actions, the policy-based algorithm learns substantially different policies depending on the seed. In contrast, the value based algorithm always learns approximately the same policy. This behavior suggests that the policy-based algorithm is *overfitting* based on the *observed actions*. This is different from overfitting to the *labels* in supervised learning since unlike labels, we don't think that the observed action should change the underlying optimal policy.

This idea of overfitting can also be seen in the learning curves. Below we show the learning curves on one seed. The objectives are the ones from above (estimated value and MSE respectively) and "train value" is the value of the policy estimated at the states in the training set. While the policy-based objective keeps increasing, the value of the policy keeps decreasing. This supports the idea that action-instability is causing overfitting.

<figure><img src="/assets/img/overparam_bandit/toy_learning.png" width="700"/></figure>

But *why* is the policy-based objective not action-stable? To see this, It is useful to inspect the policy-based objective more carefully at just one datapoint:


$$
r_i(a_i)\frac{\hat \pi(a_i\mid s_i)}{\beta(a_i\mid s_i)}.
$$


If $$ r_i(a_i) > 0 $$ then we can maximize this objective by setting $$ \hat \pi(a_i\mid s_i) = 1 $$. But if $$ r_i(a') > 0 $$ for some other action $ a'$, then if $$ \beta $$ had chosen $$ a' $$ we would instead have set $$ \hat \pi(a'\mid s_i) =1 $$. Since $$ \hat \pi(\cdot\mid s_i) $$ must be a distribution that sums to 1, these two policies are mutually exclusive. This sort of phenomena makes the objective unstable to re-sampling the actions. This becomes especially noticeable when we use a big neural net model that can actually maximize our objective at every point.

In the [paper](https://arxiv.org/abs/2006.15368) we prove a more general lemma that any objective of the form $ f(s,r,a) \pi(a\mid s) $ is action-unstable unless $ f(s,r,a) > 0 $ at exactly one action. This "unless" provides a potential route to avoiding problems with instability. For example, if we learn a baseline $ b(s) $ so that $ r_i(a_i) - b(s_i) $ is only positive for the optimal action then we would not have any problem. But, learning such a baseline is in general as hard as learning the Q function. There are certain special cases, like if we take a classification problem and turn it into a bandit with 0/1 rewards, where finding such a baseline is easy, and indeed some related work has found this to be quite successful[^joach].

In contrast, the value-based algorithm has an action-stable solution. Namely, if the vector over actions $ \widehat Q(s_i, \cdot)$ is equal to $ r_i $ then the loss is zero no matter which action we choose. This gives us the stable behavior we observe in the experiment.

---

## Some theory

We will defer the full theoretical perspective to the [paper](https://arxiv.org/abs/2006.15368), but we offer a taste here. Specifically, the following two theorems emphasize the difference between value-based and policy-based algorithms again. Since the value-based algorithms are essentially doing regression, we can reduce[^chen][^munos] the policy learning problem to a regression problem where we expect overparameterized models to still perform well [^belkin][^bach][^bartlett]. This is encoded in the following theorem which basically says that the regret of the value-based policy is not too much more than the error in learning the Q function. However, this requires an assumption that the behavior is random enough that we see all actions from all states. 

**Theorem 1 (value-based reduction to regression):** Assume that $ \beta(a\mid s) \geq \tau$ for all $ s,a$. Then,


$$
V(\pi^*) - V(\pi_{\widehat Q}) \leq \frac{2}{\sqrt{\tau}} \sqrt{\mathbb{E}_{s}\mathbb{E}_{a \sim \beta\mid s}[(Q(s,a) - \widehat Q(s,a))^2]}.
$$



But, algorithms like the generic policy-based algorithm that are action-unstable can suffer substantial regret *even* on the states/contexts in the training set and *even* when the behavior is random enough to see all states. In fact, the following theorem shows that action-unstable objectives are *worse* when the behavior is more stochastic, which confirms the idea that they are overfitting to noise in the actions. 

**Theorem 2 (policy-based in-sample regret lower bound):** Define the in-sample value $ V(\pi; S) $ to be 


$$
V(\pi; S) = \frac{1}{N}\sum_{i=1}^N \mathbb{E}_{a\sim \pi\mid s_i}\mathbb{E}_{r\mid s_i}[r(a)].
$$


Take any problem with two actions and let $$ \Delta_r(s) = \mid \mathbb{E}_{r\mid s} [r(1) - r(2)]\mid $$ bet the absolute expected gap in rewards at $s$. Define $p_u(s)$ to be the probability that the policy-based objective is action-unstable at $ x$. Assume that $ \beta(a\mid s) \geq \tau$ for all $ s,a$, and that our policy class $ \Pi $ is overparameterized so that it can interpolate the training objective. Then 


$$
\mathbb{E}_S[V(\pi^*;S) - V(\hat \pi; S)] \geq  \tau \mathbb{E}_{s}[ p_u(s) \Delta_r(s) ].
$$

While we haven't yet proved a more general lower bound on regret for specific model classes like neural nets, we have a more detailed discussion about why we think one may exist in the paper.

---

## Take aways

We took a brief look at what happens when we use overparameterized models in offline contextual bandit algorithms. While one might think that the same generalization properties from supervised learning might carry over, we find this is not the case. Specifically, policy-based algorithms are not action-stable  in general and this makes them very sensitive to overfitting. Value-based algorithms are able to generalize like supervised learning because they are essentially just doing regression. 

Of course there are a few caveats. With small model classes, model misspecification can be a much worse issue for value-based than policy-based algorithms. And without strict positivity, the use of greedy policies in value-based algorithms can lead to problems of extrapolation beyond the support of the data that we have assume away in this work. 

Full details as well as some larger scale experiments on a bandit version of CIFAR can be found in the [paper](https://arxiv.org/abs/2006.15368). And don't hesitate to reach out with comments or questions.

---

**Acknowledgements** 

Thanks to Will Whitney and Evgenii Nikishin for reading drafts of this post. And thanks to my co-authors Will Whitney (again), Rajesh Ranganath, and Joan Bruna.

---

[^zhang]: Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2016). Understanding deep learning requires rethinking generalization. *arXiv preprint arXiv:1611.03530*.
[^belkin]: Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. *Proceedings of the National Academy of Sciences*, *116*(32), 15849-15854.
[^swam2]: Swaminathan, A., & Joachims, T. (2015, June). Counterfactual risk minimization: Learning from logged bandit feedback. In *International Conference on Machine Learning* (pp. 814-823). PMLR.
[^chen]: Chen, J., & Jiang, N. (2019, May). Information-theoretic considerations in batch reinforcement learning. In *International Conference on Machine Learning* (pp. 1042-1051). PMLR.

[^swam]: Swaminathan, A., & Joachims, T. (2015). The self-normalized estimator for counterfactual learning. In *advances in neural information processing systems* (pp. 3231-3239).
[^joach]: Joachims, T., Swaminathan, A., & de Rijke, M. (2018, February). Deep learning with logged bandit feedback. In *International Conference on Learning Representations*.
[^munos]:Munos, R., & Szepesvári, C. (2008). Finite-Time Bounds for Fitted Value Iteration. *Journal of Machine Learning Research*, *9*(5).

[^bach]: Bach, F. (2017). Breaking the curse of dimensionality with convex neural networks. *The Journal of Machine Learning Research*, *18*(1), 629-681.
[^bartlett]: Bartlett, P. L., Long, P. M., Lugosi, G., & Tsigler, A. (2020). Benign overfitting in linear regression. *Proceedings of the National Academy of Sciences*, *117*(48), 30063-30070.

