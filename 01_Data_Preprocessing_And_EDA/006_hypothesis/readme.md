# Hypothesis Testing

* What a hypothesis is, and how a test turns data into a decision
* The classical (frequentist) approach: decision rules, error types, power
* The Bayesian approach: likelihood ratios and updating priors
* A worked example with coin tossing (code and figures included)

## 1. What is a hypothesis?

A **hypothesis** is a statement about a population parameter, for example "this coin lands heads 70% of the time" or "the new model's mean error is lower than the old one's."

We usually set up two competing hypotheses:
* The **null hypothesis**, $H_0$: the default or "nothing special" claim.
* The **alternative hypothesis**, $H_1$ (or $H_A$): the claim we would act on if the data disagree with $H_0$.

Which one is called the null depends on how the problem is set up. A common convention is to make $H_0$ the claim whose wrongful rejection would be the more costly mistake.

A test statistic is a number computed from the data (here: the count of heads) that we use to choose between $H_0$ and $H_1$.

----------

## 2. Decision rules (classical approach)

A **hypothesis testing procedure** is a rule that says:

* for which values of the test statistic we **fail to reject $H_0$**, and
* for which values (the **rejection region**) we **reject $H_0$** and go with **$H_1$**.

You will hear the advice that you can reject $H_0$ but never accept $H_0$ (absence of evidence is not proof). That is a careful statement about what a single test proves. In applied work the practical question is usually simply: which of two paths do we take in this project? The test gives a principled, repeatable way to choose.


### Two ways to be wrong

|                     | $H_0$ true                     | $H_1$ true                      |
|---------------------|--------------------------------|---------------------------------|
| **Reject $H_0$**    | Type I error (prob. $\alpha$)  | Correct (prob. $1-\beta$ = power) |
| **Don't reject**    | Correct                        | Type II error (prob. $\beta$)   |

- $\alpha$ is the **significance level**: the probability of rejecting $H_0$ when it is true.
- **Power** $=1-\beta$: the probability of rejecting $H_0$ when $H_1$ is true.
- Moving the decision boundary trades one error for the other.
- A **p-value** is the probability, assuming $H_0$ is true, of seeing a test statistic at least as extreme as the one observed. It is *not* the probability that $H_0$ is true.

---
 
## 3. The Bayesian approach
 
In the Bayesian interpretation there is **no decision boundary** built into the method. Instead we:
 
1. Put a **prior** probability on each hypothesis, $P(H_1)$ and $P(H_2)$.
2. Compute how likely the data are under each hypothesis (the **likelihood**).
3. Combine them with Bayes' rule to get **posterior** probabilities.
The output is a probability for each hypothesis. If a yes/no action is needed, you pick a threshold afterwards based on the costs of being wrong.
 
---
 
## 4. Example: which coin did I toss?
 
- **Coin 1** has $P(\text{heads}) = 0.7$.
- **Coin 2** has $P(\text{heads}) = 0.5$ (fair).
Pick one coin without looking, toss it 10 times, and record the number of heads $x$.
**Given $x$, which coin did you toss?**
 
The number of heads is Binomial$(10, p)$:
 
$$P(x \mid p) = \binom{10}{x} p^x (1-p)^{10-x}$$
 
```python
import numpy as np
from scipy.stats import binom
 
N, P1, P2 = 10, 0.7, 0.5
x = np.arange(N + 1)
pmf1 = binom.pmf(x, N, P1)   # coin 1
pmf2 = binom.pmf(x, N, P2)   # coin 2
```
 
![Distribution of heads for each coin](../images/coin_distributions.png)
 
Probability of each head count:
 
| x heads | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Coin 1 (p=0.7) | 0.000 | 0.000 | 0.001 | 0.009 | 0.037 | 0.103 | 0.200 | 0.267 | 0.234 | 0.121 | 0.028 |
| Coin 2 (p=0.5) | 0.001 | 0.010 | 0.044 | 0.117 | 0.205 | 0.246 | 0.205 | 0.117 | 0.044 | 0.010 | 0.001 |
 
---
 
## 5. The likelihood ratio
 
The **likelihood ratio** compares how well each hypothesis explains the observed data:
 
$$\Lambda(x) = \frac{P(x \mid H_1)}{P(x \mid H_2)}$$
 
Suppose we saw **three heads**:
 
- $P(3 \mid \text{coin 1}) = 0.009$ and $P(3 \mid \text{coin 2}) = 0.117$
- $\dfrac{P_2(3)}{P_1(3)} = \dfrac{0.117}{0.009} \approx 13$
So coin 2 (the fair coin) was about **13 times more likely** than coin 1 to produce 3 heads.
 
```python
lr = pmf1 / pmf2             # likelihood ratio, coin 1 over coin 2
print(pmf2[3] / pmf1[3])     # about 13.0
```
 
![Log likelihood ratio](../images/likelihood_ratio.png)
 
The ratio equals 1 (log = 0) near $x = 6$. Fewer heads favour the fair coin, more heads favour the biased coin.
 
---
 
## 6. Bayesian interpretation: updating priors
 
We chose the coin at random, so before seeing data:
 
$$P(H_1) = P(H_2) = \tfrac{1}{2}$$
 
After seeing data $x$, Bayes' rule gives the **posterior**:
 
$$P(H_1 \mid x) = \frac{P(x \mid H_1)\,P(H_1)}{P(x)}, \qquad P(x) = \sum_i P(x\mid H_i)P(H_i)$$
 
and in ratio form:
 
$$\underbrace{\frac{P(H_1 \mid x)}{P(H_2 \mid x)}}_{\text{posterior odds}} = \underbrace{\frac{P(H_1)}{P(H_2)}}_{\text{prior odds}} \times \underbrace{\frac{P(x \mid H_1)}{P(x \mid H_2)}}_{\text{likelihood ratio}}$$
 
The likelihood ratio does **not** depend on the priors. It tells us how much to update whatever prior we started with.
 
For 3 heads with equal priors, the odds are 1 : 1 times 1 : 13, so
 
$$P(\text{coin 2} \mid x=3) = \frac{0.117}{0.117 + 0.009} \approx 0.93$$
 
```python
prior1 = prior2 = 0.5
post1 = pmf1 * prior1 / (pmf1 * prior1 + pmf2 * prior2)
print(1 - post1[3])          # about 0.93 -> probability it was coin 2
```
 
![Posterior probability for each head count](../images/posterior.png)
 
A different prior (say, you were 90% sure you grabbed coin 1) would shift these posteriors, but the likelihood ratio stays the same.
 
### Updating one toss at a time
 
Bayes' rule can be applied repeatedly: today's posterior becomes tomorrow's prior.
 
```python
p = 0.5                                   # prior P(coin 1)
for toss in tosses:                       # True = heads
    l1 = P1 if toss else 1 - P1
    l2 = P2 if toss else 1 - P2
    p = p * l1 / (p * l1 + (1 - p) * l2)  # posterior becomes new prior
```
 
![Posterior evolving over 60 tosses](../images/posterior_updating.png)
 
---
 
## 7. The classical version of the same problem
 
Take $H_0$: the coin is fair (coin 2) and $H_1$: $p = 0.7$ (coin 1). Use the decision rule
 
> **Reject $H_0$ if the number of heads is at least 8.**
 
```python
c = 8
alpha = binom.sf(c - 1, N, P2)   # P(X >= 8 | fair)   -> 0.0547
power = binom.sf(c - 1, N, P1)   # P(X >= 8 | p=0.7)  -> 0.3828
beta  = 1 - power                #                     -> 0.6172
```
 
![Decision rule with alpha and power](../images/decision_rule.png)
 
- $\alpha \approx 0.055$: about a 5.5% chance of wrongly rejecting a fair coin.
- Power $\approx 0.38$: only a 38% chance of catching the biased coin with 10 tosses.
Lowering the cutoff raises power but also raises $\alpha$. The cure for weak power is more data: more tosses separate the two distributions.
 
---

live check:

[hypothesis testing two coins](hypothesis_testing_two_coins.html)

---
 
## 8. Side by side
 
| | Classical | Bayesian |
|---|---|---|
| Output | Reject / don't reject $H_0$ | Probability of each hypothesis |
| Needs priors? | No | Yes |
| Key quantity | Test statistic, $\alpha$, p-value, power | Likelihood ratio, posterior |
| Decision boundary | Set in advance (via $\alpha$) | Chosen afterwards from costs, if needed |
| Shared ingredient | The likelihood $P(\text{data}\mid H)$ | The likelihood $P(\text{data}\mid H)$ |
 
Both approaches are built on the same likelihoods. They differ in what they do with them.
 
## Takeaways
 
- A hypothesis is a claim about a parameter; a test is a rule mapping data to a decision.
- Every decision rule trades off Type I and Type II errors.
- The likelihood ratio measures how strongly the data favour one hypothesis over another.
- Bayes' rule says: posterior odds = prior odds × likelihood ratio.

------------------------------------------

| <img src="https://github.com/madhawapolkotuwa.png" width="80" style="border-radius: 50%;"> | **MPCoding**<br>*Machine Learning Tutorial Series*<br><br> [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/@MP-Coding) [![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/madhawapolkotuwa) |
|---|---|

<p align="left" style="font-size: 0.85em; color: gray; margin-top: 10px;">
  © 2026 MPCoding. Educational materials are free to use with proper attribution.
</p>
