"""Figures and numbers for the hypothesis-testing README (coin example)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

N = 10
P1, P2 = 0.7, 0.5          # coin 1 (biased), coin 2 (fair)
x = np.arange(N + 1)
pmf1, pmf2 = binom.pmf(x, N, P1), binom.pmf(x, N, P2)

# 1. Likelihood tables
print("x      :", x)
print("coin 1 :", np.round(pmf1, 4))
print("coin 2 :", np.round(pmf2, 4))

# 2. Likelihood ratio and posterior (equal priors)
lr = pmf1 / pmf2                     # P(x|H1)/P(x|H2)
post1 = pmf1 / (pmf1 + pmf2)         # P(H1|x) with priors 1/2
print("LR (coin1/coin2):", np.round(lr, 3))
print("P(coin1|x)      :", np.round(post1, 3))
print("x=3: P2(3)/P1(3) =", pmf2[3] / pmf1[3], " P(coin2|3) =", 1 - post1[3])

# 3. Classical test: H0 = coin 2 (fair), H1 = coin 1 (p=0.7); reject if X >= c
c = 8
alpha = binom.sf(c - 1, N, P2)
power = binom.sf(c - 1, N, P1)
print(f"c={c}: alpha={alpha:.4f} power={power:.4f} beta={1-power:.4f}")

c1, c2 = "#d1495b", "#00798c"

# Figure 1: distributions
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
ax[0].bar(x, pmf2, color=c2); ax[0].set_title("Coin 2: P(heads) = 0.5")
ax[1].bar(x, pmf1, color=c1); ax[1].set_title("Coin 1: P(heads) = 0.7")
for a in ax: a.set_xlabel("Number of heads in 10 tosses"); a.set_xticks(x)
ax[0].set_ylabel("Probability")
plt.tight_layout(); plt.savefig("images/coin_distributions.png", dpi=150); plt.close()

# Figure 2: log likelihood ratio
plt.figure(figsize=(6.5, 3.8))
plt.plot(x, np.log10(lr), "o-", color="k")
plt.axhline(0, color="gray", ls="--")
plt.xticks(x); plt.xlabel("Number of heads x")
plt.ylabel("log10( P(x|coin 1) / P(x|coin 2) )")
plt.title("Evidence for coin 1 (above 0) vs coin 2 (below 0)")
plt.tight_layout(); plt.savefig("images/likelihood_ratio.png", dpi=150); plt.close()

# Figure 3: posterior probability of coin 1 (flat prior)
plt.figure(figsize=(6.5, 3.8))
plt.bar(x, post1, color=c1, label="P(coin 1 | x)")
plt.bar(x, 1 - post1, bottom=post1, color=c2, label="P(coin 2 | x)")
plt.xticks(x); plt.xlabel("Number of heads x"); plt.ylabel("Posterior probability")
plt.legend(loc="upper left", fontsize=8, framealpha=0.9)
plt.title("Bayesian answer: no hard cutoff, just probabilities")
plt.tight_layout(); plt.savefig("images/posterior.png", dpi=150); plt.close()

# Figure 4: decision rule, type I / II error
plt.figure(figsize=(8, 4))
w = 0.38
plt.bar(x - w/2, pmf2, w, color=c2, label="H0 true (fair coin)")
plt.bar(x + w/2, pmf1, w, color=c1, label="H1 true (p = 0.7)")
plt.axvline(c - 0.5, color="k", ls="--")
plt.text(c - 0.45, max(pmf1) * 0.95, f"reject H0\nif x >= {c}", fontsize=9, va="top")
plt.xticks(x); plt.xlabel("Number of heads x"); plt.ylabel("Probability")
plt.title(f"Decision rule: alpha = {alpha:.3f}, power = {power:.3f}")
plt.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("images/decision_rule.png", dpi=150); plt.close()

# Figure 5: posterior updating over repeated tosses (sequence update)
rng = np.random.default_rng(0)
tosses = rng.random(60) < P1            # truth: coin 1
p = 0.5; path = [p]
for t in tosses:
    l1 = P1 if t else 1 - P1
    l2 = P2 if t else 1 - P2
    p = p * l1 / (p * l1 + (1 - p) * l2)
    path.append(p)
plt.figure(figsize=(6.5, 3.8))
plt.plot(path, color=c1); plt.axhline(0.5, color="gray", ls="--")
plt.xlabel("Number of tosses"); plt.ylabel("P(coin 1 | data so far)")
plt.title("Posterior evolves toss by toss (true coin = coin 1)")
plt.tight_layout(); plt.savefig("images/posterior_updating.png", dpi=150); plt.close()