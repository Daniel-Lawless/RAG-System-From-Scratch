## 5. The German Tank Problem

This problem is about estimating an unknown total from a small sample.

During World War II, the Allies wanted to estimate how many tanks Germany was producing. Captured tanks had serial numbers on them, so if they observed serial numbers such as 12, 37, 82, 103, then they immediately knew that at least 103 tanks had been produced.

But the real question was: how many tanks had been produced in total?  
The German Tank Problem uses the observed serial numbers to estimate this unknown total.

Suppose Germany produces tanks numbered:

```math
\large
1, 2, 3, \ldots, N
```

where $N$ is unknown. Suppose we capture $n$ tanks and record their serial numbers:

```math
\large
x_1, x_2, x_3, \ldots, x_n
```

The goal is to estimate the total number of tanks $N$ using these $n$ observations.

The classical German Tank Problem assumes that the observed tanks are a random sample without replacement from $\{1, \ldots, N\}$. Therefore every unordered sample of size $n$ is equally likely. So the total number of possible samples is:

```math
\large
\binom{N}{n}
```

Let $X_i$ be the serial number of the $i$th observed tank. Then marginally,

```math
\large
X_i \sim \text{DiscreteUniform}\{1,2,\ldots,N\}
```

so for any $k \in \{1,2,\ldots,N\}$

```math
\large
P(X_i = k)
=
\frac{1}{N}
```

However, the collection $X_1, \ldots, X_n$ is not independent, because we sample without replacement.

Since every unordered sample of size $n$ is equally likely, the probability of observing any particular unordered sample is:

```math
\large
P(\{X_1,\ldots,X_n\} = \{x_1,\ldots,x_n\})
=
\frac{1}{\binom{N}{n}}
```

---

## Distribution of the sample maximum

Define the maximum observed serial number by:

```math
\large
M
=
\max(X_1,\ldots,X_n)
```

and after observing the sample, let:

```math
\large
m
=
\max(x_1,\ldots,x_n)
```

We want to find $P(M=m \mid N)$.

For the maximum to be exactly $m$, two things must happen: the sample must contain $m$ and the remaining $n-1$ observed serial numbers must all come from $\{1,2,\ldots,m-1\}$.

The number of samples where $m$ will be the largest value in each sample will be given by:

```math
\large
\binom{m-1}{n-1}
```

This will give all samples of size $n-1$ that have values $\leq m-1$. Then, if $m$ were in these samples, making them of size $n$, it would be the largest value. Therefore the probability of observing the maximum observed serial number is given by:

```math
\large
P(M=m \mid N)
=
\frac{
\binom{m-1}{n-1}
}{
\binom{N}{n}
},
\qquad
n \leq m \leq N
```

---

## Likelihood and the maximum likelihood estimator

Once we observe the data, $m$ is fixed, since we can inspect the sample and find its maximum. So $N$ becomes the unknown quantity, and we view the previous expression as a likelihood:

```math
\large
L(N \mid M=m)
=
\frac{
\binom{m-1}{n-1}
}{
\binom{N}{n}
},
\qquad
N \geq m
```

The numerator is fixed because $m$ and $n$ are known. The denominator $\binom{N}{n}$ increases as $N$ increases, so the likelihood decreases as $N$ increases.

Hence, the likelihood is maximised when $N$ is as small as possible, namely $N=m$.

Therefore, the maximum likelihood estimator is:

```math
\large
\hat{N}_{MLE}
=
M
```

This is clearly biased downward, because the largest observed serial number is unlikely to be the true maximum $N$ unless we happened to observe the final tank produced.

---

## Deriving the unbiased estimator

From the previous result, for a random sample of size $n$ from $\{1,\ldots,N\}$, the expected maximum is:

```math
\large
\mathbb{E}[M]
=
\frac{n(N+1)}{n+1}
```

We will use this result to derive the unbiased estimator. A proof of this expected maximum formula is given later in the section **Deriving the expected maximum using gaps**.

Now rearrange this equation to solve for $N$.

First, multiply both sides by $n+1$:

```math
\large
(n+1)\mathbb{E}[M]
=
n(N+1)
```

Now divide both sides by $n$:

```math
\large
\frac{(n+1)\mathbb{E}[M]}{n}
=
N+1
```

Finally, subtract $1$ from both sides to isolate $N$:

```math
\large
N
=
\frac{(n+1)\mathbb{E}[M]}{n}
-
1
```

Before observing the sample, $M$ is a random variable. Once we observe the sample, we get one realised value $M=m$.

To build an estimator, we replace the theoretical expectation $\mathbb{E}[M]$ with the observed maximum $m$.

```math
\large
\hat{N}
=
\frac{m(n+1)}{n}
-
1
```

Or equivalently:

```math
\large
\hat{N}
=
m
+
\frac{m}{n}
-
1
```

So the German Tank estimator takes the largest observed serial
number and adds an estimated missing gap above it. This improves on the naive estimate by accounting for
the fact that the observed maximum is probably below the true maximum $N$.

---

## Deriving the expected maximum using gaps

Suppose the true serial numbers are $\{1,\ldots,N\}$, and suppose we observe $n$ tanks and sort their serial numbers:

```math
\large
X_{(1)} < X_{(2)} < \cdots < X_{(n)}
```

The maximum observed serial number is therefore:

```math
\large
M = X_{(n)}
```

These $n$ observed values split the full range of true serial numbers into $n+1$ gaps.

### Example

Suppose $N=10$ and we observe 1, 2, 4, 9. Then the gaps are:

The diagram below places the observed serial numbers $1, 2, 4,$ and $9$ inside the full range from $1$ to $10$. The blank spaces between and around the observed values are the gaps. The final gap is the number of unobserved serial numbers above the maximum observed value.

<p align="center">
  <img src="https://raw.githubusercontent.com/Daniel-Lawless/Statistics-for-ai/main/assets/german-tank-problem-images/german-tank-gap-example.png" alt="German Tank gap example" width="250">
</p>

So there are $n+1 = 4+1 = 5$ gaps $G_0, G_1, G_2, G_3, G_4$.

In general, the $n+1$ gap idea counts the possible slots where unobserved numbers can sit.

The diagram below shows the general structure. The $n$ observed serial numbers split the full range $\{1,\ldots,N\}$ into $n+1$ gaps: one before the first observed value, one between each neighbouring pair of observed values, and one after the maximum observed value. The final gap $G_n$ is especially important because it measures how far the observed maximum $M$ is below the true maximum $N$.

<p align="center">
  <img src="https://raw.githubusercontent.com/Daniel-Lawless/Statistics-for-ai/main/assets/german-tank-problem-images/german-tank-gaps.png" alt="German Tank gaps" width="350">
</p>

The final gap $G_n$ is the number of unobserved serial numbers above the maximum. In the example, $M=9$, and there are $10-9=1$ numbers above it. So:

```math
\large
G_n
=
N-M
```

Rearranging gives:

```math
\large
M
=
N-G_n
```

---

## How big is the expected final gap?

There are $N$ total serial numbers, and we have observed $n$ of them, so there are $N-n$ unobserved serial numbers.

These unobserved serial numbers are split across the $n+1$ gaps, so we know that:

```math
\large
G_0 + G_1 + \cdots + G_n
=
N-n
```

Now, before seeing the sample, no gap position is special. The gap before the first observation is not expected to be bigger than a gap between two observed values, and the final gap after the maximum is not expected to be bigger or smaller either. Therefore, each gap has the same expected size:

```math
\large
\mathbb{E}[G_0]
=
\mathbb{E}[G_1]
=
\cdots
=
\mathbb{E}[G_n]
```

Taking expectations:

```math
\large
\mathbb{E}[G_0 + G_1 + \cdots + G_n]
=
\mathbb{E}[N-n]
```

Using linearity of expectation:

```math
\large
\mathbb{E}[G_0]
+
\mathbb{E}[G_1]
+
\cdots
+
\mathbb{E}[G_n]
=
N-n
```

Let this common expected gap size be $c$. Then:

```math
\large
c(n+1)
=
N-n
```

So:

```math
\large
c
=
\frac{N-n}{n+1}
```

So for any gap $G_i$, including the final gap $G_n$:

```math
\large
\mathbb{E}[G_i]
=
\frac{N-n}{n+1}
```

Since $M = N - G_n$, taking expectations gives:

```math
\large
\mathbb{E}[M]
=
\mathbb{E}[N - G_n]
```

Using linearity of expectation:

```math
\large
\mathbb{E}[M]
=
\mathbb{E}[N]
-
\mathbb{E}[G_n]
```

Since $N$ is fixed, $\mathbb{E}[N] = N$, so:

```math
\large
\mathbb{E}[M]
=
N
-
\mathbb{E}[G_n]
```

From the gap argument, the expected final gap is:

```math
\large
\mathbb{E}[G_n]
=
\frac{N-n}{n+1}
```

Substitute this in:

```math
\large
\mathbb{E}[M]
=
N
-
\frac{N-n}{n+1}
```

Now write $N$ with denominator $n+1$:

```math
\large
\mathbb{E}[M]
=
\frac{N(n+1)}{n+1}
-
\frac{N-n}{n+1}
```

Combine the fractions:

```math
\large
\mathbb{E}[M]
=
\frac{N(n+1) - (N-n)}{n+1}
```

Expand and simplify the numerator:

```math
\large
\mathbb{E}[M]
=
\frac{Nn + N - N + n}{n+1}
```

```math
\large
\mathbb{E}[M]
=
\frac{nN+n}{n+1}
```

Factor out $n$:

```math
\large
\mathbb{E}[M]
=
\frac{n(N+1)}{n+1}
```
---

The table below compares the unbiased German Tank estimator with the maximum likelihood estimator. The left-hand column proves that

```math
\large
\hat{N}
=
M + \frac{M}{n} - 1
```

has zero bias. The right-hand column proves that the MLE

```math
\large
\hat{N}_{MLE}
=
M
```

is biased downward because the observed maximum $M$ is usually smaller than the true maximum $N$.

<div align="center">

| Proving the German Tank estimator is unbiased | Proving the MLE estimator is biased |
|---|---|
| $\large \displaystyle \text{bias}(\hat{N}) = \mathbb{E}[\hat{N}] - N$ | <img src="https://raw.githubusercontent.com/Daniel-Lawless/Statistics-for-ai/main/assets/german-tank-problem-images/german-tank-MLE-estimator-bias.png" alt="German Tank gaps" width="250"> |
| $\large \displaystyle = \mathbb{E}\left[M + \frac{M}{n} - 1\right] - N$ | $\large \displaystyle = \mathbb{E}[M] - N$ |
| $\large \displaystyle = \mathbb{E}[M] + \frac{\mathbb{E}[M]}{n} - 1 - N$ | $\large \displaystyle = \frac{n(N+1)}{n+1} - N$ |
| $\large \displaystyle = \frac{n(N+1)}{n+1} + \frac{1}{n}\frac{n(N+1)}{n+1} - 1 - N$ | $\large \displaystyle = \frac{nN+n-Nn-N}{n+1}$ |
| $\large \displaystyle = \frac{n(N+1)}{n+1} + \frac{N+1}{n+1} - 1 - N$ | $\large \displaystyle = \frac{n-N}{n+1}$ |
| $\large \displaystyle = \frac{(n+1)(N+1)}{n+1} - 1 - N$ | Since $n \geq 0$ and in the non-trivial case $n < N$, the numerator is negative and the denominator is positive. Thus, the bias is negative. |
| $\large \displaystyle = N+1-1-N$ | $\large \displaystyle \text{bias}(\hat{N}_{\mathrm{MLE}}) < 0$ |
| $\large \displaystyle = 0$ | Therefore, this estimator undershoots the true value of $N$ on average. |
| Since the bias is 0, this estimator estimates the correct value of $N$ on average. Therefore, $\hat{N}$ is unbiased. |  |

</div>
