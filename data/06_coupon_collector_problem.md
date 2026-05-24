## 6. Coupon Collector Problem

Suppose there are $n$ different coupon types. Each time we buy a coupon, we receive one coupon chosen uniformly at random from the $n$ possible types. The coupon collector problem asks what is the expected number of draws needed to collect all $n$ coupons.

Let $T$ = the total number of draws needed to collect all $n$ coupon types. We are required to find $\mathbb{E}[T]$.

This question is solved easiest by splitting the total number of draws into stages.

```math
\large
T = X_0 + X_1 + \cdots + X_{n-1},
```

where $X_k$ = the number of draws needed to go from $k$ collected coupons to $k+1$ collected coupons.

In the first stage, $X_0$ tells us how many draws are needed to get a new coupon when we have not collected any coupons yet. Since there are $n$ total possibilities and $n$ successes the probability of success is:

```math
\large
p_0 = \frac{n}{n} = 1
```

In the second stage, $X_1$, when we have picked one coupon, the total number of unobserved coupons is $n - 1$ and the total number of possibilites is $n$. Therefore, the probability of getting a coupon we have not observed in the second stage is:

```math
\large
p_1 = \frac{n - 1}{n}
```

```math
\large
\vdots
```

In the $k$-th stage, when we have picked $k$ coupons, the total number of unobserved coupons is $n-k$ and the total number of possibilities is $n$. Therefore the probability of getting a coupon we have not observed in the $k$-th stage is:

```math
\large
p_k = \frac{n-k}{n}
```

We can observe that as we collect more distinct coupon types, the probability of drawing a coupon we have drawing a coupon we have not yet seen before decreases. This is because the number of unseen coupon types becomes smaller at each stage. For example, once we have collected $n-1$ distinct coupon types, only one coupon type remains $n-1$ unseen, so the probability of drawing the final missing coupon is very small

The total number of draws needed to collect all $n$ coupon types is given by

```math
\large
T = X_0 + X_1 + X_2 + \cdots + X_{n-1}
```

## Expected number of draws

The total number of draws needed to collect all $n$ coupon types is:

```math
\large
T = X_0 + X_1 + X_2 + \cdots + X_{n-1}
```

Taking expectations:

```math
\large
\mathbb{E}[T]
=
\mathbb{E}[X_0 + X_1 + X_2 + \cdots + X_{n-1}]
```

Using linearity of expectation:

```math
\large
\mathbb{E}[T]
=
\mathbb{E}[X_0]
+
\mathbb{E}[X_1]
+
\mathbb{E}[X_2]
+
\cdots
+
\mathbb{E}[X_{n-1}]
```

Each $X_k$ is a geometric random variable, where the success probability is:

```math
\large
p_k = \frac{n-k}{n}
```

For a geometric random variable, the expected waiting time until the first success is:

```math
\large
\mathbb{E}[X_k] = \frac{1}{p_k}
```

Therefore:

```math
\large
\mathbb{E}[T]
=
\frac{1}{p_0}
+
\frac{1}{p_1}
+
\frac{1}{p_2}
+
\cdots
+
\frac{1}{p_{n-1}}
```

Substituting in the probabilities:

```math
\large
\mathbb{E}[T]
=
\frac{1}{\frac{n}{n}}
+
\frac{1}{\frac{n-1}{n}}
+
\frac{1}{\frac{n-2}{n}}
+
\cdots
+
\frac{1}{\frac{1}{n}}
```

Simplifying:

```math
\large
\mathbb{E}[T]
=
1
+
\frac{n}{n-1}
+
\frac{n}{n-2}
+
\cdots
+
n
```

Factoring out $n$:

```math
\large
\mathbb{E}[T]
=
n
\left(
\frac{1}{n}
+
\frac{1}{n-1}
+
\frac{1}{n-2}
+
\cdots
+
1
\right)
```

Reordering the terms:

```math
\large
\boxed{
\mathbb{E}[T]
=
n
\left(
1
+
\frac{1}{2}
+
\frac{1}{3}
+
\cdots
+
\frac{1}{n}
\right)
}
```

The harmonic series is given by:

```math
\large
H_n
=
1
+
\frac{1}{2}
+
\frac{1}{3}
+
\cdots
+
\frac{1}{n}
```

So we can write the final result as:

```math
\large
E[T] = nH_{n}
```


---

## Complementary notes: Geometric random variables

Since each $X_i$ is asking " what is the number of trials to get a success" This is a classic geometric random variable. Properties of a geometric random variable are:

$X_i$ : # of trials to get a success:

- Independent trials
- Probability of success is fixed
- Success or failure for trial outcome
- Not fixed number of trials

The PMF for a geometric random variable is:

```math
\large
P(X = x)
=
(1-p)^{x-1}p
```

Where $p$ is the probability of success. This makes sense intuitively since for a success to happen after $x$ trials we need $x-1$ failures, then a success.

The expected value of a geometric random variable is:

```math
\large
\mathbb{E}[X]
=
\frac{1}{p}
```

### Proof:

Let $X$ be a geometric random variable. Then, its possible values are $1,2,3 \ldots$ Its minimum value is 1, and maximum value has no upper bound.

Then:

```math
\large
\begin{aligned}
\mathbb{E}[X]
&=
(1)P(X=1)
+
(2)P(X=2)
+
(3)P(X=3)
+
\cdots \\
&=
p
+
2p(1-p)
+
3p(1-p)^2
+
\cdots
\end{aligned}
```

Multiply both sides by $(1-p)$:

```math
\large
\begin{aligned}
(1-p)\mathbb{E}[X]
&=
p(1-p)
+
2p(1-p)^2
+
3p(1-p)^3
+
\cdots
\end{aligned}
```

Subtract this from $\mathbb{E}[X]$:

```math
\large
\begin{aligned}
\mathbb{E}[X]
-
(1-p)\mathbb{E}[X]
&=
p
+
p(1-p)
+
p(1-p)^2
+
\cdots \\
p\mathbb{E}[X]
&=
p
+
p(1-p)
+
p(1-p)^2
+
\cdots
\end{aligned}
```

Then, solve for $\mathbb{E}[X]$ by dividing by $p$:

```math
\large
\begin{aligned}
\mathbb{E}[X]
&=
1
+
(1-p)
+
(1-p)^2
+
\cdots
\end{aligned}
```

Now, this is in the form of a geometric series. The common ratio $r$ is $(1-p)$ and the first value, $a$ is 1. For $|r| < 1$ a geometric series converges. Therefore we get:

```math
\large
\begin{aligned}
\mathbb{E}[X]
&=
\frac{a}{1-r} \\
&=
\frac{1}{1-(1-p)} \\
&=
\frac{1}{p}
\end{aligned}
```

---

## Learning outcomes from the Coupon Collector Question:

- We do not need to calculate the full distribution of $T$. Instead we can split the total collection time into smaller waiting times.

- Geometric random variables model the number of trials needed until the first success. It assumes independent trials, fixed probability, and binary outcomes, but not a fixed number of trials.

- The waiting time gets larger as more coupon types are collected, because the probability of finding a new coupon decreases.

## Main applications of the Coupon Collector Problem in AI:

### Dataset Collection:

Suppose we want to train an image classifier with $n$ classes:

```text
Dog, Cat, Horse, Sheep, Cow, Fox Rabbit, Deer
```

To train a model that generalizes well to all classes, we need samples spanning all classes. If data arrives randomly, then from the Coupon Collector Problem we can ask how many samples can we expect to collect before every class appears atleast once. In this context, we have this mapping:

<div align="center">

<table>
  <tr>
    <td>Coupon type</td>
    <td>↔</td>
    <td>Class label</td>
  </tr>
  <tr>
    <td>Buying a coupon</td>
    <td>↔</td>
    <td>Collecting one data sample</td>
  </tr>
  <tr>
    <td>Collecting all coupons</td>
    <td>↔</td>
    <td>Seeing every class in the dataset</td>
  </tr>
</table>

</div>

The Coupon Collection Problem tells us:

```math
\large
\mathbb{E}[T]
=
nH_n
```

So even with uniformly likely classes, the expected number of samples needed to see every class is more than $n$. You do not expect to see all classes after only $n$ samples, since duplicates are likely.

For example, if there are 10 equally likely classes:

```math
\large
\mathbb{E}[T]
=
10
\left(
1
+
\frac{1}{2}
+
\cdots
+
\frac{1}{10}
\right)
\approx
29.29
```

So even though there are only 10 classes, you expect to need about 29 random samples just to see every class once. So the AI lesson is random data collection may not give good coverage of all classes. This is why people often use targeted data collection, where they deliberately search for underrepresented classes.

## Class imbalance:

If a rare class appears with probability $p = 0.005$, then the expected number of samples before seeing that rare class once is:

```math
\large
\frac{1}{p}
=
\frac{1}{0.005}
=
200
```

So you would expect to need about 200 random samples just to see that rare class once. This matters because machine learning models learn mostly from what they see often. If the dataset has loads of examples from the common class but very few examples from the rare class, the model may perform badly on the rare class.

So the AI lesson is rare classes require deliberate sampling because random sampling may barely include them. reducing generalization and performance.
