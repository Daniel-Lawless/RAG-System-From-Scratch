## 3. MLE for the Poisson Distribution (Different Sized Intervals)

Suppose we observe the number of events occurring in $n$ intervals, but now the intervals do not all have the same length.

Let $t_i$ denote the length of interval $i$, and let $X_i$ be the number of events observed in that interval.

If events occur at a constant average rate of $\lambda$ per unit time, then the expected number of events in an interval of length $t_i$ is $\lambda t_i$.

So we model each observation as:

```math
\large
X_1 \sim \text{Poisson}(\lambda t_1),
\quad
X_2 \sim \text{Poisson}(\lambda t_2),
\quad
\ldots,
\quad
X_n \sim \text{Poisson}(\lambda t_n)
```

Here,

- $t_i$ = length of interval $i$
- $\lambda$ = average rate of events per unit time
- $\lambda t_i$ = expected number of events in interval $i$

So $\lambda$ is not the expected number of events in one whole interval unless that interval has length 1. Instead, it is the rate per unit time, and multiplying by $t_i$ scales the expected count to match the interval length.

---

### Probability of one observation

For one interval of length $t_i$, the probability of observing $x_i$ events is

```math
\large
p(x_i \mid \lambda, t_i)
=
\frac{(\lambda t_i)^{x_i}e^{-\lambda t_i}}{x_i!}
```

---

### Probability of $n$ observations

Assuming the observations are independent, the joint probability of observing $x_1, x_2, \ldots, x_n$ is

```math
\large
p(x_1,\ldots,x_n \mid \lambda, t_1,\ldots,t_n)
=
\prod_{i=1}^{n}
\frac{(\lambda t_i)^{x_i}e^{-\lambda t_i}}{x_i!}
```

Viewing this as a function of $\lambda$, the likelihood is

```math
\large
L(\lambda \mid x_1,\ldots,x_n,t_1,\ldots,t_n)
=
\prod_{i=1}^{n}
\frac{(\lambda t_i)^{x_i}e^{-\lambda t_i}}{x_i!}
```

---

### Taking logs

To make differentiation easier, we take the log-likelihood.

Since the logarithm is strictly increasing, the value of $\lambda$ that maximises the likelihood will also maximise the log-likelihood.

Let

```math
\large
\mathcal{J}(\lambda)
=
\ln\left(
\prod_{i=1}^{n}
\frac{(\lambda t_i)^{x_i}e^{-\lambda t_i}}{x_i!}
\right)
```

Using the rule that the log of a product becomes a sum of logs, we get:

```math
\large
\mathcal{J}(\lambda)
=
\sum_{i=1}^{n}
\ln\left(
\frac{(\lambda t_i)^{x_i}e^{-\lambda t_i}}{x_i!}
\right)
```

Now expand the logarithm of the fraction:

```math
\large
\mathcal{J}(\lambda)
=
\sum_{i=1}^{n}
\left[
\ln\big((\lambda t_i)^{x_i}\big)
+
\ln(e^{-\lambda t_i})
-
\ln(x_i!)
\right]
```

Now simplify each log term:

- $\ln\big((\lambda t_i)^{x_i}\big) = x_i\ln(\lambda t_i)$
- $\ln(e^{-\lambda t_i}) = -\lambda t_i$

So this becomes:

```math
\large
\mathcal{J}(\lambda)
=
\sum_{i=1}^{n}
\left[
x_i\ln(\lambda t_i)
-
\lambda t_i
-
\ln(x_i!)
\right]
```

Next, split $\ln(\lambda t_i)$ using $\ln(ab)=\ln(a)+\ln(b)$:

```math
\large
\mathcal{J}(\lambda)
=
\sum_{i=1}^{n}
\left[
x_i\ln(\lambda)
+
x_i\ln(t_i)
-
\lambda t_i
-
\ln(x_i!)
\right]
```

Now separate the sum into individual parts:

```math
\large
\mathcal{J}(\lambda)
=
\left(
\sum_{i=1}^{n} x_i
\right)\ln(\lambda)
+
\sum_{i=1}^{n} x_i\ln(t_i)
-
\lambda\sum_{i=1}^{n} t_i
-
\sum_{i=1}^{n}\ln(x_i!)
```

---

### Finding the optimal value of $\lambda$

We now differentiate $\mathcal{J}(\lambda)$ with respect to $\lambda$.

Only the terms involving $\lambda$ matter. The other two sums are constants with respect to $\lambda$.

So:

```math
\large
\frac{d\mathcal{J}(\lambda)}{d\lambda}
=
\frac{\sum_{i=1}^{n}x_i}{\lambda}
-
\sum_{i=1}^{n}t_i
```

To maximise the log-likelihood, we set this equal to zero:

```math
\large
0
=
\frac{\sum_{i=1}^{n}x_i}{\lambda}
-
\sum_{i=1}^{n}t_i
```

Now rearrange to solve for $\lambda$.

First move the second term to the other side:

```math
\large
\sum_{i=1}^{n}t_i
=
\frac{\sum_{i=1}^{n}x_i}{\lambda}
```

Now multiply both sides by $\lambda$:

```math
\large
\lambda\sum_{i=1}^{n}t_i
=
\sum_{i=1}^{n}x_i
```

Finally divide by $\sum_{i=1}^{n}t_i$:

```math
\large
\hat{\lambda}
=
\frac{\sum_{i=1}^{n}x_i}{\sum_{i=1}^{n}t_i}
```

---

### Interpretation

So the maximum likelihood estimator for $\lambda$ is

```math
\large
\hat{\lambda}
=
\frac{\sum_{i=1}^{n}x_i}{\sum_{i=1}^{n}t_i}
```

This is the total number of observed events divided by the total observed time.

So the most likely average rate per unit time is

```math
\large
\hat{\lambda}
=
\frac{\text{Total Observed Events}}{\text{Total Observed Time}}
```

For example, if we observe 12 events over a total of 6 seconds, then

```math
\large
\hat{\lambda}
=
\frac{12}{6}
=
2
```

so the estimated rate is 2 events per second.

Important distinction:

- Equal intervals: mean count per interval
- Unequal intervals: mean rate per unit time

---

### Special case: equal interval lengths

If all intervals have length 1, so that

```math
\large
t_1 = t_2 = \cdots = t_n = 1
```

then the estimator becomes

```math
\large
\hat{\lambda}
=
\frac{\sum_{i=1}^{n}x_i}{\sum_{i=1}^{n}1}
=
\frac{\sum_{i=1}^{n}x_i}{n}
```

which is exactly the sample mean from the equal-interval Poisson case.

So the equal-sized interval model is just a special case of the more general different-sized interval model.
