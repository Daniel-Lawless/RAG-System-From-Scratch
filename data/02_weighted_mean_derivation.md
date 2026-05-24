## 2. Weighted Mean Derivation:

Suppose we want to estimate an unknown true value $\mu$ from $n$ independent observations. Each observation is assumed to be normally distributed around
$\mu$, but with its own variance:

```math
\large
X_1 \sim N(\mu, \sigma_1^2),
\quad
X_2 \sim N(\mu, \sigma_2^2),
\quad
\cdots
\quad
X_n \sim N(\mu, \sigma_n^2)
```

We then observe the values $x_1, x_2, \ldots, x_n$. The variances $\sigma_1^2, \ldots, \sigma_n^2$ represent the uncertainty in each observation. Smaller variance means a more reliable observation, so we expect such observations to contribute more heavily to our estimate of $\mu$.

## Likelihood of one observation:

For a single observation $X_i \sim N(\mu, \sigma_i^2)$, the density is

```math
\large
p(x_i \mid \mu, \sigma_i^2)
=
\frac{1}{\sigma_i\sqrt{2\pi}}
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma_i^2}
\right)
```

## Likelihood of n observations:

Assuming the observations are independent, the joint likelihood is the product of the individual densities:

```math
\large
p(x_1, \ldots, x_n \mid \mu, \sigma_1^2, \ldots, \sigma_n^2)
=
\prod_{i=1}^{n}
\frac{1}{\sigma_i\sqrt{2\pi}}
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma_i^2}
\right)
```

Viewing this as a function of $\mu$, the likelihood is

```math
\large
L(\mu \mid x_1, \ldots, x_n, \sigma_1^2, \ldots, \sigma_n^2)
=
\prod_{i=1}^{n}
\frac{1}{\sigma_i\sqrt{2\pi}}
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma_i^2}
\right)
```

## Taking logs:

To make differentiation easier, we take the log-likelihood. Since the logarithm is strictly increasing, the value of $\mu$ that maximises the likelihood also maximises the log-likelihood.

Let

```math
\large
\begin{aligned}
J(\mu)
&=
\ln\left(
\prod_{i=1}^{n}
\frac{1}{\sigma_i\sqrt{2\pi}}
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma_i^2}
\right)
\right)
\\
&=
\sum_{i=1}^{n}
\ln\left(
\frac{1}{\sigma_i\sqrt{2\pi}}
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma_i^2}
\right)
\right)
\\
&=
\sum_{i=1}^{n}
\left[
\ln\left(
\frac{1}{\sigma_i\sqrt{2\pi}}
\right)
-
\frac{(x_i - \mu)^2}{2\sigma_i^2}
\right]
\\
&=
\sum_{i=1}^{n}
\ln\left(
\frac{1}{\sigma_i\sqrt{2\pi}}
\right)
-
\sum_{i=1}^{n}
\frac{(x_i - \mu)^2}{2\sigma_i^2}
\end{aligned}
```

## Finding the optimal value of $\mu$:

We now differentiate $J(\mu)$ with respect to $\mu$:

```math
\large
\frac{dJ(\mu)}{d\mu}
=
\sum_{i=1}^{n}
\frac{x_i - \mu}{\sigma_i^2}
```

To maximise the log-likelihood, we set this equal to zero:

```math
\large
0
=
\sum_{i=1}^{n}
\frac{x_i - \mu}{\sigma_i^2}
```

```math
\large
0
=
\sum_{i=1}^{n}
\frac{x_i}{\sigma_i^2}
-
\sum_{i=1}^{n}
\frac{\mu}{\sigma_i^2}
```

```math
\large
0
=
\sum_{i=1}^{n}
\frac{x_i}{\sigma_i^2}
-
\mu
\sum_{i=1}^{n}
\frac{1}{\sigma_i^2}
```

```math
\large
\mu
\sum_{i=1}^{n}
\frac{1}{\sigma_i^2}
=
\sum_{i=1}^{n}
\frac{x_i}{\sigma_i^2}
```

So the maximum likelihood estimator is

```math
\large
\hat{\mu}_{MLE}
=
\frac{
\sum_{i=1}^{n}
\frac{x_i}{\sigma_i^2}
}{
\sum_{i=1}^{n}
\frac{1}{\sigma_i^2}
}
```

This is called the weighted mean. Each observation $x_i$ is weighted by $\frac{1}{\sigma_i^2}$, so observations with smaller variance receive greater weight, and observations with larger variance receive less weight.

## Special case: equal variances

If all variances are equal, $\sigma_1^2 = \sigma_2^2 = \cdots = \sigma_n^2 = \sigma^2$, then the estimator simplifies to

```math
\large
\begin{aligned}
\hat{\mu}
&=
\frac{
\sum_{i=1}^{n}
\frac{1}{\sigma_i^2}x_i
}{
\sum_{i=1}^{n}
\frac{1}{\sigma_i^2}
}
\\
&=
\frac{
\frac{1}{\sigma^2}x_1
+
\frac{1}{\sigma^2}x_2
+
\cdots
+
\frac{1}{\sigma^2}x_n
}{
\frac{1}{\sigma^2}
+
\frac{1}{\sigma^2}
+
\cdots
+
\frac{1}{\sigma^2}
}
\\
&=
\frac{
\frac{1}{\sigma^2}(x_1 + \cdots + x_n)
}{
\frac{n}{\sigma^2}
}
\\
&=
\frac{
\frac{1}{\sigma^2}(x_1 + \cdots + x_n)
}{
\frac{1}{\sigma^2}n
}
\\
&=
\frac{x_1 + \cdots + x_n}{n}
\\
&=
\bar{\mu}
\end{aligned}
```

So when all observations have the same variance, the weighted mean reduces to the ordinary sample mean.
