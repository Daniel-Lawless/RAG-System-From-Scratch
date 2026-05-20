## 1. Mean and Variance Derivation

Suppose we are running an experiment where we want to estimate an unknown true value $\mu$ from $n$ independent observations, and we do not know the variance of error in our sensor. Then, each observation is assumed to be normally distributed around $\mu$ with unknown variance:

```math
\large
X_1 \sim \mathcal{N}(\mu, \sigma^2),
\qquad
X_2 \sim \mathcal{N}(\mu, \sigma^2),
\qquad
\ldots
\qquad
X_n \sim \mathcal{N}(\mu, \sigma^2)
```

## Likelihood of one observation

For a single observation $X_i = x_i$, the density is

```math
\large
p(x_i \mid \mu, \sigma)
=
\frac{1}{\sigma \sqrt{2\pi}}
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma^2}
\right)
```

## Likelihood of $n$ observations

Assuming the observations are independent, the joint likelihood is the product of the individual densities:

```math
\large
p(x_1, \ldots, x_n \mid \mu, \sigma)
=
\prod_{i=1}^{n}
\frac{1}{\sigma \sqrt{2\pi}}
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma^2}
\right)
```

Viewing this as a function of $\mu$ and $\sigma$, the likelihood is

```math
\large
L(\mu, \sigma \mid x_1, \ldots, x_n)
=
\prod_{i=1}^{n}
\frac{1}{\sigma \sqrt{2\pi}}
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma^2}
\right)
```

## Taking logs

Let

```math
\large
J(\mu, \sigma) = \ln\left(L(\mu, \sigma)\right)
```

Then

```math
\large
\begin{aligned}
J(\mu, \sigma)
&=
\ln\left(
\prod_{i=1}^{n}
\frac{1}{\sigma \sqrt{2\pi}}
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma^2}
\right)
\right)
\\[1em]
&=
\sum_{i=1}^{n}
\ln\left(
\frac{1}{\sigma \sqrt{2\pi}}
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma^2}
\right)
\right)
\\[1em]
&=
\sum_{i=1}^{n}
\left[
\ln\left(
\frac{1}{\sigma \sqrt{2\pi}}
\right)
+
\ln\left(
\exp\left(
-\frac{(x_i - \mu)^2}{2\sigma^2}
\right)
\right)
\right]
\\[1em]
&=
\sum_{i=1}^{n}
\left[
-\ln\left(\sigma \sqrt{2\pi}\right)
-
\frac{(x_i - \mu)^2}{2\sigma^2}
\right]
\\[1em]
&=
-n\ln\left(\sigma \sqrt{2\pi}\right)
-
\sum_{i=1}^{n}
\frac{(x_i - \mu)^2}{2\sigma^2}
\end{aligned}
```

<h2 align="center">Maximum Likelihood Estimators</h2>

<div align="center">
  
| Finding the optimal value of $\mu$ | Finding the optimal value of $\sigma$ |
|---|---|
| $\large \displaystyle \frac{\partial J(\mu, \sigma)}{\partial \mu} = \frac{\partial}{\partial \mu}\left[-n\ln(\sigma\sqrt{2\pi}) - \sum_{i=1}^{n}\frac{(x_i-\mu)^2}{2\sigma^2}\right]$ | $\large \displaystyle \frac{\partial J(\mu, \sigma)}{\partial \sigma} = \frac{\partial}{\partial \sigma}\left[-n\ln(\sigma\sqrt{2\pi}) - \sum_{i=1}^{n}\frac{(x_i-\mu)^2}{2\sigma^2}\right]$ |
| $\large \displaystyle = 0 - \sum_{i=1}^{n}\frac{\partial}{\partial \mu}\left[\frac{(x_i-\mu)^2}{2\sigma^2}\right]$ | $\large \displaystyle = -\frac{n}{\sigma} - \sum_{i=1}^{n}\frac{\partial}{\partial \sigma}\left[\frac{(x_i-\mu)^2}{2\sigma^2}\right]$ |
| $\large \displaystyle = -\sum_{i=1}^{n}\left[-\frac{1}{\sigma^2}(x_i-\mu)\right]$ | $\large \displaystyle = -\frac{n}{\sigma} - \sum_{i=1}^{n}\left[-\frac{(x_i-\mu)^2}{\sigma^3}\right]$ |
| $\large \displaystyle = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i-\mu)$ | $\large \displaystyle = -\frac{n}{\sigma} + \sum_{i=1}^{n}\frac{(x_i-\mu)^2}{\sigma^3}$ |
| Set equal to $0$: | Set equal to $0$: |
| $\large \displaystyle 0 = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i-\mu)$ | $\large \displaystyle 0 = -\frac{n}{\sigma} + \sum_{i=1}^{n}\frac{(x_i-\mu)^2}{\sigma^3}$ |
| $\large \displaystyle 0 = \sum_{i=1}^{n}x_i - \sum_{i=1}^{n}\mu$ | $\large \displaystyle \frac{n}{\sigma} = \sum_{i=1}^{n}\frac{(x_i-\mu)^2}{\sigma^3}$ |
| $\large \displaystyle 0 = \sum_{i=1}^{n}x_i - n\mu$ | $\large \displaystyle n\sigma^2 = \sum_{i=1}^{n}(x_i-\mu)^2$ |
| $\large \displaystyle n\mu = \sum_{i=1}^{n}x_i$ | $\large \displaystyle \hat{\sigma}^{2} = \frac{1}{n} \sum\limits_{i=1}^{n}(x_i - \hat{\mu}_{MLE})^2$ |
  
</div>

## Final Result

The maximum likelihood estimator for the mean is

```math
\large
\hat{\mu}_{MLE}
=
\bar{x}
=
\frac{1}{n}
\sum_{i=1}^{n} x_i
```

The maximum likelihood estimator for the variance is

```math
\large
\hat{\sigma}_{MLE}^{2}
=
\frac{1}{n}
\sum_{i=1}^{n}
(x_i-\bar{x})^2
```

This shows that, under the normal model, the MLE for the mean is the average of the observations, and the MLE for the variance is the average squared distance from that mean.
