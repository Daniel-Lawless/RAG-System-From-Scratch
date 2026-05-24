## 8. The Lighthouse Problem

Assume a lighthouse periodically flashes its light toward a shoreline. Given that we know how far the lighthouse is from the shore (its $y$-coordinate), 
find its x-coordinate. Let $(\alpha, \beta)$ be the lighthouse's $x$ and $y$ coordinates respectively. Then assume $x_1, \ldots, x_n$ 
are the points at which the flashes hit the shoreline.

The diagram below shows the setup for the Lighthouse Problem. The lighthouse is located at coordinates $(\alpha, \beta)$, where $\alpha$ is its unknown
horizontal position and $\beta$ is its known distance from the shoreline. The shoreline is represented by the $x$-axis, and each flash of light hits the
shoreline at some observed point $x_i$. The random angle $\Theta$ determines where a flash lands on the shore. The goal is to estimate the unknown lighthouse
position $\alpha$ from the observed hit points $x_1, \ldots, x_n$.

<p align="center">
  <img src="https://raw.githubusercontent.com/Daniel-Lawless/Statistics-for-ai/main/assets/lighthouse-problem-images/lighthouse-problem-setup.png" alt="Lighthouse problem setup" width="600">
</p>

## Bayes rule method

```math
\large
p(\alpha \mid \beta, x_1, \ldots, x_n)
=
\frac{
p(x_1, \ldots, x_n \mid \alpha, \beta)p(\alpha)
}{
\int p(x_1, \ldots, x_n \mid \alpha, \beta)p(\alpha)\,d\alpha
}
```

Initially, before seeing any flashes, we do not know anything about the horizontal coordinate of the lighthouse, $\alpha$,  
so we choose a flat prior. Since $\alpha$ can span the real numbers, we use an improper prior:

```math
\large
p(\alpha) \propto 1,
\qquad
\forall \alpha \in \mathbb{R}
```

Since the denominator is only a normalizing constant, we can write the posterior probability as proportional to the numerator:

```math
\large
p(\alpha \mid \beta, x_1, \ldots, x_n)
\propto
p(x_1, \ldots, x_n \mid \alpha, \beta)p(\alpha)
```

## Figuring out the likelihood

We need to work out the density of $X$.

For one flash,

```math
\large
\tan(\Theta) = \frac{X - \alpha}{\beta}
\qquad \Longrightarrow \qquad
X = \beta \tan(\Theta) + \alpha
```

To get the density of $X$, we use transformations of continuous random variables.

```math
\large
x = \beta \tan(\theta) + \alpha
```

is a one-to-one mapping for

```math
\large
-\frac{\pi}{2} < \theta < \frac{\pi}{2}
```

so only one small interval in $\Theta$-space corresponds to a given small interval in $X$-space.

Let

```math
\large
P(x \leq X < x + dx) = f_X(x)\,dx
```

and

```math
\large
P(\theta \leq \Theta < \theta + d\theta) = f_\Theta(\theta)\,d\theta
```

Since probability is conserved under the transformation, we have

```math
\large
f_X(x) = f_\Theta(\theta)\left|\frac{d\theta}{dx}\right|
```

From

```math
\large
\Theta \sim \text{Uniform}\left(-\frac{\pi}{2}, \frac{\pi}{2}\right),
```

we know

```math
\large
f_\Theta(\theta) = \frac{1}{\pi},
\qquad
-\frac{\pi}{2} < \theta < \frac{\pi}{2}
```

so

```math
\large
f_X(x) = \frac{1}{\pi}\left|\frac{d\theta}{dx}\right|
```

Now derive

```math
\large
\frac{d\theta}{dx}.
```

Rearranging for $\theta$,

```math
\large
\theta = \tan^{-1}\left(\frac{x - \alpha}{\beta}\right)
```

Differentiate:

```math
\large
\frac{d\theta}{dx}
=
\frac{1}{1 + \left(\frac{x - \alpha}{\beta}\right)^2}
\cdot
\frac{1}{\beta}
```

Simplifying,

```math
\large
\frac{d\theta}{dx}
=
\frac{\beta}{\beta^2 + (x - \alpha)^2}
```

Substituting back in,

```math
\large
f_X(x)
=
\frac{1}{\pi}
\frac{\beta}{\beta^2 + (x - \alpha)^2}
```

Since $\beta > 0$ and $\beta^2 + (x - \alpha)^2 > 0$, the expression is always positive, so the absolute value is unnecessary.

Thus,

```math
\large
f_X(x)
=
\frac{1}{\pi}
\frac{\beta}{\beta^2 + (x - \alpha)^2},
\qquad
x \in \mathbb{R}
```

This is the PDF of a Cauchy distribution with location parameter $\alpha$ and scale parameter $\beta$.

## Likelihood of $n$ observations

Now that we have the density of one observation $X$, the likelihood of the $n$ observed flashes, assuming they are independent, is

```math
\large
p(x_1, \ldots, x_n \mid \alpha, \beta)
=
\prod_{i=1}^{n}
\frac{1}{\pi}
\frac{\beta}{\beta^2 + (x_i - \alpha)^2}
```

Viewing this as a function of $\alpha$, the likelihood is

```math
\large
L(\alpha \mid x_1, \ldots, x_n, \beta)
=
\prod_{i=1}^{n}
\frac{1}{\pi}
\frac{\beta}{\beta^2 + (x_i - \alpha)^2}
```

## Substituting into Bayes' rule

Substituting the likelihood and the prior $p(\alpha) \propto 1$ into Bayes' rule gives

```math
\large
p(\alpha \mid \beta, x_1, \ldots, x_n)
\propto
\left(
\prod_{i=1}^{n}
\frac{1}{\pi}
\frac{\beta}{\beta^2 + (x_i - \alpha)^2}
\right)(1)
```

so

```math
\large
p(\alpha \mid \beta, x_1, \ldots, x_n)
\propto
\prod_{i=1}^{n}
\frac{1}{\pi}
\frac{\beta}{\beta^2 + (x_i - \alpha)^2}
```

## Finding the optimal value of $\alpha$

Now, if we want to find the $x$-coordinate $\alpha$ that best explains the observed flashes $x_1, \ldots, x_n$, we maximise the MAP.  
Since we chose a flat improper prior, this will be equal to the MLE.

```math
\large
p(\alpha \mid \beta, x_1, \ldots, x_n)
\propto
\prod_{i=1}^{n}
\frac{1}{\pi}
\frac{\beta}{\beta^2 + (x_i - \alpha)^2}
```

Let

```math
\large
\begin{aligned}
\mathcal{J}(\alpha) &= \ln\left( \prod_{i=1}^{n} \frac{1}{\pi} \frac{\beta}{\beta^2 + (x_i - \alpha)^2} \right) \\
&= \sum_{i=1}^{n} \left[ \ln\left( \frac{1}{\pi} \frac{\beta}{\beta^2 + (x_i - \alpha)^2} \right) \right] \\
&= \sum_{i=1}^{n} \left[ \ln\left(\frac{1}{\pi}\right) + \ln(\beta) - \ln\left(\beta^2 + (x_i - \alpha)^2\right) \right]
\end{aligned}
```

Differentiate with respect to $\alpha$:

```math
\large
\frac{d\mathcal{J}(\alpha)}{d\alpha}
=
\sum_{i=1}^{n}
\left[
\frac{2(x_i - \alpha)}
{\beta^2 + (x_i - \alpha)^2}
\right]
```

Let the $x$-coordinate that best describes the observed flashes be $\hat{\alpha}$. This value would be one such that

```math
\large
\sum_{i=1}^{n}
\left[
\frac{x_i - \hat{\alpha}}
{\beta^2 + (x_i - \hat{\alpha})^2}
\right]
=
0
```

Unlike the Gaussian case, where we obtain a neat closed-form expression for the maximum likelihood estimator, this is not the case for the Cauchy likelihood. Therefore, $\hat{\alpha}$ must generally be found numerically. The resulting $\hat{\alpha}$ is the value of the lighthouse's $x$-location that makes the observed flash locations most plausible, and thus is our estimate of the lighthouse's horizontal position.
