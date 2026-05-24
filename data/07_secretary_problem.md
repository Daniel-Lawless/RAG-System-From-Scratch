## 7. The Secretary Problem

The Secretary Problem is an optimal stopping problem. Imagine you want to hire the best secretary out of $n$ applicants. The problem assumes that applicants arrive one at a time in a random order, we rank them, and we must decide if we hire them or reject them immediately. Once a candidate has been rejected, we cannot go back. The goal of the problem is maximise the probability of choosing the best candidate.

There is a trade off to be made. If we pick too early, then we haven't seen enough candidates to know what good looks like yet. However, if we wait too long, we may have already rejected the best applicant. So we should split our choosing strategy into 2 phases. We will call phase 1 the sample phase and denote it by $r$. In this first phase we take $r$ applicants and use them to set a benchmark. We do not hire people in this phase, we use it to know what a good candidate looks like. We will call phase two the selection phase. In this phase we hire the first person we see that beats our benchmark.

## When does this strategy succeed?

This is best seen with a concrete example. Assume we see 7 candidates and we give them a ranking from 1 to 7, 1 being the best and 5 being the lowest:

```math
\large
\underbrace{3, 7, 2}_{\text{sample}}, 5, 1, 6, 4
```

Suppose that the position of the best overall candidate is at position $k$. Here 1 is the best candidate and they appear after the sample phase. Our proposed strategy only succeeds if two conditions are met:

- The best candidate must come after the sample phase, since they would be skipped otherwise. So: $k > r$

- The best candidate before position $k$ must lie in the sample phase. I.e, in positions $1, \ldots, r$. This ensures that every candidate in positions $r+1, \ldots, k-1$ is worse than our sample benchmark, so none of them will be selected before the true best candidate at position $k$.

First let us figure out what the probability of success is given that the best candidate is at some given position $k$. The layout we have is:

```math
\large
\underbrace{1, 2, \ldots, r}_{\text{sample phase}}
\quad
\underbrace{r+1, \ldots, k-1}_{\text{danger zone}}
\quad
\underbrace{k}_{\text{best overall}}
```

There are $k-1$ applicants before the best applicant. Among those $k-1$ applicants, there is one person who is the best so far before the true best appears. For our strategy to succeed, that person must be in the sample phase. The probability of someone being the second best in any one of these $k-1$ positions is:

```math
\large
\frac{1}{k-1}
```

We only get a success if they're in one of the $r$ positions, so the probability of success for a given position of the best candidate $k$ is given by:

```math
\large
P(\text{Success} \mid \text{best at position } k)
=
\underbrace{
\frac{1}{k-1}
+
\frac{1}{k-1}
+
\cdots
+
\frac{1}{k-1}
}_{r \text{ times}}
=
\frac{r}{k-1}
```

We know that there are $n$ applicants total and that $k > r$, so the possible values for $k$ are: $r+1, r+2, \ldots, n$. Therefore, the probability of success would be:

```math
\large
\begin{aligned}
P(\text{Success})
&=
P(\text{best is at position } r+1)
P(\text{Success} \mid \text{best is at position } r+1)
\\
&\quad+
P(\text{best is at position } r+2)
P(\text{Success} \mid \text{best is at position } r+2)
\\
&\quad+
\cdots
\\
&\quad+
P(\text{best is at position } n)
P(\text{Success} \mid \text{best is at position } n)
\end{aligned}
```

So:

```math
\large
P(\text{Success})
=
\sum_{k=r+1}^{n}
P(\text{best is at position } k)
P(\text{Success} \mid \text{best is at position } k)
```

We know that the probability of the best candidate being at any position $k$ is $\frac{1}{n}$, and we worked out the probability of success given that the best candidate is at a particular position $k$. Plugging these in we get:

```math
\large
\begin{aligned}
P(\text{Success})
&=
\sum_{k=r+1}^{n}
P(\text{best is at position } k)
P(\text{Success} \mid \text{best is at position } k)
\\
&=
\sum_{k=r+1}^{n}
\frac{1}{n}
\frac{r}{k-1}
\\
&=
\frac{r}{n}
\sum_{k=r+1}^{n}
\frac{1}{k-1}
\end{aligned}
```

We can approximate this sum by a logarithm because it is a partial harmonic sum, thus:

```math
\large
P(\text{Success})
\approx
\frac{r}{n}
\ln\left(
\frac{n}{r}
\right)
```

Now, if we let $x = \frac{r}{n}$, substitute, and take the derivative:

```math
\large
P(\text{Success})
\approx
x
\ln\left(
\frac{1}{x}
\right)
=
x(\ln(1) - \ln(x))
=
-x\ln(x)
```

```math
\large
\frac{d}{dx}
[-x\ln(x)]
=
-x \cdot \frac{1}{x}
-
\ln(x)
=
-1
-
\ln(x)
```

Set the derivative equal to 0:

```math
\large
\begin{aligned}
0
&=
-1
-
\ln(x)
\\
\ln(x)
&=
-1
\\
x
&=
\frac{1}{e}
\end{aligned}
```

Since $x = \frac{r}{n}$, we get:

```math
\large
\begin{aligned}
\frac{r}{n}
&\approx
\frac{1}{e}
\\
r
&\approx
\frac{n}{e}
\end{aligned}
```

Thus, the best possible chance of success we can get is:

```math
\large
\begin{aligned}
P(\text{success})
&=
\frac{\frac{n}{e}}{n}
\ln\left(
\frac{n}{\frac{n}{e}}
\right)
\\
&=
\frac{n}{en}
\ln\left(
\frac{en}{n}
\right)
\\
&=
\frac{1}{e}
\ln(e)
\\
&=
\frac{1}{e}
\\
&\approx
0.37
\end{aligned}
```

So, in the secretary problem the optimal strategy of rejecting the first $$\frac{n}{e} \approx 37 \$$ % of candidates and then choosing the first candidate that is the best out of those rejected gives us a 37% chance of picking the best candidate. The Secretary problem is closely related to AI because it shows a fundemental challenge in AI, that is when to stop exploring and start exploiting. At first an agent will explore to learn about its environment and gather information on what a good option looks like. But, at some point the agent must make a decision based on the information it has seen without knowing future options will appear.

This problem comes up in many different areas of AI, such as reinforcement learning, recommendation systems, hyperparameter tuning, etc. So although the secretary problem is formulated as a hiring problem, it is actual a simple mathematical model for optimal stopping under uncertainty.
