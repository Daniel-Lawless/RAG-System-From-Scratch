## 4. Buffon's Needle

Imagine a floor with infinitely many parallel horizontal lines, each separated by a fixed distance $t$. Now we drop a needle of length $L$ onto
the floor at random. The question is, what is the probability that the needle crosses one of these lines? The standard case is $L \leq t$.

The diagram below shows the basic Buffon's Needle setup. The floor is represented by infinitely many parallel horizontal lines, each separated by distance $t$. A needle of length $L$ is dropped at random. A crossing occurs if any part of the needle touches or passes over one of the horizontal lines.

<p align="center">
  <img src="https://raw.githubusercontent.com/Daniel-Lawless/Statistics-for-ai/main/assets/buffons-needle-images/buffons-needle-problem-setup.png" alt="Buffon's needle problem setup" width="400">
</p>

To make it easier to visualize, let's do a vertical translation where we imagine sliding all the needles to sit between the same pair of lines.

The diagram below illustrates the vertical translation simplification. Because the floor consists of repeated parallel strips of width $t$, we can translate every needle vertically so that its centre lies between the same pair of neighbouring lines. This does not change whether the needle crosses a line; it just lets us study one representative strip instead of infinitely many strips.

<p align="center">
  <img src="https://raw.githubusercontent.com/Daniel-Lawless/Statistics-for-ai/main/assets/buffons-needle-images/buffons-needle-vertical-translation.png" alt="Buffon's needle vertical translation" width="700">
</p>

When the needle lands on the floor, there are two things that really matter, its angle, and its distance from the nearest line. Let $\theta$ be the angle the needle makes from the horizontal lines, represented by the random variable $\Theta$. We do not need all angles from $0$ to $2\pi$, so we can reduce this to:

```math
\large
\theta \in \left[0, \frac{\pi}{2}\right]
```

Since angles outside of this range are just mirrored versions of angles inside it. Since each angle is equally likely to occur, we can model $\Theta$ as:

```math
\large
\Theta \sim \text{Uniform}\left(0, \frac{\pi}{2}\right)
```

Let $x$ be the perpendicular distance from the centre of the needle to the closest line, represented by the random variable $X$. Since the distance between the horizontal lines is $t$, the furthest the needle centre can be from the nearest line is halfway between them. Thus:

```math
\large
x \in \left[0, \frac{t}{2}\right]
```

We assume all such positions are equally likely, so:

```math
\large
X \sim \text{Uniform}\left(0, \frac{t}{2}\right)
```

At this point our problem looks like:

```math
\large
X \sim \text{Uniform}\left(0, \frac{t}{2}\right),
\qquad
f_X(x) = \frac{2}{t},
\qquad
\text{for } x \in \left[0, \frac{t}{2}\right]
```

```math
\large
\Theta \sim \text{Uniform}\left(0, \frac{\pi}{2}\right),
\qquad
f_{\Theta}(\theta) = \frac{2}{\pi},
\qquad
\text{for } \theta \in \left[0, \frac{\pi}{2}\right]
```

Knowing these two variables will tell us whether or not the needle has crossed a line. Assuming $X$ and $\Theta$ are independent, to get a joint PDF we just multiply their individual PDFs:

```math
\large
f_{X,\Theta}(x,\theta)
=
\frac{2}{t}
\cdot
\frac{2}{\pi}
=
\frac{4}{\pi t},
\qquad
0 \leq x \leq \frac{t}{2},
\qquad
0 \leq \theta \leq \frac{\pi}{2}
```

If we take the needle and split it into 2 halves, each half has length $L/2$. If the needle is at angle $\theta$ then the distance from the center of the needle to one endpoint is:

The diagram below shows the geometric crossing condition. Splitting the needle into two halves gives a half-length of $L/2$. If the needle makes angle $\theta$ with the horizontal lines, then the perpendicular vertical distance from the centre of the needle to an endpoint is $(L/2)\sin(\theta)$.

<p align="center">
  <img src="https://raw.githubusercontent.com/Daniel-Lawless/Statistics-for-ai/main/assets/buffons-needle-images/buffons-needle-crossing-condition.png" alt="Buffon's needle crossing condition" width="800">
</p>

```math
\large
y
=
\frac{L}{2}\sin(\theta)
```

Therefore the crossing condition is:

```math
\large
x
\leq
\frac{L}{2}\sin(\theta)
```

Now all we need to do is calculate the probability of this event, since when this is true, a needle lands on a line. This event can occur with any $\theta$, so it is free to vary from $0$ to $\pi/2$. For this event to occur, $x$ can be from $0$ up to the crossing condition.

```math
\large
\begin{array}{rcll}
P\left(
X \leq \frac{L}{2}\sin\theta
\right)
&=&
\displaystyle
\int_{0}^{\pi/2}
\int_{0}^{\frac{L}{2}\sin\theta}
\frac{4}{\pi t}
\, dx \, d\theta
\\[1em]

&=&
\displaystyle
\int_{0}^{\pi/2}
\left[
\frac{4}{\pi t}x
\right]_{0}^{\frac{L}{2}\sin\theta}
d\theta
&\qquad
\text{Integrate w.r.t. } x
\\[1em]

&=&
\displaystyle
\frac{2L}{\pi t}
\int_{0}^{\pi/2}
\sin\theta
\, d\theta
&\qquad
\text{Simplify constants}
\\[1em]

&=&
\displaystyle
\frac{2L}{\pi t}
\left[
-\cos\theta
\right]_{0}^{\pi/2}
&\qquad
\text{Integrate w.r.t. } \theta
\\[1em]

&=&
\displaystyle
\frac{2L}{\pi t}
&\qquad
\text{Final Probability}
\end{array}
```

This is actually a very interesting result since we can use it to approximate $\pi$. Assume we are at home and we have lines drawn on our floor of equal length apart, and suppose we know the distance
between those lines and the size of our needle, then we can approximate $\pi$. Say we take our needle and throw it at random and count whenever the needle crosses a line and repeat this many, many times, this will approximate:

```math
\large
P(\text{crossing})
\approx
\frac{2L}{\pi t}
```

Then, since we know $L$ and $t$, we can rearrange for and get an approximation for $\pi$:

```math
\large
\pi
\approx
\frac{2L}{tP(\text{crossing})}
```

In reality, we have better methods for calculating $\pi$, but there are many applications in physics and engineering where certain quantities are hard to calculate directly and can be approximated using a trick of this kind using simulation. These are called Monte Carlo simulations.
