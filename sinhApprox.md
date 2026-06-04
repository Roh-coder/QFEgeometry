# Plan for Approximating K from sinh(2K) = cot(alpha)

## Goal

We want a systematic way to determine how many terms are needed in the expansion of

$$
\sinh(2K) = \cot(\alpha)
$$

in order to recover $K$ to a chosen accuracy, using variables built from the symmetric and antisymmetric combinations of $\delta_1$ and $\delta_2$, where

$$
\alpha_i = \pi/3 + \delta_i,
$$

and the triangle constraint implies

$$
\delta_1 + \delta_2 + \delta_3 = 0.
$$

## Natural Variables

Define

$$
s = \delta_1 + \delta_2, \qquad a = \delta_1 - \delta_2.
$$

Then

$$
\delta_1 = \frac{s + a}{2}, \qquad \delta_2 = \frac{s - a}{2}, \qquad \delta_3 = -s.
$$

These variables are useful because:

- $s$ is symmetric under $1 \leftrightarrow 2$
- $a$ is antisymmetric under $1 \leftrightarrow 2$
- any observable with triangle-label symmetry can be organized cleanly by parity in $a$

## Exact Starting Point

From the rule,

$$
K(\alpha) = \frac{1}{2}\,\operatorname{arsinh}(\cot \alpha).
$$

Expand around the equilateral point $\alpha = \pi/3$. Define

$$
K_0 = K(\pi/3) = \frac{1}{2}\,\operatorname{arsinh}(1/\sqrt{3}).
$$

Then for small $\delta$,

$$
K(\pi/3 + \delta) = K_0 + c_1 \delta + c_2 \delta^2 + c_3 \delta^3 + \cdots.
$$

The first concrete task is to compute the coefficients $c_n$ to whatever order is needed.

## Edge Labels and Triangle Symmetry

It is better to label the couplings by edges rather than by angle number. For a triangle with vertices $(i,j,k)$, write the three edge couplings as

$$
K_{ij}, \qquad K_{jk}, \qquad K_{ki}.
$$

These labels should satisfy two basic symmetry requirements:

- edge symmetry: $K_{ij} = K_{ji}$
- cyclic triangle symmetry: the triple $(K_{ij}, K_{jk}, K_{ki})$ should transform covariantly under the cyclic permutation $i \to j \to k \to i$

If the rule is that the coupling on edge $(ij)$ is determined by the angle opposite that edge, then the natural identification is

$$
K_{ij} = K(\alpha_k), \qquad K_{jk} = K(\alpha_i), \qquad K_{ki} = K(\alpha_j),
$$

where

$$
K(\alpha) = \frac{1}{2}\,\operatorname{arsinh}(\cot \alpha).
$$

This automatically gives $K_{ij}=K_{ji}$, since the edge is unordered, and it also respects cyclic symmetry because a cyclic relabeling of $(i,j,k)$ just permutes the three opposite angles.

## Re-express the Three Couplings

Choose the pair $(\delta_i, \delta_j)$ as the variables used to define

$$
s = \delta_i + \delta_j, \qquad a = \delta_i - \delta_j.
$$

Then

$$
\delta_i = \frac{s + a}{2}, \qquad \delta_j = \frac{s - a}{2}, \qquad \delta_k = -s.
$$

With the edge-opposite-angle convention,

$$
K_{jk} = K\!\left(\pi/3 + \frac{s + a}{2}\right),
$$

$$
K_{ki} = K\!\left(\pi/3 + \frac{s - a}{2}\right),
$$

$$
K_{ij} = K(\pi/3 - s).
$$

Now expand these in powers of $s$ and $a$.

This automatically separates into:

- even powers of $a$ for symmetric combinations
- odd powers of $a$ for antisymmetric combinations

## Symmetric and Antisymmetric Combinations

Useful derived variables are:

$$
K_+ = \frac{K_{jk} + K_{ki}}{2}, \qquad K_- = \frac{K_{jk} - K_{ki}}{2}.
$$

Their structure must be:

- $K_+$ contains only even powers of $a$
- $K_-$ contains only odd powers of $a$

and $K_{ij}$ depends only on $s$.

This is the cleanest basis for judging which terms matter.

## Practical Expansion Program

### Step 1: Expand the scalar function K(delta)

Write

$$
f(\delta) := \frac{1}{2}\,\operatorname{arsinh}(\cot(\pi/3 + \delta)).
$$

Compute its Taylor series about $\delta = 0$ to order $N$:

$$
f(\delta) = K_0 + \sum_{n=1}^N c_n \delta^n + O(\delta^{N+1}).
$$

### Step 2: Substitute delta_i = (s+a)/2, delta_j = (s-a)/2, delta_k = -s

This gives truncated formulae for $K_{ij}$, $K_{jk}$, $K_{ki}$, and therefore also for $K_+$ and $K_-$.

### Step 3: Organize by symmetry class

Collect terms by monomials:

- symmetric sector: $s$, $s^2$, $a^2$, $s^3$, $s a^2$, ...
- antisymmetric sector: $a$, $s a$, $a^3$, $s^2 a$, ...

This makes clear which corrections can appear in observables that are even or odd under exchange of the labels $i \leftrightarrow j$.

### Step 4: Define an accuracy target

Choose a tolerance such as

$$
|K_{\text{exact}} - K_{\text{trunc}}| < \varepsilon
$$

for a desired $\varepsilon$, for example $10^{-3}$, $10^{-4}$, or relative error below a chosen threshold.

### Step 5: Measure the size of delta in the data

Using the actual triangle data, compute ranges such as:

- max $|\delta_i|$
- max $|s|$
- max $|a|$

The needed truncation order depends on these ranges. Small $|a|$ may let the antisymmetric sector truncate much earlier than the symmetric sector.

### Step 6: Compare truncation orders empirically

For each triangle in the data:

1. compute exact $K_{ij}$, $K_{jk}$, $K_{ki}$ from the corresponding opposite angles
2. compute truncated approximations at orders 1, 2, 3, ...
3. compare errors in:
   - $K_{ij}$, $K_{jk}$, $K_{ki}$
   - $K_+$ and $K_-$
   - any downstream observable of interest

Then identify the smallest order that satisfies the target accuracy across the full dataset.

## Suggested Deliverables

### 1. Symbolic series table

A short table listing the coefficients $c_1, c_2, c_3, ...$ for

$$
K(\pi/3 + \delta).
$$

### 2. Series in (s, a)

Explicit formulae for:

- $K_{ij}(s)$
- $K_{jk}(s,a)$
- $K_{ki}(s,a)$
- $K_+(s,a)$
- $K_-(s,a)$

through order 2, 3, or 4.

### 3. Error plots

Plot truncation error versus:

- $|\delta|$
- $|s|$
- $|a|$
- truncation order

### 4. Final decision rule

A statement of the form:

- first order is enough if max $|\delta_i| < ...$
- second order is enough if max $|s| < ...$ and max $|a| < ...$
- third order is needed for the current q5, K=4 data

or whatever the data actually show.

## Minimal First Pass

The quickest useful workflow is:

1. derive the Taylor expansion of $K(\pi/3+\delta)$ to fourth order
2. convert to $(s,a)$ variables
3. evaluate on the existing triangle-angle data
4. compare exact and truncated $K_{ij}(s)$
5. decide the lowest order that meets the desired tolerance

## Explicit Expansion for K_ij

For the edge coupling opposite vertex $k$,

$$
K_{ij} = K(\alpha_k) = K(\pi/3 - s).
$$

So in the $(s,a)$ basis, $K_{ij}$ actually depends only on $s$ and has no $a$-dependence at all. Through sixth order,

$$
K_{ij}(s) = \frac{1}{2}\operatorname{arsinh}\!\left(\frac{1}{\sqrt{3}}\right)
+ \frac{\sqrt{3}}{3}s
+ \frac{1}{6}s^2
+ \frac{5\sqrt{3}}{54}s^3
+ \frac{7}{72}s^4
+ \frac{17\sqrt{3}}{360}s^5
+ \frac{403}{6480}s^6
+ O(s^7).
$$

Numerically,

$$
K_{ij}(s) \approx 0.2746530721670274
+ 0.5773502691896258\,s
+ 0.1666666666666667\,s^2
+ 0.1603750747748960\,s^3
+ 0.0972222222222222\,s^4
+ 0.0817912881351970\,s^5
+ 0.0621913580246914\,s^6.
$$

## Full Expansions in s and a

The other two edge couplings can be organized as

$$
K_{jk}(s,a) = K_+(s,a) + K_-(s,a),
$$

$$
K_{ki}(s,a) = K_+(s,a) - K_-(s,a),
$$

where $K_+$ is even in $a$ and $K_-$ is odd in $a$.

Through sixth order,

$$
\begin{aligned}
K_+(s,a) = {} & \frac{1}{2}\operatorname{arsinh}\!\left(\frac{1}{\sqrt{3}}\right)
 - \frac{\sqrt{3}}{6}s
 + \frac{1}{24}s^2
 - \frac{5\sqrt{3}}{432}s^3
 + \frac{7}{1152}s^4
 - \frac{17\sqrt{3}}{11520}s^5
 + \frac{403}{414720}s^6 \\
& + a^2\left(
\frac{1}{24}
 - \frac{5\sqrt{3}}{144}s
 + \frac{7}{192}s^2
 - \frac{17\sqrt{3}}{1152}s^3
 + \frac{403}{27648}s^4
\right) \\
& + a^4\left(
\frac{7}{1152}
 - \frac{17\sqrt{3}}{2304}s
 + \frac{403}{27648}s^2
\right)
 + \frac{403}{414720}a^6
 + O\!\left((s,a)^7\right).
\end{aligned}
$$

and

$$
\begin{aligned}
K_-(s,a) = {} & a\left(
-\frac{\sqrt{3}}{6}
+ \frac{1}{12}s
- \frac{5\sqrt{3}}{144}s^2
+ \frac{7}{288}s^3
- \frac{17\sqrt{3}}{2304}s^4
+ \frac{403}{69120}s^5
\right) \\
& + a^3\left(
- \frac{5\sqrt{3}}{432}
+ \frac{7}{288}s
- \frac{17\sqrt{3}}{1152}s^2
+ \frac{403}{20736}s^3
\right) \\
& + a^5\left(
- \frac{17\sqrt{3}}{11520}
+ \frac{403}{69120}s
\right)
 + O\!\left((s,a)^7\right).
\end{aligned}
$$

So the full sixth-order expansions are obtained by combining these as

$$
K_{jk}(s,a) = K_+(s,a) + K_-(s,a),
$$

$$
K_{ki}(s,a) = K_+(s,a) - K_-(s,a).
$$

## Dedicated Evaluation Script

The empirical truncation study is implemented in:

- `Playground/evaluate_k_series_terms.py`

It evaluates the symmetry-respecting expansion for

$$
K_{ij}(s) = K(\pi/3 - s),
$$

using all three cyclic relabelings of $(i,j,k)$ for each triangle. This respects the cyclic symmetry of the triangle labels.

The script compares the exact values against truncated expansions derived from

$$
K(\pi/3 + \delta) = K_0 + c_1 \delta + c_2 \delta^2 + \cdots
$$

and reports the smallest Taylor order whose maximum absolute error is below each target tolerance.

The exact scalar coupling is still

$$
K(\alpha) = \frac{1}{2}\,\operatorname{arsinh}(\cot \alpha)
$$

Command used for the current evaluation:

```bash
python Playground/evaluate_k_series_terms.py
```

By default this checks:

- q5 refined K=4
- q5 refined K=64
- q5 projected K=64

Here, “min terms” counts the constant term, so:

- order 1 means 2 terms: $K_0 + c_1 \delta$
- order 4 means 5 terms: $K_0 + c_1 \delta + \cdots + c_4 \delta^4$
- order 6 means 7 terms: $K_0 + c_1 \delta + \cdots + c_6 \delta^6$

## Current Empirical Results

Quoted output from the script:

```text
q5 refined K=4
  max|s| = 1.951197447765e-01
  max|a| = 2.926796171648e-01
  target 1e-02: K_ij min order = 1, min terms = 2
  target 1e-04: K_ij min order = 4, min terms = 5
  target 1e-06: K_ij min order = 6, min terms = 7

q5 refined K=64
  max|s| = 2.093811086796e-01
  max|a| = 3.147694782338e-01
  target 1e-02: K_ij min order = 1, min terms = 2
  target 1e-04: K_ij min order = 4, min terms = 5
  target 1e-06: K_ij min order = 6, min terms = 7

q5 projected K=64
  max|s| = 2.094034188376e-01
  max|a| = 3.141051282564e-01
  target 1e-02: K_ij min order = 1, min terms = 2
  target 1e-04: K_ij min order = 4, min terms = 5
  target 1e-06: K_ij min order = 6, min terms = 7
```

So for the expansion that actually determines $K_{ij}$, the present datasets give the practical rule:

- for accuracy $10^{-2}$, keep terms through first order
- for accuracy $10^{-4}$, keep terms through fourth order
- for accuracy $10^{-6}$, keep terms through sixth order

These statements are based on the maximum absolute error over all cyclic relabelings of all triangles in each tested dataset.

## Why This Basis Is Useful

The $(s,a)$ basis separates deformation into:

- a symmetric mode that shifts angles $i$ and $j$ together
- an antisymmetric mode that distinguishes angles $i$ and $j$

Because the triangle constraint fixes $\delta_k = -s$, the full three-angle problem reduces naturally to these two variables. This is the right language for both symmetry arguments and practical truncation estimates.