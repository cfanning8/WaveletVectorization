# Wavelet-Persistence Vectorization for Mammography

Topological conditioning for mammography classification via stable wavelet-persistence vectorization.

## Method

The method computes cubical persistent homology on a grayscale image $I \in [0,1]^{m \times n}$ via sublevel set filtration $K_\delta = \{(x,y) \in \Omega : I(x,y) \leq \delta\}$ to obtain persistence diagrams $D(I) \subset \{(b,d) : 0 \leq b < d \leq 1\}$ in dimensions 0 and 1. If $\|I - \tilde{I}\|_\infty \leq \varepsilon$, then $d_B(D(I), D(\tilde{I})) \leq \varepsilon$ and $W_p^{(\infty)}(D(I), D(\tilde{I})) \leq \varepsilon$ for $p \in [1, \infty]$.

A gate function maps birth-death pairs $(b,d)$ to spatial locations on wavelet subbands. For a $J$-level orthogonal Haar transform with normalized subbands $\tilde{\psi}_{i,j} = \tau(\psi_{i,j})$ where $\tau: \mathbb{R} \to [0,1]$ is monotone and 1-Lipschitz, the gate function $w(\tilde{\psi}; b, d)$ equals $(\tilde{\psi}-b)(d-\tilde{\psi})/(d-b+\varepsilon)$ when $b \leq \tilde{\psi} < d$, and 0 otherwise, where $\varepsilon > 0$ is a regularizer. Aggregation over persistence pairs yields spatial maps: $W_{i,j}(D) = \sum_{(b,d) \in D} W_{i,j}(\cdot; b, d)$ on the wavelet subband grid $\Omega_{i,j}$. The vectorization satisfies $\|W_{i,j}(D) - W_{i,j}(D')\|_F \leq L_p \sqrt{|\Omega_{i,j}|} W_1^{(p)}(D, D')$ for $p \in \{1, 2, \infty\}$ with $L_p \in \{1, \sqrt{2}, 2\}$, which proves global Lipschitz continuity with respect to the 1-Wasserstein distance on persistence diagrams.

## Implementation

The method computes cubical persistent homology on aspect-preserving downsampled images (maximum side length 96 pixels). For $H_0$, the method retains all finite bars except the maximum-lifetime bar. For $H_1$, the method retains a fraction $h_{1,\text{pct}}$ of bars in persistence order. Input-level concatenation integrates the 8-channel vectorization (6 wavelet-aligned channels + 2 baseline channels) into a two-stage pipeline: patch-level multiple-instance learning followed by Faster R-CNN detection.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```
