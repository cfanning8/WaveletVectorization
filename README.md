# Wavelet-Persistence Vectorization for Mammography

Topological conditioning for mammography classification via stable wavelet-persistence vectorization.

## Method

The method computes cubical persistent homology on a grayscale image I ∈ [0,1]^(m×n) via sublevel set filtration K_δ = {(x,y) ∈ Ω : I(x,y) ≤ δ} to obtain persistence diagrams D(I) ⊂ {(b,d) : 0 ≤ b < d ≤ 1} in dimensions 0 and 1. If ||I - Ĩ||_∞ ≤ ε, then d_B(D(I), D(Ĩ)) ≤ ε and W_p^(∞)(D(I), D(Ĩ)) ≤ ε for p ∈ [1, ∞].

A gate function maps birth-death pairs (b,d) to spatial locations on wavelet subbands. For a J-level orthogonal Haar transform with normalized subbands ψ̃_{i,j} = τ(ψ_{i,j}) where τ: ℝ → [0,1] is monotone and 1-Lipschitz, the gate function is

```
w(ψ̃; b, d) = { (ψ̃-b)(d-ψ̃)/(d-b+ε),  if b ≤ ψ̃ < d
              { 0,                     otherwise
```

where ε > 0 is a regularizer. Aggregation over persistence pairs yields spatial maps: W_{i,j}(D) = Σ_{(b,d)∈D} W_{i,j}(·; b, d) on the wavelet subband grid Ω_{i,j}. The vectorization satisfies ||W_{i,j}(D) - W_{i,j}(D')||_F ≤ L_p √|Ω_{i,j}| W_1^(p)(D, D') for p ∈ {1, 2, ∞} with L_p ∈ {1, √2, 2}, which proves global Lipschitz continuity with respect to the 1-Wasserstein distance on persistence diagrams.

## Implementation

The method computes cubical persistent homology on aspect-preserving downsampled images (maximum side length 96 pixels). For H_0, the method retains all finite bars except the maximum-lifetime bar. For H_1, the method retains a fraction h_{1,pct} of bars in persistence order. Input-level concatenation integrates the 8-channel vectorization (6 wavelet-aligned channels + 2 baseline channels) into a two-stage pipeline: patch-level multiple-instance learning followed by Faster R-CNN detection.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```
