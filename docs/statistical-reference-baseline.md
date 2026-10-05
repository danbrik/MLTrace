# Statistical Reference Baseline

Select **Statistical Reference Baseline** in **Methods**, set a positive epsilon
(default `0.000001`), and save the method. Create a Training Pipeline using a
Train/Test dataset covering normal operation and the desired preprocessing/ROI.
Fitting runs on the CPU without epochs, optimizers or a validation split.

The model stores pixel-wise mean μ and population standard deviation σ
(`ddof=0`). Images are processed sequentially with Welford's algorithm in
float64. Mean and standard deviation remain float64 in the internal `.npz`
artifact, together with image count and epsilon. Memory use depends on image
size, not image count. There is no extra intensity normalization or rounding.
The pipeline must return finite, equally sized, two-dimensional grayscale images.
Unreadable files and incompatible outputs fail the fit. A single reference image
is allowed and yields σ=0. Stationary spatial correspondence is assumed.

Inference evaluates **A = |I − μ| / (σ + epsilon)**. The score is dimensionless;
epsilon is specified in the intensity units of the preprocessing output. With
zero variance, deviations are divided by epsilon; choosing a very small epsilon
can therefore produce very large scores. Invalid or overflowing values fail
explicitly. Spatial aggregation supports mean (default), p95, p99, p99.9 and max,
including existing ROI/tile aggregation. Score settings cannot replace this
measure with MSE/SSIM. Single-image and video heatmaps also use A; their display
threshold applies to A. The reference preview shows μ.

This implements the requested continuous normalized statistical baseline inspired
by the Variation Model, rather than claiming to reproduce a published thresholding
implementation. Existing Mean Image methods remain unchanged. No migration is
required; the existing project-specific Methods, training and inference interfaces
store this additional method and artifact kind.
