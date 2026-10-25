# Methods

## 1. CNN from scratch (`src/vision/numpy_cnn.py`)

**Architecture** (13,706 parameters):

| Layer | Output | Parameters |
|---|---|---|
| Conv 3×3, 16 filters, pad 1 + ReLU | 16×8×8 | 160 |
| Max-pool 2×2 | 16×4×4 | 0 |
| Conv 3×3, 32 filters, pad 1 + ReLU | 32×4×4 | 4,640 |
| Max-pool 2×2 | 32×2×2 | 0 |
| Flatten → Dense 64 + ReLU + Dropout 0.3 | 64 | 8,256 |
| Dense 10 (softmax) | 10 | 650 |

**Convolution via im2col.** Every k×k patch is unrolled into a row so a convolution becomes one matrix multiply (`patches @ W.T`). The backward pass is the transpose: `col2im` scatter-adds patch gradients back into the image. `col2im` is tested to be the exact adjoint of `im2col`.

**Backpropagation.** Every layer implements `forward` and `backward`. Softmax and cross-entropy are combined, so the logit gradient is simply `p − onehot(y)` divided by batch size (numerically stable). Max-pool routes the gradient to the max location (ties split evenly). Dropout uses inverted scaling, so inference needs no rescale.

**Verification.** Analytic gradients are compared with central finite differences for every parameter tensor (relative error ≈ 1e-9). Biases are randomised for the check because zero biases put blank-pixel activations exactly on the ReLU kink, where finite differences are undefined.

**Training.** Adam (lr 2e-3, β = 0.9 / 0.999, decoupled-style L2 weight decay 1e-4 on weights), batch 64, He initialisation, early stopping on validation loss (patience 6, best weights restored).

**Augmentation.** Each mini-batch is randomly translated by −1, 0 or +1 pixel in each direction (zero-filled). On 8×8 images a 1-pixel shift is large: 12.5% of the image width.

## 2. Baselines
- Logistic regression on 64 flat pixels (`max_iter=3000`).
- MLP with one 128-unit hidden layer (scikit-learn, Adam).

## 3. Evaluation
- **Accuracy and macro-F1**: classes are balanced, so accuracy is fair; macro-F1 guards against a weak class.
- **Log loss**: rewards confident correct predictions and punishes confident mistakes.
- **Expected calibration error (ECE)**: 10 equal-width confidence bins; weighted mean |accuracy − confidence|.
- **Robustness**: the same test images with (a) a random 1-pixel shift and (b) Gaussian noise σ = 0.2, clipped to [0, 1]. The notebook also sweeps σ from 0 to 0.4.
- **Selective prediction**: share of images auto-accepted, and accuracy on them, at confidence thresholds 0.5–0.95.

## 4. Explainability
- **First-layer filters** and **feature maps**.
- **Saliency**: |∂ score_class / ∂ input pixel|, computed with the network's own backward pass.
- **Grad-CAM** (ResNet-18): gradients of the class score w.r.t. `layer4` activations are averaged into channel weights; the weighted, ReLU-ed sum of activations is the heat-map.

## 5. Transfer learning (`src/vision/transfer_learning.py`)
- ResNet-18 pretrained on ImageNet; the final layer is replaced by a new 10-class head. The whole network is fine-tuned (or head only with `--freeze-backbone`).
- Images resized to 224×224 and normalised with ImageNet statistics.
- Augmentation: horizontal and vertical flips (satellite images have no "up"), ±15° rotation, colour jitter.
- AdamW (weight decay 1e-4), OneCycle learning-rate schedule, label smoothing 0.05, mixed precision on CUDA. The epoch with the best validation accuracy is kept.
- Reports accuracy, macro-F1, top-3 accuracy, ECE and a per-class report.
