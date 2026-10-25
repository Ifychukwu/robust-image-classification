# Model card: NumPy CNN digit classifier

| | |
|---|---|
| **Model** | 2-conv-layer CNN, 13,706 parameters, written in NumPy |
| **Task** | Classify a single handwritten digit (0–9) from an 8×8 greyscale image |
| **Training data** | 1,257 images from the UCI handwritten digits set (43 writers); 180 for validation |
| **Evaluation data** | 360 held-out images, stratified |
| **Weights** | `models/numpy_cnn_digits.npz` |

## Intended use
A teaching and portfolio model showing how convolution, backpropagation and augmentation work, and how to evaluate robustness and calibration. **Not** intended for reading real documents, cheques, forms or postcodes.

## Performance (test set, n = 360)

| Condition | Accuracy |
|---|---|
| Clean | 97.8% (macro-F1 0.978, log loss 0.051, ECE 0.019) |
| Shifted by 1 pixel | 95.3% |
| Gaussian noise σ = 0.2 | 88.9% |

Per class: digits 0, 6 and 7 are perfect; the lowest F1 is 0.957 (8 and 9). With a confidence threshold of 0.9, 92% of test images are accepted automatically with no errors among them; all 8 errors fall below the threshold. With only 360 test images, these figures carry several points of uncertainty.

## Limitations
- **Noise sensitivity**: trained with shift augmentation only, it is *less* robust to pixel noise than logistic regression. Add noise augmentation before using it on noisy scans.
- **Input format**: expects a single centred digit, light strokes on a dark background, scaled to [0, 1]. The app converts uploads to this format, but photos with shadows, multiple digits or thick borders will be misread.
- **Population**: 43 writers from one collection. Accuracy for other handwriting styles, age groups or numeral systems is unknown.
- **Resolution**: 8×8 discards detail; some errors are genuinely ambiguous at this size.

## Ethical considerations
- Handwriting can identify a person. The demo processes uploads in memory and does not store them.
- Saliency maps show which pixels influence a decision; they are not a guarantee the reasoning is sound.
- In any real deployment, route low-confidence predictions to a human and monitor accuracy by input source.

---

# Model card: ResNet-18 land-use classifier (EuroSAT)

| | |
|---|---|
| **Model** | ResNet-18 (11.2M parameters), ImageNet-pretrained, fine-tuned |
| **Task** | Classify a 64×64 Sentinel-2 RGB tile into 10 land-use classes |
| **Data** | EuroSAT, 27,000 tiles from 34 European countries; 70/15/15 split |
| **Weights** | `models/resnet18_eurosat.pt` (produced by the Colab notebook; not in the repo) |

**Performance:** run `notebooks/transfer_learning_colab.ipynb`; metrics are written to `reports/transfer_metrics_eurosat.json`. Published results for fine-tuned ResNets on EuroSAT RGB are typically around 97–99% accuracy.

**Limitations and considerations**
- Trained on European landscapes only: expect lower accuracy on other climates, building styles and crop types.
- Random splits can put neighbouring, near-identical tiles in train and test, which inflates accuracy. A geographic split is the stricter test.
- Satellite imagery at this resolution does not identify individuals, but land-use maps can still inform decisions about people (insurance, planning). Validate locally before any such use.
