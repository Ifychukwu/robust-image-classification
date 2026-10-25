# Data

Nothing needs downloading for the main project.

| Dataset | Used by | How to get it | Size |
|---|---|---|---|
| **UCI handwritten digits** (1,797 images, 8×8 greyscale, 10 classes, 43 writers) | NumPy CNN, baselines, notebook, app | Bundled with scikit-learn: `sklearn.datasets.load_digits()` | < 1 MB |
| **EuroSAT RGB** (27,000 Sentinel-2 tiles, 64×64, 10 land-use classes) | ResNet-18 transfer learning | Downloaded automatically by `torchvision.datasets.EuroSAT` into `data/eurosat/` on first run | ~90 MB |
| **Your own images** | ResNet-18 transfer learning | Arrange as `data/images/<class_name>/*.jpg` and run `python -m vision.transfer_learning --data folder --root data/images` | any |

Splits:
- Digits: stratified 70 / 10 / 20 (train / validation / test), `random_state=0`. The validation set drives early stopping; test is used once.
- EuroSAT: random 70 / 15 / 15 with a fixed seed (42).

Licences: the digits set is from the UCI Machine Learning Repository (CC BY 4.0). EuroSAT is released by Helber et al. (2019) under the MIT licence, built on Copernicus Sentinel-2 data.

`data/eurosat/` and `data/images/` are git-ignored.
