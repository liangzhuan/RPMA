# Image clustering experiments: COIL-20 and Extended Yale B

This project now includes scripts for validating the RPA/RPMA method on image clustering datasets.

## 1. Expected data placement

### COIL-20
Place COIL-20 images under a folder such as:

```text
datasets/data/coil20/
  obj1__0.png
  obj1__5.png
  ...
  obj20__355.png
```

The loader parses labels from `objXX` in the filename.

### Extended Yale B
Place the cropped Yale B dataset under a folder such as:

```text
datasets/data/CroppedYale/
  yaleB01/*.pgm
  yaleB02/*.pgm
  ...
```

The loader parses labels from `yaleBXX` folder names and skips files with `Ambient` in the name.

## 2. Run a quick smoke test

Use a small number of images per class first:

```bash
python -m experiments.run_image_experiment \
  --dataset coil20 \
  --data-root datasets/data/coil20 \
  --image-size 32x32 \
  --max-per-class 10 \
  --pca-dim 50 \
  --k-neighbors 10 \
  --lam 0.02 \
  --delta 1e-3 \
  --rpa-max-iter 50
```

```bash
python -m experiments.run_image_experiment \
  --dataset yaleB \
  --data-root datasets/data/CroppedYale \
  --image-size 32x32 \
  --max-per-class 10 \
  --pca-dim 50 \
  --k-neighbors 10 \
  --lam 0.02 \
  --delta 1e-3 \
  --rpa-max-iter 50
```

## 3. Run fuller experiments

```bash
python -m experiments.run_image_experiment \
  --dataset coil20 \
  --data-root datasets/data/coil20 \
  --image-size 32x32 \
  --pca-dim 100 \
  --k-neighbors 10 \
  --lam 0.02 \
  --delta 1e-3 \
  --rpa-max-iter 200
```

```bash
python -m experiments.run_image_experiment \
  --dataset yaleB \
  --data-root datasets/data/CroppedYale \
  --image-size 32x32 \
  --pca-dim 100 \
  --k-neighbors 10 \
  --lam 0.02 \
  --delta 1e-3 \
  --rpa-max-iter 200
```

## 4. Outputs

Results are written to:

```text
results/image/
```

Each run saves:

- `*_metrics.csv`: ACC, NMI, ARI, runtime, and RPA iteration information.
- `*_matrices.npz`: affinity matrix `A`, RPA projection matrix `X_rpa`, labels, and K.
- `*_config.json`: run configuration.

## 5. Interpretation

For the paper, compare `Spectral` and `RPA-Huber` on ACC/NMI/ARI. If RPA-Huber improves the metrics, it supports the claim that regularized rank-K projection recovery improves downstream clustering on real image data.
