# Image dataset download and placement

This project expects image datasets under:

```text
Community_detection/
└── datasets/
    └── data/
        ├── coil20/
        └── CroppedYale/
```

## Download COIL-20

From the project root:

```powershell
python scripts/download_image_datasets.py --dataset coil20
```

After success, files should look like:

```text
datasets/data/coil20/obj1__0.png
datasets/data/coil20/obj1__5.png
...
```

Then run:

```powershell
python -m experiments.run_image_experiment --dataset coil20 --data-root datasets/data/coil20 --image-size 32x32 --max-per-class 10 --pca-dim 50 --k-neighbors 10 --lam 0.02 --delta 1e-3 --rpa-max-iter 50
```

## Download Cropped Extended Yale B

Try the legacy UCSD URL first:

```powershell
python scripts/download_image_datasets.py --dataset yaleb
```

If the UCSD legacy URL is unavailable, use the public mirror fallback:

```powershell
python scripts/download_image_datasets.py --dataset yaleb --allow-yale-mirror
```

After success, files should look like:

```text
datasets/data/CroppedYale/yaleB01/*.pgm
datasets/data/CroppedYale/yaleB02/*.pgm
...
```

Then run:

```powershell
python -m experiments.run_image_experiment --dataset yaleB --data-root datasets/data/CroppedYale --image-size 32x32 --max-per-class 10 --pca-dim 50 --k-neighbors 10 --lam 0.02 --delta 1e-3 --rpa-max-iter 50
```
