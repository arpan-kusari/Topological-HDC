```markdown
## Topology-HDC

Topology-guided hyperdimensional computing (HDC) experiments for robust image classification under distribution shifts.  
This repository implements an offline experiment runner for MNIST/EMNIST/SVHN-style benchmarks using HOG, spatial-pyramid Zernike descriptors, and hole/topological shape descriptors encoded into high-dimensional bipolar hypervectors.

The code is designed for reproducible production-type experiments: config-driven runs, feature extraction, HDC prototype training, late fusion, OnlineHD adaptation, and artifact logging.

---

## Overview

Standard pixel-based HDC representations can degrade sharply under small corruptions such as rotation, noise, cutout, and zoom. This project augments HDC with explicit topological/geometric features:

- **HOG features** for local stroke/edge structure
- **Spatial-pyramid Zernike features** for global and coarse regional shape
- **Hole descriptors** for topological structure, including hole shape and relative geometry
- **Late fusion** of separate HDC prototype classifiers
- **OnlineHD-style adaptation** after initial prototype construction

The main pipeline:

1. Load dataset and create a stratified train/validation split.
2. Apply a specified corruption to the test set.
3. Extract HOG, Zernike, and hole features.
4. Encode features into bipolar hypervectors.
5. Train class prototypes for each feature channel.
6. Tune late-fusion weights on validation data.
7. Evaluate on corrupted test data.
8. Apply OnlineHD training and evaluate again.
9. Save configs, metrics, predictions, confusion matrices, and prototypes.

---

## Repository Structure

```text
Topology-HDC/
  pyproject.toml
  README.md
  configs/
    mnist.yaml                 # optional config files
  scripts/
    run_mnist_sweep.sh         # optional experiment sweep script
  runs/                        # generated experiment outputs
  cache/                       # optional feature cache
  src/
    topo_hdc/
      __init__.py
      cli.py                   # command-line entry point
      run.py                   # experiment orchestration
      config.py                # dataclass configs
      data.py                  # dataset loading/splitting
      corruptions.py           # corruption dispatcher
      plot.py                  # plotting utilities
      Image_processing.py      # image feature utilities
      features/
        __init__.py
        extractor.py           # feature extraction + padding
      hdc/
        __init__.py
        encoders.py            # HDC encoder fitting + roles
        prototypes.py          # prototype and OnlineHD training
        fusion.py              # late fusion prediction/tuning
        RandomProjEncoder.py
        HoleSetHDC.py
        Helper_functions.py
      utils/
        __init__.py
        batching.py            # HV batch generators
        io.py                  # saving configs/metrics/artifacts
  tests/
    test_prototypes.py
    test_alignment.py
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Topology-HDC.git
cd Topology-HDC
```

### 2. Create an environment

Using `conda`:

```bash
conda create -n topo-hdc python=3.10 -y
conda activate topo-hdc
```

or using `venv`:

```bash
python -m venv .topo-hdc-env
source .topo-hdc-env/bin/activate
```

### 3. Install PyTorch with CUDA support

For CUDA 12.4:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Expected output should include something like:

```text
2.x.x+cu124
12.4
True
```

If you do not have a CUDA GPU, install CPU PyTorch instead:

```bash
python -m pip install torch torchvision
```

### 4. Install the package

From the repository root:

```bash
python -m pip install -e .
```

This installs the package in editable mode and registers the CLI command if configured in `pyproject.toml`.

---

## Basic Usage

### Run clean MNIST

```bash
topo-hdc-run \
  --dataset mnist \
  --seed 0 \
  --corrupt none \
  --out-dir runs/mnist/clean_seed0/
```

Equivalent module call:

```bash
python -m topo_hdc.cli \
  --dataset mnist \
  --seed 0 \
  --corrupt none \
  --out-dir runs/mnist/clean_seed0/
```

### Run with cutout corruption

```bash
topo-hdc-run \
  --dataset mnist \
  --seed 0 \
  --corrupt cutout \
  --cutout-size 12 \
  --epochs 30 \
  --hmax 4 \
  --out-dir runs/mnist/cutout12_seed0/
```

### Run with rotation

```bash
topo-hdc-run \
  --dataset mnist \
  --seed 0 \
  --corrupt rotation \
  --angle-deg 20 \
  --out-dir runs/mnist/rotation20_seed0/
```

### Run with Gaussian noise

```bash
topo-hdc-run \
  --dataset mnist \
  --seed 0 \
  --corrupt gaussian \
  --sigma 0.1 \
  --out-dir runs/mnist/gaussian01_seed0/
```

### Run with salt-and-pepper noise

```bash
topo-hdc-run \
  --dataset mnist \
  --seed 0 \
  --corrupt saltpepper \
  --saltpepper-p 0.1 \
  --out-dir runs/mnist/saltpepper01_seed0/
```

### Run with zoom

```bash
topo-hdc-run \
  --dataset mnist \
  --seed 0 \
  --corrupt zoom \
  --zoom 0.75 \
  --out-dir runs/mnist/zoom075_seed0/
```

---

## CLI Options

Common options:

```text
--dataset              Dataset name: mnist, emnist, svhn
--seed                 Random seed
--device               Device: cuda or cpu
--dim                  Hypervector dimension
--batch-size           Batch size for HDC encoding/training
--epochs               OnlineHD training epochs
--lr                   OnlineHD learning rate
--val-frac             Validation split fraction
--corrupt              Corruption: none, rotation, gaussian, saltpepper, cutout, zoom
--angle-deg            Rotation angle in degrees
--sigma                Gaussian noise standard deviation
--saltpepper-p         Salt-and-pepper probability
--cutout-size          Cutout square size
--zoom                 Zoom scale
--hole-k-shape         Number of Fourier coefficients for hole shape
--hole-q               Quantization levels for hole encoding
--hmax                 Maximum number of holes kept per image
--out-dir              Output directory for run artifacts
--cache-dir            Feature cache directory
--no-cache             Disable feature caching
```

Check your installed CLI:

```bash
topo-hdc-run --help
```

or:

```bash
python -m topo_hdc.cli --help
```

---

## Experiment Outputs

Each run writes artifacts to the specified `--out-dir`, for example:

```text
runs/mnist/cutout12_seed0/
  config.json
  metrics.json
  y_test.npy
  predictions_before.npy
  predictions_after.npy
  confusion_before.npy
  confusion_after.npy
  confusion_before.png
  confusion_after.png
  prototypes_before.pt
  prototypes_after.pt
  run.log
```

### `metrics.json`

Contains scalar metrics such as:

```json
{
  "acc_before": 0.9322,
  "acc_after": 0.9703,
  "alpha_before": 0.4,
  "beta_before": 0.2,
  "alpha_after": 0.5,
  "beta_after": 0.1,
  "num_train": 48000,
  "num_val": 12000,
  "num_test": 10000,
  "runtime_sec": 1234.5
}
```

### Prototype artifacts

Prototype files are saved as PyTorch objects:

```text
prototypes_before.pt
prototypes_after.pt
```

They contain:

```python
{
    "protos": {
        "hog": ...,
        "zernike": ...,
        "holes": ...
    },
    "alpha": ...,
    "beta": ...
}
```

---

## Reproducing Sweeps

Example sweep script:

```bash
#!/usr/bin/env bash
set -e

SEED=0
EPOCHS=30
HMAX=4

topo-hdc-run --dataset mnist --seed $SEED --corrupt none \
  --epochs $EPOCHS --hmax $HMAX \
  --out-dir runs/mnist/clean_seed${SEED}/

topo-hdc-run --dataset mnist --seed $SEED --corrupt rotation --angle-deg 20 \
  --epochs $EPOCHS --hmax $HMAX \
  --out-dir runs/mnist/rotation20_seed${SEED}/

topo-hdc-run --dataset mnist --seed $SEED --corrupt gaussian --sigma 0.1 \
  --epochs $EPOCHS --hmax $HMAX \
  --out-dir runs/mnist/gaussian01_seed${SEED}/

topo-hdc-run --dataset mnist --seed $SEED --corrupt gaussian --sigma 0.2 \
  --epochs $EPOCHS --hmax $HMAX \
  --out-dir runs/mnist/gaussian02_seed${SEED}/

topo-hdc-run --dataset mnist --seed $SEED --corrupt saltpepper --saltpepper-p 0.1 \
  --epochs $EPOCHS --hmax $HMAX \
  --out-dir runs/mnist/saltpepper01_seed${SEED}/

for SIZE in 4 8 12; do
  topo-hdc-run --dataset mnist --seed $SEED --corrupt cutout --cutout-size $SIZE \
    --epochs $EPOCHS --hmax $HMAX \
    --out-dir runs/mnist/cutout${SIZE}_seed${SEED}/
done

for ZOOM in 0.75 0.5; do
  topo-hdc-run --dataset mnist --seed $SEED --corrupt zoom --zoom $ZOOM \
    --epochs $EPOCHS --hmax $HMAX \
    --out-dir runs/mnist/zoom${ZOOM}_seed${SEED}/
done
```

Run:

```bash
chmod +x scripts/run_mnist_sweep.sh
./scripts/run_mnist_sweep.sh
```

---

## Method Summary

The model uses three channels:

1. **HOG channel**
   \[
   H_h(x) = \mathrm{Bind}(\mathrm{Enc}_h(f_h(x)), r_h)
   \]

2. **Zernike/spatial-pyramid shape channel**
   \[
   H_z(x) = \mathrm{Bind}(\mathrm{Enc}_z(f_z(x)), r_z)
   \]

3. **Hole/topology channel**
   \[
   H_o(x) = \mathrm{Bind}(\mathrm{Enc}_o(f_o(x)), r_o)
   \]

Each channel forms class prototypes by bundling training hypervectors:

\[
P_t^c = \sum_{i:y_i=c} H_t(x_i),
\qquad t\in\{h,z,o\}.
\]

Prediction uses late fusion of cosine similarities:

\[
s_c(x)
=
\cos(H_h(x),P_h^c)
+
\alpha \cos(H_z(x),P_z^c)
+
\beta \cos(H_o(x),P_o^c),
\]

where \(\alpha,\beta\) are tuned on a validation set.

OnlineHD adaptation updates prototypes when a sample is misclassified:

\[
P^{y_i} \leftarrow P^{y_i} + \eta H(x_i),
\qquad
P^{\hat y_i} \leftarrow P^{\hat y_i} - \eta H(x_i).
\]

---

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

Format code:

```bash
black src tests
```

Lint:

```bash
ruff check src tests
```

---

## Troubleshooting

### `topo-hdc-run: command not found`

Run:

```bash
python -m pip install -e .
```

Then check:

```bash
python -m topo_hdc.cli --help
```

If the module command works but `topo-hdc-run` does not, your environment’s script directory may not be on `PATH`.

### `ModuleNotFoundError: No module named topo_hdc`

Install the package from the repository root:

```bash
python -m pip install -e .
```

or temporarily run with:

```bash
PYTHONPATH=src python -m topo_hdc.cli --help
```

### PyTorch installed without CUDA

Uninstall CPU PyTorch:

```bash
python -m pip uninstall -y torch torchvision torchaudio
```

Install CUDA 12.4 PyTorch:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### `PosixPath is not JSON serializable`

Use `json.dump(..., default=str)` when saving configs:

```python
json.dump(config_dict, f, indent=2, default=str)
```

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{kusari2026topohdc,
  title     = {Encoding Robust Topological Signatures for Hyperdimensional Computing},
  author    = {Arpan Kusari},
  booktitle = {arXiv},
  year      = {2026}
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

This repository uses NumPy, SciPy, scikit-image, scikit-learn, TensorFlow/Keras datasets, PyTorch, mahotas, OpenCV, matplotlib, seaborn, and joblib.
```
