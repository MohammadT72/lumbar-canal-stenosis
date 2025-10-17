# README

## Instructions

### 1. Install PyTorch and torchvision via conda

```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

This installs:

- **PyTorch** 2.3.0
- **torchvision** 0.18.0
- **torchaudio** 2.3.0
- **CUDA toolkit** 12.1

### 2. Install remaining packages via pip

```bash
pip install -r requirements.txt
```

Ensure you're in the directory containing `requirements.txt`.

### 3. Install `ultralytics` package

```bash
pip install ultralytics
```

### 4. Run the Grid Search Training Process

Use `main.py` to start training.

#### Basic Usage

```bash
python main.py
```

#### Command-Line Arguments

Run `python main.py --help` to see all options.

#### Examples

- **Use GPU and save models:**

  ```bash
  python main.py --gpu --save_models
  ```

- **Specify class names:**

  ```bash
  python main.py --list_classes_names cat dog bird
  ```

- **Use a custom config file:**

  ```bash
  python main.py --config_path ./config/grid_search_config.yaml
  ```

### 5. Grid Search Configuration

Define hyperparameters in a YAML file (default is `grid_search_config.yaml` in `folder_path`).

#### Example (`grid_search_config.yaml`)

```yaml
model:
  - resnet18
  - resnet34
learning_rate:
  - 0.001
  - 0.0001
batch_size:
  - 32
  - 64
optimizer:
  - Adam
  - SGD
```

### 6. Dataset Structure

Organize your dataset under `--folder_path` with proper annotations and images.

### 7. Results

Outputs are saved in `--save_path`, including logs and optionally models.

### 8. Additional Notes

- **GPU Acceleration:** Use `--gpu`.
- **AMP:** Enable with `--amp`.
- **Over-Sampling:** Use `--over_sampling` for class imbalance.
- **Data Fraction:** Use `--data_fraction` to subset data.

### 9. Dependencies

Ensure all dependencies are installed, including those in `requirements.txt`.

---
