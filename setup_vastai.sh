#!/bin/bash
# ============================================================================
# Piste 2 — VAST.ai Setup Script (V4)
# Target: RTX 5090 (Blackwell, sm_120) with CUDA 12.8+ and PyTorch 2.7+
# ============================================================================
set -e

echo "============================================"
echo "  Piste 2 — Environment Setup (VAST.ai)"
echo "============================================"
echo ""

# 1. Check GPU
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# 2. Check PyTorch + CUDA
echo "[2/6] Verifying PyTorch & CUDA compatibility..."
python3 -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version    : {torch.version.cuda}')
    print(f'  GPU device      : {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # Quick tensor test
    x = torch.randn(2, 2, device='cuda')
    y = x @ x.T
    print(f'  GPU compute OK  : True')
else:
    print('  ERROR: CUDA not available!')
    exit(1)
"
echo ""

# 3. Install dependencies (includes opencv-python-headless for V2)
echo "[3/6] Installing Python dependencies..."
pip install -r requirements.txt --quiet
echo "  Dependencies installed."
echo ""

# 4. W&B login
echo "[4/6] Weights & Biases setup..."
if python3 -c "import wandb" 2>/dev/null; then
    if [ -z "$WANDB_API_KEY" ]; then
        echo "  WANDB_API_KEY not set. You can either:"
        echo "    1. Run: wandb login"
        echo "    2. Set: export WANDB_API_KEY=your_key"
        echo "    3. Disable wandb in config.yaml (wandb.enabled: false)"
        echo ""
        echo "  Attempting interactive login..."
        wandb login || echo "  Skipping wandb login (will use offline mode or config setting)"
    else
        echo "  WANDB_API_KEY detected. Logging in..."
        wandb login --relogin "$WANDB_API_KEY"
    fi
else
    echo "  WARNING: wandb not installed. Experiment tracking disabled."
fi
echo ""

# 5. Create output directories
echo "[5/6] Creating output directories..."
mkdir -p outputs/checkpoints
mkdir -p outputs/logs
mkdir -p outputs/metrics
mkdir -p outputs/figures
echo "  Output directories created."
echo ""

# 6. Verify dataset access (checks RAW dataset — preprocessing happens in pipeline)
echo "[6/6] Checking dataset access..."
python3 -c "
import os, yaml

# Check config.yaml for dataset path
config_file = 'config.yaml'
with open(config_file, 'r') as f:
    cfg = yaml.safe_load(f)

# Check raw input dir (preprocessing.input_dir or data.root_dir)
preproc = cfg.get('preprocessing', {})
raw_root = preproc.get('input_dir', cfg['data']['root_dir'])

if not os.path.isdir(raw_root):
    print(f'  WARNING: Raw dataset root not found at {raw_root}')
    print(f'  Please upload the dataset or update config data.root_dir')
else:
    normal = os.path.join(raw_root, '01_Normal')
    abnormal = os.path.join(raw_root, '02_Abnormal')
    n_normal = len([d for d in os.listdir(normal) if os.path.isdir(os.path.join(normal, d))]) if os.path.isdir(normal) else 0
    n_abnormal = len([d for d in os.listdir(abnormal) if os.path.isdir(os.path.join(abnormal, d))]) if os.path.isdir(abnormal) else 0
    print(f'  Raw dataset root  : {raw_root}')
    print(f'  Normal patients   : {n_normal}')
    print(f'  Abnormal patients : {n_abnormal}')
    print(f'  Total             : {n_normal + n_abnormal}')
    if preproc.get('enabled', False):
        print(f'  Preprocessed dest : {preproc.get(\"output_dir\", \"N/A\")}')
        print(f'  (will be created during pipeline Step 2: new_prepocessing_pipeline/run_preprocessing.py)')
"
echo ""

echo "============================================"
echo "  Setup V2 complete! Ready to run."
echo "  Next: ./run_all.sh"
echo "============================================"
