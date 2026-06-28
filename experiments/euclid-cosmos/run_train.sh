#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_train
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_train_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # matches NUM_WORKERS in train.py
#SBATCH --mem=32G
#SBATCH --gres=gpu:RTX8000:1        # request 1 RTX8000 GPU specifically
#SBATCH --time=1-00:00:00
#SBATCH --partition=compl      # change to your GPU partition
#SBATCH --chdir=/n03data/fontirro/galaxy-counter
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

echo "=== DIAGNOSTIC ==="
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}"
echo "PATH=$PATH"
echo "command -v python3: $(command -v python3)"
echo "--- contents of venv bin/ (python*) ---"
ls -la /n03data/fontirro/.galaxy-counter-env/bin/ | grep -i python || true
python3 -c "import sys, os; print('executable:', sys.executable); print('realpath:', os.path.realpath(sys.executable))" || true
echo "--- sys.path ---"
python3 -c "import sys; [print(p) for p in sys.path]" || true
echo "--- ls via /n03data path ---"
ls -la /n03data/fontirro/.galaxy-counter-env/lib/python3.12/site-packages/torch 2>&1 | head -5 || true
echo "--- ls via /automnt/n03data path ---"
ls -la /automnt/n03data/fontirro/.galaxy-counter-env/lib/python3.12/site-packages/torch 2>&1 | head -5 || true
echo "=== END DIAGNOSTIC ==="

python3 /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/train.py
