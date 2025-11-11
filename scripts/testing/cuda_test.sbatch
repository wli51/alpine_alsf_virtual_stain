#!/bin/bash
#SBATCH --job-name=test-gpu-qos
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --time=00:02:00 # 2 minutes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=est_gpu_%j.out
#SBATCH --error=test_gpu_%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vsf2  # or a lighter env with torch if you prefer

echo "[$(date)] Starting test GPU job on $(hostname)"

python /projects/wli19@xsede.org/alpine_alsf_virtual_stain/tests/test_gpu.py

echo "[$(date)] Test GPU job finished"
