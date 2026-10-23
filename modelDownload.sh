#!/bin/bash
#SBATCH --job-name=HFdownload
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00
#SBATCH --partition=gpu_mig
#SBATCH --gpus-per-node=1

set -e

source $HOME/projectVenv/bin/activate
export HF_HOME=$HOME/ULLMPROJ/hf_cache
export HF_HOME=$HOME/ULLMPROJ/hf_cache

#hf download openai/gpt-oss-20b \
#    --exclude "original/*" \
#    --exclude "metal/*" \
#    --local-dir $HOME/ULLMPROJ/models/gpt-oss

python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "openai/gpt-oss-20b",
    allow_patterns=["*.safetensors", "*.json", "*.jinja"],
    ignore_patterns=["original/*", "metal/*"],
    local_dir=os.path.expanduser("~/ULLMPROJ/models/gpt-oss"),
)
print("Download finished")
EOF
grep -q gpt_oss $HOME/ULLMPROJ/models/gpt-oss/config.json

deactivate