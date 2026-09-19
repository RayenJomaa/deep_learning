#!/bin/bash

#SBATCH --job-name=setup-deeplearning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup-deeplearning-%j.out
#SBATCH --error=logs/setup-deeplearning-%j.err

set -e

MAMBA="$HOME/miniforge3/bin/mamba"
ENV_PATH="$HOME/miniforge3/envs/deeplearning"

echo "========================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "========================================"

echo
echo "=== 1. Suppression de l'ancien environnement ==="

if [ -d "$ENV_PATH" ]; then
    rm -rf "$ENV_PATH"
    echo "Ancien environnement supprimé."
else
    echo "Aucun ancien environnement à supprimer."
fi

echo
echo "=== 2. Création du nouvel environnement ==="

"$MAMBA" create -n deeplearning \
    --override-channels \
    --channel-priority flexible \
    -c pytorch \
    -c nvidia \
    -c defaults \
    python=3.10 \
    pytorch::pytorch=2.5.1 \
    pytorch::torchvision=0.20.1 \
    pytorch::torchaudio=2.5.1 \
    pytorch::pytorch-cuda=12.1 \
    -y

echo
echo "=== 3. Vérification du Python ==="

"$ENV_PATH/bin/python" --version
echo "Python utilisé : $ENV_PATH/bin/python"

echo
echo "=== 4. Vérification PyTorch / torchvision / CUDA ==="

"$ENV_PATH/bin/python" - <<'PY'
import torch
import torchvision
import torchaudio

print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("torchaudio  :", torchaudio.__version__)
print("CUDA PyTorch:", torch.version.cuda)
print("CUDA dispo  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
else:
    print("GPU         : aucun")
PY

echo
echo "========================================"
echo "INSTALLATION TERMINÉE AVEC SUCCÈS"
echo "========================================"
