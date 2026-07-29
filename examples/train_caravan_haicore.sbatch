#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=1-12:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:full:1
#SBATCH --job-name=lstm_caravan_haicore
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

# HAICORE (haicore.scc.kit.edu) variant of train_caravan.sbatch. HAICORE's queues (see docs) are GPU4 nodes
# (4 full GPUs each) under a shared access policy - "normal" is the standard queue, max 72h walltime, same
# cap as uc3's gpu_h100. At epochs=10 (see caravan.yml) a run should fit well within that in a single job.
# gres uses "gpu:full:1" (not "gpu:1") to match HAICORE's GRES naming for a full (non-MIG) GPU.
#
# Run from the examples/ directory (repo checkout). Uses $HOME rather than $SLURM_SUBMIT_DIR / $BASH_SOURCE,
# since SLURM executes a spooled copy of this script and neither reliably points back here. Unlike uc3, the
# haicore clone lives directly at $HOME/Hy2DL_new (no Hy2DL_eduardo wrapper folder).
cd "$HOME/Hy2DL_new/examples" || exit 1

source "$HOME/Hy2DL_working/miniconda3/etc/profile.d/conda.sh"
conda activate sb_working_311

python caravan_training.py