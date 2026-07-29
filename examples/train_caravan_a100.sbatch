#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=1-12:00:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --job-name=lstm_caravan_a100
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

# A100 variant of train_caravan.sbatch: gpu_a100_il is far less contended than gpu_h100 (fewer jobs
# specifically need H100-class compute), so this queues faster. At epochs=10 (see caravan.yml), a run should
# finish in well under gpu_a100_il's 2-day partition cap - no job chaining needed like the 30-epoch/gpu_h100
# case required.
#
# Run from the examples/ directory (repo checkout). Uses an absolute path rather than $SLURM_SUBMIT_DIR /
# $BASH_SOURCE, since SLURM executes a spooled copy of this script and neither reliably points back here.
cd /pfs/data6/home/ka/ka_iwu/ka_si3685/Hy2DL_eduardo/Hy2DL_new/examples || exit 1

source /home/ka/ka_iwu/ka_si3685/miniconda3/etc/profile.d/conda.sh
conda activate sb_working_311

python caravan_training.py
