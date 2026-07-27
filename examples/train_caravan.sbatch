#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --job-name=lstm_caravan
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

# Run from the examples/ directory (repo checkout). Uses an absolute path rather than $SLURM_SUBMIT_DIR /
# $BASH_SOURCE, since SLURM executes a spooled copy of this script and neither reliably points back here.
cd /pfs/data6/home/ka/ka_iwu/ka_si3685/Hy2DL_eduardo/Hy2DL_new/examples || exit 1

source /home/ka/ka_iwu/ka_si3685/miniconda3/etc/profile.d/conda.sh
conda activate sb_working_311

python caravan_training.py
