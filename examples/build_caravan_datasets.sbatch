#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --job-name=build_caravan_datasets
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

# Dataset creation is CPU/IO-bound (no model training here), so this runs on the plain cpu partition -
# no GPU needed, and cpu has far more nodes (80) than any GPU partition, so it queues much faster.
cd /pfs/data6/home/ka/ka_iwu/ka_si3685/Hy2DL_eduardo/Hy2DL_new/examples || exit 1

source /home/ka/ka_iwu/ka_si3685/miniconda3/etc/profile.d/conda.sh
conda activate sb_working_311

python build_caravan_datasets.py
