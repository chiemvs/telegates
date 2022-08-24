#!/usr/bin/bash
#SBATCH --job-name="desstream"
#SBATCH --nodelist=node241
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30gb
#SBATCH --time=240:00:00
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

./run.sh $HOME/VENV ./deseason.py
