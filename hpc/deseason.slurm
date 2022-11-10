#!/usr/bin/bash
#SBATCH --job-name="de40v"
#SBATCH --nodelist=node002
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50gb
#SBATCH --time=240:00:00
#SBATCH --partition=defq
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

./run.sh $HOME/VENV ./deseason.py v300_nhnorm
