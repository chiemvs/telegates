#!/usr/bin/bash
#SBATCH --job-name="sstcomp"
#SBATCH --nodelist=node011
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50gb
#SBATCH --time=240:00:00
#SBATCH --partition=defq
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

module load cdo

./run.sh $HOME/VENV ./composites.py sst_nhplus last20 True # This is compvar, anom, aggtime
