#!/usr/bin/bash
#SBATCH --job-name="olrcomp"
#SBATCH --nodelist=node240
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50gb
#SBATCH --time=240:00:00
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

module load cdo

./run.sh $HOME/VENV ./composites.py olr_tropics last20 True # This is compvar, anom, aggtime
