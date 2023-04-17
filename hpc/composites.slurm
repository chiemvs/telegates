#!/usr/bin/bash
#SBATCH --job-name="compmidlat"
#SBATCH --nodelist=node241
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50gb
#SBATCH --time=240:00:00
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

module load cdo

./run.sh $HOME/VENV ./composites.py t2m_midlats last40 True True # This is compvar, anom, aggtime, rollthresh
