#!/bin/bash

#SBATCH --account=tau
#SBATCH --job-name=SystMLXP
#SBATCH --output=%j_SystMLXP.log
#SBATCH -t 8:00:00             # max runtime hours:min:sec
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

#OLDSBATCH --nodelist titanic-3
#OLDSBATCH -N 1

# sbatch ... -C kepler  ... for Titan Blacks (kepler GPU architechture)
# sbatch ... -C pascal ... for GTX1080Ti (pascal GPU architechture)

date;hostname;pwd

# Manually add miniconda to PATH. Don't know why the .basrc is not correctly sourced
export PATH="/home/tao/${USER}/miniconda3/bin:$PATH"
# source /home/tao/${USER}/miniconda3/bin/activate default

WORKDIR="/home/tao/${USER}/workspace/SystML/SystMLXP"
cd $WORKDIR

python main.py $@

echo "DONE"
date

exit 0
