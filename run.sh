#!/bin/bash

#SBATCH --account=tau
#SBATCH --job-name=SystMLXP
#SBATCH --output=%j_SystMLXP.log
#SBATCH -t 2-00:00:00             # max runtime days-hours:min:sec
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

#OLDSBATCH --nodelist titanic-3
#OLDSBATCH -N 1

# sbatch ... -C kepler  ... for Titan Blacks (kepler GPU architechture)
# sbatch ... -C pascal ... for GTX1080Ti (pascal GPU architechture)

date
hostname
pwd


WORKDIR="/home/tao/vestrade/workspace/SystML/SystMLXP"

sdocker -i  -v /data/titanic_3/users/vestrade/datawarehouse:~/datawarehouse \
			-v /data/titanic_3/users/vestrade/savings:/data/titanic_3/users/vestrade/savings \
			vestrade/systml:latest \
            [/bin/sh -c "cd ${WORKDIR}; python main.py $*"]


echo "DONE"
date

exit 0
