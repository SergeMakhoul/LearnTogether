#!/bin/bash
#---------------------------------------------------------------------------
#SBATCH --job-name=msesim
#SBATCH --ntasks=20
#---------------------------------------------------------------------------

source ${HOME}/.bashrc
source activate flwr-fl

num_players=100
num_seeds=300

# rm -r dataset/*

for i in `seq 1 +50 $num_seeds`; do
    for j in `seq 0 49`; do
        python dataset.py -s $((1700+$i+$j)) -c $num_players &
    done;
    wait
done

echo 'running mlr'
time python mlr.py -n $num_players