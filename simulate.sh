#!/bin/bash
#---------------------------------------------------------------------------
#SBATCH --job-name=federated_learning
#SBATCH --ntasks=20
#---------------------------------------------------------------------------

source ${HOME}/.bashrc
source activate flwr-fl

nb_sim=$1
nb_clients=$2

if [[ $3 != '' && $3 -gt 0 ]]
then
    starting_seed=$3
else
    starting_seed=1000
fi

if [[ $4 != '' && $4 -gt 0 ]]
then
    base_port=$4
else
    base_port=9000
fi

for sim in `seq 1 $nb_sim`; do
    port=$base_port
    seed=$(($starting_seed+$sim))
    
    for clients in `seq 1 20`; do
        bash ./run.sh $clients $seed $port
    done
    
    for clients in `seq 21 +2 49`; do
        bash ./run.sh $clients $seed $port
    done
    
    for clients in `seq 50 +10 100`; do
        bash ./run.sh $clients $seed $port
    done
done