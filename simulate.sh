#! /bin/bash

nb_sim=$1
nb_clients=$2

fs=`ls ./simulation_history | wc -l`

if (($fs != 0)); then
    sim_num=$((`ls ./archive | wc -l`+1))
    mkdir ./archive/simulation_$sim_num
    mv ./simulation_history/* ./archive/simulation_$sim_num
fi;


for sim in `seq 1 $nb_sim`; do
    port=9000
    
    seed=$((1000+$sim))
    python dataset.py -c $nb_clients -s $seed
    
    for clients in `seq 1 $nb_clients`; do
        bash ./run.sh $clients $seed $port &
        port=$(($port+1))
    done
    
    wait
done