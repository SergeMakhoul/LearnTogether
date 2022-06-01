#! /bin/bash

if [[ $1 != '' ]]
then
    nb=$1
else
    nb=10
fi

if [[ $2 != '' ]]
then
    nb_clients=$2
else
    nb_clients=3
fi

fs=`ls ./simulation_history | wc -l`
if (($fs != 0)); then
    sim_num=$((`ls ./archive | wc -l`+1))
    mkdir ./archive/simulation_$sim_num
    mv ./simulation_history/* ./archive/simulation_$sim_num
fi;

for i in `seq 1 $nb`; do
    echo "[INFO] Simulate | Creating new dataset"
    seed=$((1000+$i))
    python dataset.py -c $nb_clients -s $seed

    seed_path=./simulation_history/seed_$seed
    mkdir $seed_path

    for j in `seq 1 $nb_clients`; do
        clients_path=$seed_path/clients_$j
        mkdir $clients_path

        rm ./output/* >/dev/null 2>&1
        rm ./models/* >/dev/null 2>&1

        echo "[INFO] Simulate | Running simulation $j"
        bash ./run.sh $j

        mv ./simulation/* $clients_path
    done

    dataset_path=./dataset_history/seed_$seed
    mkdir $dataset_path
    mv ./dataset/* $dataset_path
done