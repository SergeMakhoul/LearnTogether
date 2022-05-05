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

echo '[INFO] Simulate | Clearing old outputs and models'
rm ./output/* >/dev/null 2>&1
rm ./models/* >/dev/null 2>&1

# echo '[INFO] Simulate | Clearing old simulations'
# rm ./simulation/* >/dev/null 2>&1
# rm ./simulation_server/* >/dev/null 2>&1

# Getting the number of simulations already done
num_simulations=`ls ./simulation_history | wc -l`
# Creating the string for the directory extension to use
sim=simulation$(($num_simulations+1))

echo '[INFO] Simulate | Moving client simulation into history'
mkdir ./simulation_history/$sim
mv ./simulation/* ./simulation_history/$sim

echo '[INFO] Simulate | Moving server simulation into history'
mkdir ./simulation_server_history/$sim
mv ./simulation_server/* ./simulation_server_history/$sim

for i in `seq 1 $nb`; do
    # echo '[INFO] Simulate | Clearing old dataset'
    # rm ./dataset/*
    echo '[INFO] Simulate | Moving old dataset'
    mkdir ./dataset_history/$sim
    mv ./dataset/* ./dataset_history/$sim

    echo '[INFO] Simulate | Creating new dataset'
    python dataset.py $nb_clients

    for f in simulation/*; do if [[ "$f" =~ client[[:digit:]]$ ]]; then mv "$f" "$f-$i"; fi; done

    echo '[INFO] Simulate | Running simulation'
    # for j in `seq 1 10`; do
    #     bash ./run.sh $nb_clients
    # done
    bash ./run.sh $nb_clients
done