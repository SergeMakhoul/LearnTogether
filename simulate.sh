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

echo 'clearing old outputs...'
rm ./output/* >/dev/null 2>&1
rm ./models/* >/dev/null 2>&1
rm ./simulation/* >/dev/null 2>&1
rm ./simulation_server/* >/dev/null 2>&1

for i in `seq 1 $nb`; do
    echo 'clearing old dataset...'
    rm ./dataset/*
    echo 'creating new dataset...'
    python dataset.py $nb_clients
    wait

    for f in simulation/*; do if [[ "$f" =~ client[[:digit:]]$ ]]; then mv "$f" "$f-$i"; fi; done

    echo 'running simulation...'
    # for j in `seq 1 10`; do
    #     bash ./run.sh $nb_clients
    #     wait
    # done
    bash ./run.sh $nb_clients
    wait
done