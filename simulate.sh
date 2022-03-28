#! /bin/bash

echo 'clearing old outputs'
rm ./output/*
rm ./models/*
rm ./simulation/*

if [[ $1 != '' && $1 -gt 2 ]]
then
    nb=$1
else
    nb=20
fi

for i in `seq 0 $(($nb - 1))`; do
    bash ./run.sh 1
    wait
done