#! /bin/bash

if [[ $1 != '' && $1 -gt 0 ]]
then
    nb=$1
else
    nb=20
fi

for i in `seq 0 ${nb}`; do
    bash ./run.sh 2
    wait
done