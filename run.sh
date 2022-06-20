#!/bin/bash

# if [[ $1 != '' && $1 -gt 0 ]]
# then
#     nb=$1
# else
#     nb=2
# fi

nb=$1
seed=$2

if [[ $3 != '' && $3 -gt 0 ]]
then
    port=$3
else
    port=8080
fi

tmp="simulation/seed_$seed"
mkdir $tmp
dir="$tmp/clients_$nb"
mkdir $dir

for i in `seq 0 $(($nb - 1))`; do
    python tfclient.py -c $i -s $seed -p $port -d $dir > /dev/null &
done

python tfserver.py -c $(($i+1)) -s $seed -p $port -d $dir > /dev/null &

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait