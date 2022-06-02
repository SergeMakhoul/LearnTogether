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

for i in `seq 0 $(($nb - 1))`; do
    tmp="simulation/seed_$seed"
    mkdir $tmp
    dir="$tmp/clients_$nb"
    mkdir $dir
    python tfclient.py -c $i -s $seed -p $port -d $dir > /dev/null &
done

python tfserver.py -s $seed -p $port -d $dir > /dev/null &

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait