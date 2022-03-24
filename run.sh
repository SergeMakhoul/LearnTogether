#!/bin/bash

# echo "Running benchmark simple linear regression code"
# python lr.py > output/benchmark.txt &
# wait

echo "Starting server"
python tfserver.py &
sleep 5

if [[ $1 != '' && $1 -gt 0 ]]
then
    nb=$1
else
    nb=2
fi

for i in `seq 0 $(($nb - 1))`; do
    echo "Starting client $i"
    python tfclient.py $i > output/client$i.txt &
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait