#!/bin/bash

if [[ $1 != '' && $1 -gt 0 ]]
then
    nb=$1
else
    nb=2
fi

for i in `seq 0 $(($nb - 1))`; do
    echo "[INFO] Run | Starting client $i"
    # python tfclient.py -c $i > output/client$i.txt &
    python tfclient.py -c $i > /dev/null &
done

echo "[INFO] Run | Starting server"
python tfserver.py > /dev/null &

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait