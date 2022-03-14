#!/bin/bash

echo "Starting server"
python tfserver.py > output/server.txt &
sleep 5

if [[ $1 != '' && $1 > 10 ]]
then
    nb=$(($1-1))
else
    nb=9
fi

for i in `seq 0 ${nb}`; do
    echo "Starting client $i"
    python tfclient.py > output/client$i.txt &
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait