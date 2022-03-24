#! /bin/bash

for i in `seq 0 50`; do
    bash ./run.sh 2
    wait
done