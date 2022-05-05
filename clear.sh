#! /bin/bash

# This file purely just empties all directories

rm -r \
    ./dataset/* \
    ./dataset_history/* \
    ./models/* \
    ./output/* \
    ./simulation/* \
    ./simulation_history/* \
    ./simulation_server/* \
    ./simulation_server_history/* \
    ./tests_average/* \
>/dev/null 2>&1