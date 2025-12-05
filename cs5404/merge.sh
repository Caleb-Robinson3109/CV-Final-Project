#!/bin/bash

NUM_CHUNKS=100
NUM_MERGES=$((NUM_CHUNKS - 5))


for ((i=0; i<NUM_MERGES; i++)); do
    idx1=$(printf "%03d" $i)
    idx2=$(printf "%03d" $((i+5)))

    echo "Running merge merged${idx1}.pt + refined${idx2}.pt → merged${idx2}.pt ..."

    python merge.py data/merged${idx1}.pt data/refined${idx2}.pt data/merged${idx2}.pt 

    if [[ $? -ne 0 ]]; then
        exit 1
    else
        echo "merge $idx completed successfully."
    fi

    rm data/merged${idx1}.pt
    echo "---------------------------------------------"
done