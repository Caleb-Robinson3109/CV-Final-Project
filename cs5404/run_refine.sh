#!/bin/bash

NUM_CHUNKS=100

for ((i=0; i<NUM_CHUNKS; i++)); do
    # Format i as 3 digits (e.g., 000, 001, 099)
    idx=$(printf "%03d" $i)

    echo "Running chunk $idx ..."
    
    # Run the chunk and continue even if it fails
    python refine_dataset.py --data_path data/dataset_chunk_${idx}.pt --output_path data/refined${idx}.pt

    # Check exit code
    if [[ $? -ne 0 ]]; then
        echo "Chunk $idx FAILED."
    else
        echo "Chunk $idx completed successfully."
    fi

    echo "---------------------------------------------"
done

echo "All chunks attempted."
