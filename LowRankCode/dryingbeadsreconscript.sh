#!/bin/bash

# Iterate through numbers 0 to 23, padded with zeros to 4 digits
for i in $(seq -f "%04g" 19 23)  # -f "%04g" ensures zero-padding to 4 digits
do
    # Construct the YAML config file path with proper zero-padding
    config_file="./Config_Files/dryingbeadsdrop_2024-02-01/config_dryingbeadsdrop_${i}.yaml"

    # Check if the file exists before running the Python command
    if [[ -f "$config_file" ]]; then
        echo "Running recon.py with $config_file"
        python recon.py --gpu_index 0 --config_file_path "$config_file"
    else
        echo "File not found: $config_file"
    fi
done
