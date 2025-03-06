#!/bin/bash

# Iterate through numbers 0 to 23, padded with zeros to 4 digits
for i in $(seq -f "%04g" 1 23)  # -f "%04g" ensures zero-padding to 4 digits
do
    # Construct the YAML config file path with proper zero-padding
    config_file="./Config_Files/dryingbeadsdrop_noinit/dryingbeadsdrop_2025-02-27_meas${i}.yaml"
    # modify the meas and the sample name
    sed -i "s|meas0000|meas${i}|g" "$config_file"
    sed -i "s|run_name: drying_beads_drop|run_name: drying_beads_drop_noinit_${i#0}|g" "$config_file"
    sed -i "s|Results/Drying_Beads_Drop_initializewithprevUonly|Results/Drying_Beads_Drop_noinit|g" "$config_file"
    
    # Check if the file exists before running the Python command
    if [[ -f "$config_file" ]]; then
        echo "Running recon.py with $config_file"
        python recon.py --gpu_index 3 --config_file_path "$config_file"
    else
        echo "File not found: $config_file"
    fi
done
