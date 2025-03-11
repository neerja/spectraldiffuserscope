#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Base config file - now using full path
CONFIG_FILE="$SCRIPT_DIR/dryingbeadsdrop_2025-02-27.yaml"
RESULTS_DIR="/home/neerja/CODE/SpectralDiffuserScopeGitRepo/PlottingCode/Results/Drying_Beads_Drop_initializewithprevUonly"

# Loop from 1 to 23
for i in $(seq -f "%04g" 1 23); do
    echo "Processing measurement $i"
    
    # Create configs subfolder if it doesn't exist
    CONFIG_SUBFOLDER="$SCRIPT_DIR/drying_beads_drop_2025-02-27"
    mkdir -p "$CONFIG_SUBFOLDER"

    # Create new config filename with full path in subfolder
    new_config="$CONFIG_SUBFOLDER/${CONFIG_FILE##*/}"
    new_config="${new_config%.*}_meas${i}.yaml"
    
    # Copy original config
    cp "$CONFIG_FILE" "$new_config"
    
    # Update measurement folder
    sed -i "s|meas0000|meas${i}|g" "$new_config"
    
    # Update run name to include measurement number
    sed -i "s|run_name: drying_beads_drop|run_name: drying_beads_drop_${i#0}|g" "$new_config"
    
    # Find most recent pickle file for previous measurement
    prev_num=$(printf "%04d" $((10#$i - 1)))
    prev_pickle=$(find "$RESULTS_DIR" -name "drying_beads_drop_${prev_num#0}*.pkl" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
    # print prev_pickle
    echo "Using previous recon file: $prev_pickle"

    if [ ! -z "$prev_pickle" ] && [ -f "$prev_pickle" ]; then
        # Check if recon_init_path exists in the file
        if grep -q "recon_init_path:" "$new_config"; then
            # Update existing recon_init_path
            sed -i "s|recon_init_path:.*|recon_init_path: $prev_pickle|g" "$new_config"
        else
            # Add recon_init_path under the reconstruction section
            sed -i '/reconstruction:/a\  recon_init_path: '"$prev_pickle" "$new_config"
        fi
    else
        echo "No previous pickle file found for measurement $prev_num, skipping recon_init_path update"
    fi
    
    echo "Created config file: $new_config"

    # Run reconstruction with the new config file
    echo "Running reconstruction for measurement $i"
    python recon.py --gpu_index 0 --config_file_path "$new_config"

done