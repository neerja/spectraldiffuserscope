 #!/bin/bash

# Directory containing config files
CONFIG_DIR="Config_Files/sims"

# Loop through all files in the config directory
for config in "$CONFIG_DIR"/*.yaml; do
    if [ -f "$config" ]; then
        echo "Running configuration: $config"
        python recon.py --config_file_path "$config" --gpu_index 0
        echo "Completed: $config"
        echo "----------------------------------------"
    fi
done

echo "All configurations completed!"