#!/bin/bash

# Find all directories matching the GPU types pattern
gpu_types=("CPU" "NV-A100-80G" "NV-4090" "EF-S60")
dests=("auto_openai_master")

# Use find command to get all matching directories
for gpu_type in "${gpu_types[@]}"; do
    while read -r dir; do
        if [ -d "$dir" ]; then
            dests+=("$dir")
        fi
    done < <(find . -maxdepth 1 -type d -name "auto_openai_${gpu_type}_*card" | sed 's|^\./||')
done

# Copy conf directory to gpu and cpu paths
for generate_dir in "${dests[@]}"; do
    conf_dest="${generate_dir}/"
    if [ -d "conf" ]; then
        # if [ -d "$conf_dest" ]; then
        #     rm -rf "$conf_dest"
        # fi
        cp -rf conf "$conf_dest"
    fi
done
