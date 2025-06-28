#!/bin/bash

# Copy conf directory to gpu and cpu paths
for generate_dir in "auto_openai_gpu_node" "auto_openai_master_node"; do
    conf_dest="${generate_dir}/conf"
    if [ -d "conf" ]; then
        # if [ -d "$conf_dest" ]; then
        #     rm -rf "$conf_dest"
        # fi
        cp -rf conf "$conf_dest"
    fi
done
