commands=(
        "python main.py --config=CD_siamunet_diff --eval --save_img --ckpt_version 16 --device 1"
        "python main.py --config=CD_siamunet_conc --eval --save_img --ckpt_version 40 --device 1")

# !/bin/bash

# Run the commands in sequence and log their output

for command in "${commands[@]}"; do
    echo "Running command: $command"
    $command 2>&1 | tee "logs/${command// /_}_output.log"
done
