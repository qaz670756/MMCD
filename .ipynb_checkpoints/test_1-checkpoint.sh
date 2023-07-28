commands=(
        "python main.py --config=CD_ori_range_snunet --eval --save_img --ckpt_version 27 --device 0"
        "python main.py --config=CD_p2vnet --eval --save_img --ckpt_version 22 --device 1"
        "python main.py --config=CD_ori_range_nofilter --eval --save_img --ckpt_version 17 --device 1")

#!/bin/bash

# Run the commands in sequence and log their output

for command in "${commands[@]}"; do
    echo "Running command: $command"
    $command 2>&1 | tee "logs/${command// /_}_output.log"
done