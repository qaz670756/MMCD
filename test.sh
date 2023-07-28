commands=(
        "python main.py --config=CD_siamunet_diff --eval --save_img --ckpt_version 7"
        "python main.py --config=CD_siamunet_conc --eval --save_img --ckpt_version 17"
        "python main.py --config=CD_ifn_deep3d --eval --save_img --ckpt_version 1"
        "python main.py --config=CD_ori_range_snunet --eval --save_img --ckpt_version 15"
        "python main.py --config=CD_changeformer_emb128 --eval --save_img --ckpt_version 7 --device 1"
        "python main.py --config=CD_p2vnet --eval --save_img --ckpt_version 7"
        "python main.py --config=CD_ori_range_nofilter --eval --save_img --ckpt_version 1")

#!/bin/bash

# Run the commands in sequence and log their output

for command in "${commands[@]}"; do
    echo "Running command: $command"
    $command 2>&1 | tee "logs/${command// /_}_output.log"
done



