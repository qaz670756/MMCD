commands=("python main.py --config=CD_changeformer_emb128 --device 3"
        "python main.py --config=CD_p2vnet --device 0"
        "python main.py --config=CD_ori_range_nofilter --device 3")

# !/bin/bash

# Run the commands in sequence and log their output

for command in "${commands[@]}"; do
    echo "Running command: $command"
    $command 2>&1 | tee "logs/train/${command// /_}_output.log"
done



