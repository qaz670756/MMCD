commands=("python main.py --config=CD_siamunet_diff --device 1"
          "python main.py --config=CD_siamunet_conc --device 1")

#!/bin/bash

# Run the commands in sequence and log their output

for command in "${commands[@]}"; do
    echo "Running command: $command"
    $command 2>&1 | tee "logs/train/${command// /_}_output.log"
done



