#!/bin/bash
DATASET=hot3d 
N_GPUS=$(python -c 'print(__import__("torch").cuda.device_count())')
N_CPUS=$(python -c 'print(__import__("multiprocessing").cpu_count())')
let BATCH_SIZE=16*$N_GPUS
torchrun \
    --nproc_per_node=$N_GPUS \
    --nnodes=1 \
    --node_rank='0' \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    main.py --batch_size $BATCH_SIZE \
    --num_workers $N_CPUS \
    --log_dir logs_$DATASET \
    --chkpt_dir chkpt_$DATASET \
    --dataset $DATASET \
    --warmup_step 1000 \
    --is_parallel \
    --n_epochs 200
 
