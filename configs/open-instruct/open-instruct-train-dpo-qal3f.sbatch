#!/bin/bash -x

#SBATCH --output=open-instruct-train-dpo-qal3f-%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=47:59:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:4
#SBATCH --account=pr_95_tandon_advanced
#SBATCH --constraint="a100|h100"
#SBATCH --reservation=bf996
#SBATCH --job-name=open-instruct-train-dpo-qal3f
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

#command variable

# Training setup
GPUS_PER_NODE=4
# so processes know who to talk to
MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
MASTER_PORT=47801
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
RDZV_CONF="\"rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT rdzv_backend=c10d\""

MY_COMMAND="accelerate launch \
            --num_machines $NNODES \
            --num_processes $WORLD_SIZE \
            --gpu_ids='all' \
            --mixed_precision=bf16 \
            --dynamo_backend="no" \
            --main_process_ip "$MASTER_ADDR" \
            --main_process_port $MASTER_PORT \
            --machine_rank \$SLURM_PROCID \
            --rdzv_conf $RDZV_CONF \
            --use_deepspeed \
            --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
            open_instruct/dpo_tune.py \
            configs/train_configs/llama3/dpo-qwen2572b-athene70b-jdg-Llama3-Factuality-AllenAI-fmt-dpo-8b.yaml"

#run command

#CUDA_VISIBLE_DEVICES=\"\";

srun \
    /bin/bash /scratch/bf996/open-instruct/scripts/run-singularity.bash \
    /bin/bash -c \
    "cd /scratch/bf996/open-instruct; rm /home/bf996/.cache/huggingface/accelerate/default_config.yaml; $MY_COMMAND "