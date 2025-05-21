# Training for finetune TiTok-L32
# all in one stage

# set python path
export PYTHONPATH=$PYTHONPATH:/root/qingfeli/titok_finetune_version5
export TORCH_DISTRIBUTED_DEBUG=INFO

WANDB_MODE=offline accelerate launch --multi_gpu --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=100.71.2.11 --main_process_port=29501 --same_network scripts/train_titok.py config=configs/training/LFQ-TiTok/finetune_titok_l32.yaml \
    experiment.project="titok_l_32_finetune_LFQ_TiTok_L1_Loss" \
    experiment.name="titok_l_32_finetune_LFQ_TiTok_L1_Loss_run1" \
    experiment.output_dir="titok_l_32_finetune_LFQ_TiTok_L1_Loss_run1" \
    training.per_gpu_batch_size=32 