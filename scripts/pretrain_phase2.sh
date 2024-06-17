#! /bin/bash

export LOGLEVEL=INFO

export OMP_NUM_THREADS=1

GPUS_PER_NODE=8

DIST_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes 1 --node_rank 0 --rdzv_conf timeout=5400"
echo $DIST_ARGS

WORKSPACE='./LMtrainer'
cd $WORKSPACE

lr=1e-4
adam_beta1=0.9
adam_beta2=0.95
weight_decay=0.1
pretrained_model=./model/
data_dir=$1
seed=2032
OUTPUT_DIR='/cache/models2'
gradient_accumulation_steps=1
MP_SIZE=1
export MPU_DIR=./

mv /cache/models/checkpoint-35303/scheduler.pt /cache/models/checkpoint-35303/scheduler.pt.bak

torchrun $DIST_ARGS \
                pretrain.py \
		--resume_from_checkpoint /cache/models/checkpoint-35303 \
                --resume_new_data True \
                --output_dir $OUTPUT_DIR \
                --model_name_or_path ${pretrained_model} \
                --overwrite_output_dir \
                --validation_split_percentage 0.00004 \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 1 \
                --do_train \
                --seed $seed \
                --logging_strategy steps \
                --save_strategy steps \
                --save_steps 1000 \
                --save_total_limit 1000 \
                --gradient_accumulation_steps ${gradient_accumulation_steps} \
                --preprocessing_num_workers 64 \
                --model_max_length 2048 \
                --output_dir $OUTPUT_DIR \
                --logging_first_step True \
                --num_train_epochs 1 \
                --fp16 True \
                --report_to tensorboard \
                --logging_dir $OUTPUT_DIR/tensorboard \
                --logging_steps 1 \
                --evaluation_strategy steps \
                --eval_steps 100000000 \
                --fp16_full_eval \
                --gradient_checkpointing True \
                --flash_attention True \
                --model_parallel_size $MP_SIZE \
                --lr_scheduler_type cosine \
                --learning_rate ${lr} \
                --adam_beta1 ${adam_beta1} \
                --adam_beta2 ${adam_beta2} \
                --weight_decay ${weight_decay} \
                --warmup_steps 1000 \
                --ddp_timeout 5400 \
                --dataset_dir $data_dir \
                # --data_cache_dir $data_dir \
                # --read_cached \

python ../compress_pythia.py /cache/models2/checkpoint-1000 /cache/models/checkpoint-35000 --output /cache/models2/checkpoint-1000 --recon
python ../compress_pythia.py /cache/models2/checkpoint-2000 /cache/models2/checkpoint-1000 --output /cache/models2/checkpoint-2000 --recon
python ../compress_pythia.py /cache/models2/checkpoint-3000 /cache/models2/checkpoint-2000 --output /cache/models2/checkpoint-3000 --recon
python ../compress_pythia.py /cache/models2/checkpoint-4000 /cache/models2/checkpoint-3000 --output /cache/models2/checkpoint-4000 --recon
python ../compress_pythia.py /cache/models2/checkpoint-5000 /cache/models2/checkpoint-4000 --output /cache/models2/checkpoint-5000 --recon

for iter in {5000..35000..5000}
do
	torchrun $DIST_ARGS \
                pretrain.py \
                --output_dir $OUTPUT_DIR \
                --model_name_or_path ${pretrained_model} \
                --overwrite_output_dir \
                --validation_split_percentage 0.00004 \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 1 \
                --do_train \
                --resume_from_checkpoint /cache/models2/checkpoint-$iter \
                --seed $seed \
                --logging_strategy steps \
                --save_strategy steps \
                --save_steps 1000 \
                --save_total_limit 1000 \
                --gradient_accumulation_steps ${gradient_accumulation_steps} \
                --preprocessing_num_workers 64 \
                --model_max_length 2048 \
                --output_dir $OUTPUT_DIR \
                --logging_first_step True \
                --num_train_epochs 1 \
                --fp16 True \
                --report_to tensorboard \
                --logging_dir $OUTPUT_DIR/tensorboard \
                --logging_steps 1 \
                --evaluation_strategy steps \
                --eval_steps 100000000 \
                --fp16_full_eval \
                --gradient_checkpointing True \
                --flash_attention True \
                --model_parallel_size $MP_SIZE \
                --lr_scheduler_type cosine \
                --learning_rate ${lr} \
                --adam_beta1 ${adam_beta1} \
                --adam_beta2 ${adam_beta2} \
                --weight_decay ${weight_decay} \
                --warmup_steps 1000 \
                --ddp_timeout 5400 \
                --dataset_dir $data_dir \
                # --data_cache_dir $data_dir \
                # --read_cached \

        next_iter=$[iter+5000]
        last_iter=$[iter+4000]
        last_iter2=$[iter+3000]
        last_iter3=$[iter+2000]
        last_iter4=$[iter+1000]
        python ../compress_pythia.py /cache/models2/checkpoint-$last_iter4 /cache/models2/checkpoint-$iter --output /cache/models2/checkpoint-$last_iter4 --recon
        python ../compress_pythia.py /cache/models2/checkpoint-$last_iter3 /cache/models2/checkpoint-$last_iter4 --output /cache/models2/checkpoint-$last_iter3 --recon
        python ../compress_pythia.py /cache/models2/checkpoint-$last_iter2 /cache/models2/checkpoint-$last_iter3 --output /cache/models2/checkpoint-$last_iter2 --recon
        python ../compress_pythia.py /cache/models2/checkpoint-$last_iter /cache/models2/checkpoint-$last_iter2 --output /cache/models2/checkpoint-$last_iter --recon
        python ../compress_pythia.py /cache/models2/checkpoint-$next_iter /cache/models2/checkpoint-$last_iter --output /cache/models2/checkpoint-$next_iter --recon
done


