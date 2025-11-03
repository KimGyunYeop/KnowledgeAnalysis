#/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

python train.py \
    --data_path data/bio_data_50000_42.json \
    --template_path bio_templates.json \
    --num_steps 50000 \
    --eval_steps 500 \
    --use_pretrained \
    --model_name meta-llama/Llama-3.2-1B \
    --run_name "pt" \
    --gpu 1

for n_layer in 8; do
    # python train.py \
    #     --data_path data/bio_data_50000_42.json \
    #     --template_path bio_templates.json \
    #     --n_layers $n_layer \
    #     --num_steps 200000 \
    #     --mix_augmented_prefixed_data \
    #     --prefix_mode "metadata" \
    #     --run_name "pt" \
    #     --gpu 1
        
    # python train.py \
    #     --data_path data/bio_data_30000_42.json \
    #     --template_path bio_templates.json \
    #     --n_layers $n_layer \
    #     --num_steps 200000 \
    #     --run_name "pt" \
    #     --gpu 0
done
