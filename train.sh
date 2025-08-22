#/bin/bash


# python train.py \
#     --data_path data/bio_data_50000_42.json \
#     --template_path bio_templates.json \
#     --n_layers 8 \
#     --num_steps 200000 \
#     --use_pretrained \
#     --run_name "pt" \
#     --gpu 0

for n_layer in 8; do
    python train.py \
        --data_path data/bio_data_50000_42.json \
        --template_path bio_templates.json \
        --n_layers $n_layer \
        --num_steps 200000 \
        --mix_augmented_prefixed_data \
        --prefix_mode "metadata" \
        --run_name "pt" \
        --gpu 1
        
    # python train.py \
    #     --data_path data/bio_data_30000_42.json \
    #     --template_path bio_templates.json \
    #     --n_layers $n_layer \
    #     --num_steps 200000 \
    #     --run_name "pt" \
    #     --gpu 0
done
