#/bin/bash
checkpoint_paths=(
    # "results/plain_transformer/pt_bio_data_30000_42_transformer_step100000_d512_d_hidden2048_n_layers8_n_heads8/model_first_all_correct_step_84400"
    # "results/plain_transformer/pt_bio_data_30000_42_transformer_step100000_d512_d_hidden2048_n_layers8_n_heads8/model_step_100000"
    # "results/plain_transformer/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_step_200000"
    "results/plain_transformer/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_first_all_correct_step_171300"
    # "results/plain_transformer/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_first_all_correct_step_185100"
)
data_path="data/bio_data_50000_42_other_1764.json"
split_key="first_attr_loss" # ["attr_loss", "first_attr_loss", "attr_confidence", "first_attr_confidence"]
gpu=0

for checkpoint_path in "${checkpoint_paths[@]}"; do
    python -u finetune_splited_attc_loss_data.py \
        --data_path "$data_path" \
        --template_path bio_templates.json \
        --checkpoint_path "$checkpoint_path" \
        --num_steps 20000 \
        --split_key "$split_key" \
        --run_name "ft" \
        --gpu $gpu

    python -u finetune_splited_attc_loss_data.py \
        --data_path "$data_path" \
        --template_path bio_templates.json \
        --checkpoint_path "$checkpoint_path" \
        --add_pretrained_data \
        --p_pretrained_data 0.5 \
        --num_steps 20000 \
        --split_key "$split_key" \
        --run_name "ft" \
        --gpu $gpu
done