#/bin/bash
# checkpoint_path="results/plain_transformer/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8_mix_augmented_metadata/model_step_100000"
# checkpoint_path="results/gpt2/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_step_100000"
checkpoint_path="results/meta-llama/Llama-3.2-1B/pt_bio_data_50000_42_transformer_step50000_d512_d_hidden2048_n_layers8_n_heads8/model_step_50000"

python test_and_analysis.py \
    --data_path data/bio_data_50000_42_other_1764.json \
    --template_path bio_templates.json \
    --checkpoint_path "$checkpoint_path" \
    --run_name "test" \
    --gpu 0 \
    --add_augmented_prefix \
    --prefix_mode "metadata" \