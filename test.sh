#/bin/bash

python plain_transformer_test_and_analysis.py \
    --data_path data/bio_data_50000_42_other_1764.json \
    --template_path bio_templates.json \
    --checkpoint_path results/plain_transformer/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_first_all_correct_step_171300 \
    --run_name "test" \
    --gpu 0