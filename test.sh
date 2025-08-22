#/bin/bash
checkpoint_path="/home/nlplab/ssd2/gyop/KnowledgeAnalysis/results/gpt2/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_step_80000"

python test_and_analysis.py \
    --data_path data/bio_data_50000_42_other_1764.json \
    --template_path bio_templates.json \
    --checkpoint_path "$checkpoint_path" \
    --run_name "test" \
    --gpu 1 \
    --add_augmented_prefix \
    --prefix_mode "metadata" \