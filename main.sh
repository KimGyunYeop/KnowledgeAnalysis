#/bin/bash

checkpoint_paths=(
    # "results/plain_transformer/pt_bio_data_30000_42_transformer_step100000_d512_d_hidden2048_n_layers8_n_heads8/model_first_all_correct_step_84400"
    # "results/plain_transformer/pt_bio_data_30000_42_transformer_step100000_d512_d_hidden2048_n_layers8_n_heads8/model_step_100000"
    # "results/plain_transformer/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_step_200000"
    # "results/plain_transformer/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_first_all_correct_step_171300"
    # "results/plain_transformer/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_first_all_correct_step_185100"
    # "results/gpt2/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_step_100000"
    # "results/plain_transformer/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8_mix_augmented_metadata/model_step_100000"
    # "/home/nlplab/ssd2/gyop/KnowledgeAnalysis/results/gpt2/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_step_80000"
    # "results/meta-llama/Llama-3.2-1B/pt_bio_data_50000_42_transformer_step50000_d512_d_hidden2048_n_layers8_n_heads8/model_step_50000"
    "results/gpt2/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8/model_step_150000"
    # "results/plain_transformer/pt_bio_data_50000_42_transformer_step200000_d512_d_hidden2048_n_layers8_n_heads8_mix_augmented_metadata/model_step_150000"
)
data_path="data/bio_data_50000_42_other_1764.json"
gpu=2

# parser.add_argument("--knowledge_deleting", action="store_true", default=False, help="Whether to delete knowledge from the model")
# parser.add_argument("--knowledge_deleting_loss_type", type=str, default="ppo", choices=["default", "ppo"], help="Method to delete knowledge from the model")
# parser.add_argument("--knowledge_deleting_finish_strategy", type=str, default="step", choices=["step", "max_value"], help="Method to delete knowledge from the model")
# parser.add_argument("--knowledge_deleting_steps", type=int, default=1000, help="Number of steps to delete knowledge")
# parser.add_argument("--knowledge_deleting_max_value", type=float, default=0.1, help="Maximum value to delete knowledge")
# parser.add_argument("--knowledge_deleting_use_kl", action="store_true", default=False, help="Whether to use KL divergence for knowledge deletion")

# parser.add_argument("--current_prob_based_loss", action="store_true", default=False, help="Whether to use current probability-based loss")

        
# parser.add_argument("--add_augmented_prefix", action="store_true", default=False, help="Whether to add augmented prefix")
# parser.add_argument("--prefix_mode", type=str, default="repeat", choices=["repeat", "metadata"], help="Prefix mode")

for checkpoint_path in "${checkpoint_paths[@]}"; do
    python -u main.py \
        --data_path "$data_path" \
        --template_path bio_templates.json \
        --checkpoint_path "$checkpoint_path" \
        --num_steps 10000 \
        --run_name "tmp_proposed" \
        --add_pretrained_data \
        --p_pretrained_data 0.5 \
        --gpu $gpu
        
    python -u main.py \
        --data_path "$data_path" \
        --template_path bio_templates.json \
        --checkpoint_path "$checkpoint_path" \
        --run_name "tmp_proposed" \
        --use_soft_prompt \
        --soft_prompt_type "add" \
        --add_pretrained_data \
        --p_pretrained_data 0.5 \
        --gpu $gpu
        

    python -u main.py \
        --data_path "$data_path" \
        --template_path bio_templates.json \
        --checkpoint_path "$checkpoint_path" \
        --num_steps 10000 \
        --run_name "tmp_proposed" \
        --use_kl \
        --add_pretrained_data \
        --p_pretrained_data 0.5 \
        --gpu $gpu

    python -u main.py \
        --data_path "$data_path" \
        --template_path bio_templates.json \
        --checkpoint_path "$checkpoint_path" \
        --run_name "tmp_proposed" \
        --use_soft_prompt \
        --soft_prompt_type "add" \
        --use_kl \
        --add_pretrained_data \
        --p_pretrained_data 0.5 \
        --gpu $gpu


    # python -u main.py \
    #     --data_path "$data_path" \
    #     --template_path bio_templates.json \
    #     --checkpoint_path "$checkpoint_path" \
    #     --num_steps 10000 \
    #     --run_name "proposed" \
    #     --add_augmented_prefix \
    #     --prefix_mode "metadata" \
    #     --gpu $gpu

    # python -u main.py \
    #     --data_path "$data_path" \
    #     --template_path bio_templates.json \
    #     --checkpoint_path "$checkpoint_path" \
    #     --num_steps 20000 \
    #     --knowledge_deleting \
    #     --knowledge_deleting_use_kl \
    #     --run_name "proposed" \
    #     --gpu $gpu

    # python -u main.py \
    #     --data_path "$data_path" \
    #     --template_path bio_templates.json \
    #     --checkpoint_path "$checkpoint_path" \
    #     --num_steps 20000 \
    #     --current_prob_based_loss \
    #     --run_name "proposed" \
    #     --gpu $gpu

    # python -u main.py \
    #     --data_path "$data_path" \
    #     --template_path bio_templates.json \
    #     --checkpoint_path "$checkpoint_path" \
    #     --num_steps 20000 \
    #     --knowledge_deleting \
    #     --knowledge_deleting_use_kl \
    #     --current_prob_based_loss \
    #     --run_name "proposed" \
    #     --gpu $gpu
done

