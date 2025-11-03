from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import transformers

from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torch

from tqdm import tqdm
import numpy as np

import random
import json
import os

import argparse

from tqdm import tqdm
from utils import ATTRIBUTE_LIST 
from dataset import RandomBioDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train a plain transformer model for knowledge analysis")
    
    parser.add_argument("--run_name", type=str, default="test", help="Name of the run for logging purposes")

    parser.add_argument("--use_entropy_loss", action="store_true", default=False, help="Use entropy loss")
    parser.add_argument("--mix_augmented_prefixed_data", action="store_true", default=False, help="Mix prefixed data")
    parser.add_argument("--prefix_mode", type=str, default="repeat", choices=["repeat", "metadata"], help="Prefix mode")
    
    # use pretrained model
    parser.add_argument("--use_pretrained", action="store_true", help="Whether to use a pretrained model")

    parser.add_argument("--model_name", type=str, default="gpt2", help="Pretrained model name or path")
    parser.add_argument("--data_path", type=str, default="data/bio_data_100000_42.json", help="Path to the training data")
    parser.add_argument("--template_path", type=str, default="bio_templates.json", help="Path to the templates JSON file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the model")
    parser.add_argument("--num_eval_data", type=int, default=10000, help="Number of evaluation data points")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_steps", type=int, default=100000, help="Number of training epochs")
    # parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for the optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for the learning rate scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--logging_steps", type=int, default=50, help="Number of steps between logging")
    parser.add_argument("--save_steps", type=int, default=10000, help="Number of steps between model saves")
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps between evaluations")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use for training")
    
    # model configuration
    parser.add_argument("--model_type", type=str, default="transformer", help="Type of model to use")
    parser.add_argument("--n_layers", type=int, default=8, help="Number of layers in the transformer model")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads in the transformer model")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension (residual stream)")
    parser.add_argument("--d_hidden", type=int, default=2048, help="Hidden dimension of the multi-layer perceptron")
    parser.add_argument("--key_size", type=int, default=64, help="Dimension of the key and values")
    parser.add_argument("--sequence_mixer", type=str, default="attention", choices=["attention", "mlp"], help="Which sequence mixing block to use")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length for the model")


    return parser.parse_args()

    
def main():
    args = parse_args()
    

    if args.use_pretrained:
        args.output_dir = os.path.join(args.output_dir, args.model_name)
    else:
        args.output_dir = os.path.join(args.output_dir, "plain_transformer")

    if args.model_type == "transformer":
        model_run_name = f"transformer_step{args.num_steps}_d{args.d_model}_d_hidden{args.d_hidden}_n_layers{args.n_layers}_n_heads{args.n_heads}"

    args.run_name = f"{args.run_name}_{args.data_path.split('/')[-1].split('.')[0]}_{model_run_name}"
    
    if args.use_entropy_loss:
        args.run_name += "_entropy"

    if args.mix_augmented_prefixed_data:
        args.run_name += f"_mix_augmented_{args.prefix_mode}"

    os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    
    # total_steps = args.num_steps * args.accumulation_steps
    total_steps = args.num_steps
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for GPT-2 compatibility
    tokenizer.add_special_tokens({"additional_special_tokens":["[X]", "[Y]"]})  # Add special tokens for placeholders
    
    if args.use_pretrained:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
        print(f"load pretrained model from {args.model_name}")
    else:
        config = AutoConfig.from_pretrained(args.model_name)
        config.n_layer = args.n_layers
        config.n_head = args.n_heads
        config.hidden_size = args.d_model
        config.d_ff = args.d_hidden
        config.n_positions = args.max_len
        config.n_embd = args.d_model
        config.key_size = args.key_size
        config.sequence_mixer = args.sequence_mixer
        
        model = GPT2LMHeadModel(config)
        print(f"make new model from scratch")
        
    model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings to match tokenizer size
    # device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # if "llama" in args.model_name.lower():
    #     print("Enable gradient checkpointing")
    #     model.gradient_checkpointing_enable()

    train_dataset = RandomBioDataset(
        data_path=args.data_path,
        template_path=args.template_path,
        tokenizer=tokenizer,
        mode="train"
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    print(f"Using original training data with {len(train_dataset)} samples.")
    if args.mix_augmented_prefixed_data:
        augmented_prefix_train_dataset = RandomBioDataset(
            data_path=args.data_path,
            template_path=args.template_path,
            tokenizer=tokenizer,
            mode="train",
            add_augmented_prefix=True,
            prefix_mode=args.prefix_mode
        )
        train_dataset = ConcatDataset([train_dataset, augmented_prefix_train_dataset])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=augmented_prefix_train_dataset.collate_fn)
        print(f"Using augmented prefix training data with {len(augmented_prefix_train_dataset)} samples. all dataset size: {len(train_dataset)}")

    test_dataset = RandomBioDataset(
        data_path=args.data_path,
        template_path=args.template_path,
        tokenizer=tokenizer,
        mode="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    # only evaluate on 16k samples
    indices = torch.randperm(len(test_dataset))[:args.num_eval_data]
    test_loader = DataLoader(Subset(test_dataset, indices), batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
                
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #cosine scheduler
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    
    logging_train_loss = []
    logging_train_attr_loss = []
    logging_train_em_acc = []
    logging_test_loss = []
    logging_test_attr_loss = []
    logging_test_em_acc = []
    
    logging_train_loss_dict = {}
    logging_train_attr_loss_dict = {}
    logging_train_em_acc_dict = {}
    logging_test_loss_dict = {}
    logging_test_attr_loss_dict = {}
    logging_test_em_acc_dict = {}
    
    update_steps = 0
    save_first_all_correct = False
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    for i in tqdm(range(total_steps)):
        batch = next(iter(train_loader))
        if batch is None:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
            batch = next(iter(train_loader))
            
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        name_mask = batch["name_mask"].to(model.device)
        value_mask = batch["value_mask"].to(model.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift logits and input_ids for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_name_mask = name_mask[..., 1:].contiguous()
        shift_value_mask = value_mask[..., 1:].contiguous()
        
        #ignore padding tokens in loss calculation
        shift_labels[shift_labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss calculation

        # Calculate loss
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        attr_loss = loss[shift_value_mask.view(-1) >= 1]
        
        train_one_batch_em_acc = []
        shift_predicted = shift_logits.argmax(dim=-1)
        # Calculate EM of each attribute
        for b_i in range(shift_value_mask.size(0)):
            one_data_em_acc = 0
            for attr_i in range(len(ATTRIBUTE_LIST)):
                label_value = tokenizer.decode(shift_labels[b_i][shift_value_mask[b_i] == (attr_i + 1)])
                
                # Get the predicted value for the attribute
                predicted_value = tokenizer.decode(shift_predicted[b_i][shift_value_mask[b_i] == (attr_i + 1)])
                
                # Check if the predicted value matches the actual value
                if predicted_value.strip() == label_value.strip():
                    one_data_em_acc += 1
            train_one_batch_em_acc.append(one_data_em_acc / len(ATTRIBUTE_LIST))
        train_one_batch_em_acc = np.mean(train_one_batch_em_acc)
            
        
        # loss = loss.mean() / args.accumulation_steps  # Normalize loss by accumulation steps
        loss = loss.mean()  # Normalize loss by accumulation steps
        
        if args.use_entropy_loss:
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            probs = log_probs.exp()

            entropy_loss = -(probs * log_probs).sum(dim=-1) * shift_mask
            entropy_loss = entropy_loss.sum() / shift_mask.sum()

            loss += entropy_loss

        attr_loss = attr_loss.sum() / (len(ATTRIBUTE_LIST) * args.batch_size)  # Normalize attribute loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        update_steps += 1
        
        # if (i + 1) % args.accumulation_steps == 0:
        #     optimizer.step()
        #     scheduler.step()
        #     optimizer.zero_grad()
        #     update_steps += 1
        
        if (i + 1) % args.eval_steps == 0:
            print(f"Evaluating at step {i+1}...")
            # Evaluation on test set
            model.eval()
            with torch.no_grad():
                all_test_loss = 0.0
                all_test_attr_loss = 0.0
                all_test_em_acc = []
                
                for test_batch in tqdm(test_loader):
                    test_input_ids = test_batch["input_ids"].to(model.device)
                    test_attention_mask = test_batch["attention_mask"].to(model.device)
                    test_name_mask = test_batch["name_mask"].to(model.device)
                    test_value_mask = test_batch["value_mask"].to(model.device)

                    test_outputs = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    test_logits = test_outputs.logits

                    shift_test_logits = test_logits[..., :-1, :].contiguous()
                    shift_test_labels = test_input_ids[..., 1:].contiguous()
                    shift_test_value_mask = test_value_mask[..., 1:].contiguous()
                    shift_test_labels[shift_test_labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss calculation

                    test_loss = loss_fct(shift_test_logits.view(-1, shift_test_logits.size(-1)), shift_test_labels.view(-1))
                    test_attr_loss = test_loss[shift_test_value_mask.view(-1) >= 1]
                    all_test_loss += test_loss.mean().item()
                    all_test_attr_loss += test_attr_loss.sum().item() / (len(ATTRIBUTE_LIST) * args.eval_batch_size)
                    
                    test_predicted = shift_test_logits.argmax(dim=-1)
                    for b_i in range(shift_test_value_mask.size(0)):
                        one_data_em_acc = 0
                        for attr_i in range(len(ATTRIBUTE_LIST)):
                            label_value = tokenizer.decode(shift_test_labels[b_i][shift_test_value_mask[b_i] == (attr_i + 1)])
                            # print("Label Value:", label_value)
                            
                            # Get the predicted value for the attribute
                            predicted_value = tokenizer.decode(test_predicted[b_i][shift_test_value_mask[b_i] == (attr_i + 1)])
                            # print("Predict Value:", predicted_value)
                            
                            # Check if the predicted value matches the actual value
                            if predicted_value.strip() == label_value.strip():
                                one_data_em_acc += 1
                        all_test_em_acc.append(one_data_em_acc / len(ATTRIBUTE_LIST))
                    

                print(f"Test Loss: {all_test_loss / len(test_loader)}")
                print(f"Test Attribute Loss: {all_test_attr_loss / len(test_loader)}")
                print(f"Test EM Accuracy: {np.mean(all_test_em_acc)}")
                
                logging_test_loss.append(all_test_loss / len(test_loader))
                logging_test_attr_loss.append(all_test_attr_loss / len(test_loader))
                logging_test_em_acc.append(np.mean(all_test_em_acc))
                
                logging_test_loss_dict[update_steps] = all_test_loss / len(test_loader)
                logging_test_attr_loss_dict[update_steps] = all_test_attr_loss / len(test_loader)
                logging_test_em_acc_dict[update_steps] = np.mean(all_test_em_acc)
                
                if not save_first_all_correct and np.mean(all_test_em_acc) == 1.0:
                    print(f"Saving model at step {i+1} as first all correct...")
                    model.save_pretrained(os.path.join(args.output_dir, args.run_name, f"model_first_all_correct_step_{i+1}"))
                    tokenizer.save_pretrained(os.path.join(args.output_dir, args.run_name, f"model_first_all_correct_step_{i+1}"))
                    save_first_all_correct = True
                
            model.train()
        
        if (i + 1) % args.logging_steps == 0:
            print(f"Step {i+1}, Loss: {loss.item()}")
            print(f"Step {i+1}, Attribute Loss: {attr_loss.item()}")
            print(f"Step {i+1}, Train EM Accuracy: {train_one_batch_em_acc}")
            logging_train_loss.append(loss.item())
            logging_train_attr_loss.append(attr_loss.item())
            logging_train_em_acc.append(train_one_batch_em_acc)
            
            logging_train_loss_dict[update_steps] = loss.item()
            logging_train_attr_loss_dict[update_steps] = attr_loss.item()
            logging_train_em_acc_dict[update_steps] = train_one_batch_em_acc
        
        
                
        if (i + 1) % args.save_steps == 0:
            model.save_pretrained(os.path.join(args.output_dir, args.run_name, f"model_step_{i+1}"))
        
            import matplotlib.pyplot as plt
            
            plot_data_list = {
                "train_loss_dict": logging_train_loss_dict,
                "train_attr_loss_dict": logging_train_attr_loss_dict,
                "train_em_acc_dict": logging_train_em_acc_dict,
                "test_loss_dict": logging_test_loss_dict,
                "test_attr_loss_dict": logging_test_attr_loss_dict,
                "test_em_acc_dict": logging_test_em_acc_dict
            }
            
            for key, value in plot_data_list.items():
                with open(os.path.join(args.output_dir, args.run_name, f"{key}.json"), 'w') as f:
                    json.dump(value, f, indent=4)
                
                # Plotting the data
                plt.figure(figsize=(6, 6))
                plt.plot(list(value.keys()), list(value.values()), label=key)
                plt.xlabel('Update Steps')
                if "attr_loss" in key:
                    plt.ylabel('Attribute Loss')
                    plt.yscale('log')  # y축을 로그 스케일로 설정
                elif "em_acc" in key:
                    plt.ylabel('Attribute Accuracy')
                    plt.ylim(0, 1)  # EM Accuracy 범위 설정
                elif "loss" in key:
                    plt.ylabel('Loss')
                    plt.yscale('log')
                plt.title(f'{key}')
                plt.legend()
                plt.savefig(os.path.join(args.output_dir, args.run_name, f"{key}.png"))
                plt.close()
            
    
    # # plotting the training losses
    # plt.figure(figsize=(12, 6))
    # plt.plot(logging_train_loss, label='Train Loss', color='blue')
    # plt.plot(logging_train_attr_loss, label='Train Attribute Loss', color='orange')
    # plt.xlabel('Steps')
    # plt.ylabel('Loss')
    # plt.yscale('log')            # y축을 로그 스케일로 설정
    # plt.title('Training Losses')
    # plt.legend()    
    # plt.savefig(os.path.join(args.output_dir, args.run_name, 'training_losses.png'))
    
    # # plotting the evaluation losses
    # plt.figure(figsize=(12, 6))
    # plt.plot(logging_test_loss, label='Test Loss', color='green')
    # plt.plot(logging_test_attr_loss, label='Test Attribute Loss', color='red')
    # plt.xlabel('Steps')
    # plt.ylabel('Loss')
    # plt.yscale('log')            # y축을 로그 스케일로 설정
    # plt.title('Evaluation Losses')
    # plt.legend()    
    # plt.savefig(os.path.join(args.output_dir, args.run_name, 'evaluation_losses.png'))
    
    # with open(os.path.join(args.output_dir, args.run_name, 'train_loss.json'), 'w') as f:
    #     json.dump(logging_train_loss, f)
    # with open(os.path.join(args.output_dir, args.run_name, 'train_attr_loss.json'), 'w') as f:
    #     json.dump(logging_train_attr_loss, f)
    # with open(os.path.join(args.output_dir, args.run_name, 'test_loss.json'), 'w') as f:
    #     json.dump(logging_test_loss, f)
    # with open(os.path.join(args.output_dir, args.run_name, 'test_attr_loss.json'), 'w') as f:
    #     json.dump(logging_test_attr_loss, f)
    
main()