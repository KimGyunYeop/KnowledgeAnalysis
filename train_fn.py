from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import transformers

from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torch.nn.functional as F
import torch

from tqdm import tqdm
import numpy as np

import random
import json
import os

import argparse

from tqdm import tqdm
        
import matplotlib.pyplot as plt
from utils import ATTRIBUTE_LIST 
from dataset import RandomBioDataset
import copy

def filtering_data(args, model, tokenizer, dataset):
    # tmp_data = copy.deepcopy(dataset.data)
    tmp_data = []
    
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    count = 0
    for i, test_batch in enumerate(tqdm(test_loader)):
        B, L = test_batch["input_ids"].shape
        test_input_ids = test_batch["input_ids"].to(model.device)
        test_attention_mask = test_batch["attention_mask"].to(model.device)
        test_name_mask = test_batch["name_mask"].to(model.device)
        test_value_mask = test_batch["value_mask"].to(model.device)
        test_labels = test_batch["custom_label"].to(model.device)

        test_outputs = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        test_logits = test_outputs.logits

        shift_test_logits = test_logits[..., :-1, :].contiguous()
        shift_test_labels = test_labels[..., 1:].contiguous()
        shift_test_value_mask = test_value_mask[..., 1:].contiguous()
        prev = F.pad(shift_test_value_mask[:, :-1], (1, 0), value=0)
        starts = (shift_test_value_mask != 0) & (shift_test_value_mask != prev)
        shift_test_first_value_mask = shift_test_value_mask * starts
        shift_test_labels[shift_test_labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss calculation
        # print(f"shift_test_labels shape: {shift_test_labels}")
        # print(f"shift_test_value_mask shape: {shift_test_value_mask}")
        # print(f"shift_test_first_value_mask shape: {shift_test_first_value_mask}")
        # # breakpoint()

        test_loss = loss_fct(shift_test_logits.view(-1, shift_test_logits.size(-1)), shift_test_labels.view(-1)).view(B, L - 1)
        test_confidence = torch.softmax(shift_test_logits, dim=-1).max(dim=-1).values
        # test_attr_loss = test_loss[shift_test_value_mask.view(-1) >= 1]
        # all_test_loss += test_loss.mean().item()
        # all_test_attr_loss += test_attr_loss.sum().item() / (len(ATTRIBUTE_LIST) * args.eval_batch_size)
        
        test_predicted = shift_test_logits.argmax(dim=-1)
        for b_i in range(B):
            one_data_test_attr_loss = test_loss[b_i][shift_test_value_mask[b_i] >= 1].sum().item() / (shift_test_value_mask[b_i] > 0).sum().item()
            one_data_test_first_attr_loss = test_loss[b_i][shift_test_first_value_mask[b_i] >= 1].sum().item() / (shift_test_first_value_mask[b_i] > 0).sum().item()
            one_data_test_confidence = test_confidence[b_i][shift_test_value_mask[b_i] >= 1].sum().item() / (shift_test_value_mask[b_i] > 0).sum().item()
            one_data_test_first_attr_confidence = test_confidence[b_i][shift_test_first_value_mask[b_i] >= 1].sum().item() / (shift_test_first_value_mask[b_i] > 0).sum().item()

        if one_data_test_first_attr_loss < 1:
            tmp_data.append(dataset.data[i])
            count += 1

    print(f"filtered data count: {count}")
    return tmp_data

def train(args, model, tokenizer, train_data, test_data, original_data=None, mode="train"):
    if mode == "knowledge_deleting" and args.knowledge_deleting_use_kl:
        reference_model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path)
        reference_model.to(model.device)
        # Freeze original model parameters
        for param in reference_model.parameters():
            param.requires_grad = False

    print(f"Starting {mode}...")
    model.train()

    train_dataset = RandomBioDataset(
        data_path=train_data,
        template_path=args.template_path,
        tokenizer=tokenizer,
        mode="train",
        add_augmented_prefix=args.add_augmented_prefix,
        prefix_mode=args.prefix_mode
    )
    if args.add_pretrained_data:
        original_dataset = RandomBioDataset(
            data_path=original_data,
            template_path=args.template_path,
            tokenizer=tokenizer,
            mode="train"
        )
        train_dataset = ConcatDataset([train_dataset, original_dataset])

    if args.mix_augmented_prefixed_data:
        # tmp_train_dataset = RandomBioDataset(
        #     data_path=train_data,
        #     template_path=args.template_path,
        #     tokenizer=tokenizer,
        #     mode="train",
        #     add_augmented_prefix= not args.add_augmented_prefix,
        #     prefix_mode=args.prefix_mode
        # )
        # train_dataset = ConcatDataset([train_dataset, tmp_train_dataset])
        tmp_original_dataset = RandomBioDataset(
            data_path=original_data,
            template_path=args.template_path,
            tokenizer=tokenizer,
            mode="train",
            add_augmented_prefix= args.add_augmented_prefix,
            prefix_mode=args.prefix_mode
        )
        train_dataset = ConcatDataset([train_dataset, tmp_original_dataset])
    
    test_dataset = RandomBioDataset(
        data_path=test_data,
        template_path=args.template_path,
        tokenizer=tokenizer,
        mode="test"
    )
    
    # filtered_data = filtering_data(args, model, tokenizer, train_dataset)
    # train_dataset = RandomBioDataset(
    #     data_path=filtered_data,
    #     template_path=args.template_path,
    #     tokenizer=tokenizer,
    #     mode="train",
    #     add_augmented_prefix=args.add_augmented_prefix,
    #     prefix_mode=args.prefix_mode
    # )
    # tmp_train_dataset = RandomBioDataset(
    #     data_path=filtered_data,
    #     template_path=args.template_path,
    #     tokenizer=tokenizer,
    #     mode="train",
    # )
    # train_dataset = ConcatDataset([train_dataset, tmp_train_dataset])
    # test_dataset = RandomBioDataset(
    #     data_path=filtered_data,
    #     template_path=args.template_path,
    #     tokenizer=tokenizer,
    #     mode="test"
    # )

    original_test_dataset = RandomBioDataset(
        data_path=original_data,
        template_path=args.template_path,
        tokenizer=tokenizer,
        mode="test"
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Original Test dataset size: {len(original_test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    original_test_loader = DataLoader(original_test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=original_test_dataset.collate_fn)
    # only evaluate on 16k samples
    indices = torch.randperm(len(original_test_dataset))[:args.num_eval_data]
    if len(test_dataset) > args.num_eval_data:
        test_indices = torch.randperm(len(test_dataset))[:args.num_eval_data]
        test_loader = DataLoader(Subset(test_dataset,test_indices), batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    original_test_loader = DataLoader(Subset(original_test_dataset, indices), batch_size=args.eval_batch_size, shuffle=False, collate_fn=original_test_dataset.collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if mode == "train":
        # total_steps = args.num_steps * args.accumulation_steps
        total_steps = args.num_steps
    elif mode == "knowledge_deleting":
        total_steps = args.knowledge_deleting_steps if args.knowledge_deleting_finish_strategy == "step" else args.num_steps
        
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
    
    logging_original_test_loss = []
    logging_original_test_attr_loss = []
    logging_original_test_em_acc = []
    logging_original_test_loss_dict = {}
    logging_original_test_attr_loss_dict = {}
    logging_original_test_em_acc_dict = {}
    
    update_steps = 0
    save_first_all_correct = False
    
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    for i in tqdm(range(total_steps)):
        if i % args.eval_steps == 0:
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
                    test_labels = test_batch["custom_label"].to(model.device)

                    test_outputs = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    test_logits = test_outputs.logits

                    shift_test_logits = test_logits[..., :-1, :].contiguous()
                    shift_test_labels = test_labels[..., 1:].contiguous()
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

                all_original_test_loss = 0.0
                all_original_test_attr_loss = 0.0
                all_original_test_em_acc = []
                
                for original_test_batch in tqdm(original_test_loader):
                    original_test_input_ids = original_test_batch["input_ids"].to(model.device)
                    original_test_attention_mask = original_test_batch["attention_mask"].to(model.device)
                    original_test_name_mask = original_test_batch["name_mask"].to(model.device)
                    original_test_value_mask = original_test_batch["value_mask"].to(model.device)
                    original_test_custom_label = original_test_batch["custom_label"].to(model.device)

                    original_test_outputs = model(input_ids=original_test_input_ids, attention_mask=original_test_attention_mask)
                    original_test_logits = original_test_outputs.logits

                    shift_original_test_logits = original_test_logits[..., :-1, :].contiguous()
                    shift_original_test_labels = original_test_custom_label[..., 1:].contiguous()
                    shift_original_test_value_mask = original_test_value_mask[..., 1:].contiguous()
                    shift_original_test_labels[shift_original_test_labels == tokenizer.pad_token_id] = -100
                    
                    original_test_loss = loss_fct(shift_original_test_logits.view(-1, shift_original_test_logits.size(-1)), shift_original_test_labels.view(-1))
                    original_test_attr_loss = original_test_loss[shift_original_test_value_mask.view(-1) >= 1]
                    all_original_test_loss += original_test_loss.mean().item()
                    all_original_test_attr_loss += original_test_attr_loss.sum().item() / (len(ATTRIBUTE_LIST) * args.eval_batch_size)
                    original_test_predicted = shift_original_test_logits.argmax(dim=-1)
                    for b_i in range(shift_original_test_value_mask.size(0)):
                        one_data_em_acc = 0
                        for attr_i in range(len(ATTRIBUTE_LIST)):
                            label_value = tokenizer.decode(shift_original_test_labels[b_i][shift_original_test_value_mask[b_i] == (attr_i + 1)])
                            # print("Label Value:", label_value)
                            
                            # Get the predicted value for the attribute
                            predicted_value = tokenizer.decode(original_test_predicted[b_i][shift_original_test_value_mask[b_i] == (attr_i + 1)])
                            # print("Predict Value:", predicted_value)
                            # Check if the predicted value matches the actual value
                            if predicted_value.strip() == label_value.strip():
                                one_data_em_acc += 1
                        all_original_test_em_acc.append(one_data_em_acc / len(ATTRIBUTE_LIST))
                print(f"Original Test Loss: {all_original_test_loss / len(original_test_loader)}")
                print(f"Original Test Attribute Loss: {all_original_test_attr_loss / len(original_test_loader)}")
                print(f"Original Test EM Accuracy: {np.mean(all_original_test_em_acc)}")
                logging_original_test_loss_dict[update_steps] = all_original_test_loss / len(original_test_loader)
                logging_original_test_attr_loss_dict[update_steps] = all_original_test_attr_loss / len(original_test_loader)
                logging_original_test_em_acc_dict[update_steps] = np.mean(all_original_test_em_acc)
                logging_original_test_loss.append(all_original_test_loss / len(original_test_loader))
                logging_original_test_attr_loss.append(all_original_test_attr_loss / len(original_test_loader))
                logging_original_test_em_acc.append(np.mean(all_original_test_em_acc))
                
            model.train()
            
        batch = next(iter(train_loader))
        if batch is None:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
            batch = next(iter(train_loader))
            
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        name_mask = batch["name_mask"].to(model.device)
        value_mask = batch["value_mask"].to(model.device)
        labels = batch["custom_label"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift logits and input_ids for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_name_mask = name_mask[..., 1:].contiguous()
        shift_value_mask = value_mask[..., 1:].contiguous()
        
        tmp_shift_label = shift_labels.clone()
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
        
        if mode == "train":
            # loss = loss.mean() / args.accumulation_steps  # Normalize loss by accumulation steps
            if getattr(args, "current_prob_based_loss", False) and args.current_prob_based_loss:
            # if args.current_prob_based_loss:
                probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                label_probs = torch.gather(probs, dim=-1, index=tmp_shift_label.unsqueeze(-1)).squeeze(-1).view(-1)
                # loss = (loss * label_probs.detach()).view(-1)
                loss = -torch.exp(torch.log(label_probs + 1e-10) - torch.log(label_probs.detach() + 1e-10))
                loss = loss[shift_labels.view(-1) != -100]

            loss = loss.mean()  
            # loss = attr_loss.mean() 
        elif mode == "knowledge_deleting":
            probs = torch.nn.functional.softmax(shift_logits, dim=-1)
            max_probs = probs.max(dim=-1)[0].view(-1)
            # attr_max_probs = max_probs[shift_value_mask.view(-1) >= 1]
            if args.knowledge_deleting_loss_type == "ppo":
                loss = torch.exp(torch.log(max_probs + 1e-10) - torch.log(max_probs.detach() + 1e-10))
            else:
                loss = max_probs
            loss = loss[shift_value_mask.view(-1) >= 1]
            loss = loss.sum() / (shift_value_mask.view(-1) >= 1).sum()

            if args.knowledge_deleting_use_kl:
                with torch.no_grad():
                    reference_outputs = reference_model(input_ids=input_ids, attention_mask=attention_mask)
                    reference_shift_logits = reference_outputs.logits[..., :-1, :].contiguous()

                tmp_mask = shift_value_mask < 1
                tmp_mask = tmp_mask.unsqueeze(-1).expand_as(shift_logits)
                shift_logits = shift_logits.masked_fill(tmp_mask, 0.0)
                reference_shift_logits = reference_shift_logits.masked_fill(tmp_mask, 0.0)
                kl_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(shift_logits, dim=-1), torch.nn.functional.softmax(reference_shift_logits.detach(), dim=-1), reduction='batchmean')
                
                # kl_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(shift_logits, dim=-1), torch.nn.functional.softmax(reference_shift_logits.detach(), dim=-1), reduction='none').sum(dim=-1).view(-1)
                # kl_loss = kl_loss[shift_value_mask.view(-1) < 1]
                # kl_loss = kl_loss.sum() / shift_logits.size(0)
                loss += kl_loss

        else:
            raise ValueError("Unknown mode")

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
            if mode == "train":
                model.save_pretrained(os.path.join(args.output_dir, args.run_name, f"model_step_{i+1}"))
                
                plot_data_list = {
                    "train_loss_dict": logging_train_loss_dict,
                    "train_attr_loss_dict": logging_train_attr_loss_dict,
                    "train_em_acc_dict": logging_train_em_acc_dict,
                    "test_loss_dict": logging_test_loss_dict,
                    "test_attr_loss_dict": logging_test_attr_loss_dict,
                    "test_em_acc_dict": logging_test_em_acc_dict,
                    "original_test_loss_dict": logging_original_test_loss_dict,
                    "original_test_attr_loss_dict": logging_original_test_attr_loss_dict,
                    "original_test_em_acc_dict": logging_original_test_em_acc_dict
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
                
                knowledge_collapse_data = {
                    # "loss" : [list(logging_train_loss_dict.values()), list(logging_test_loss_dict.values())],
                    "attr_loss" : [list(logging_original_test_attr_loss_dict.values()) ,list(logging_test_attr_loss_dict.values())],
                    "em_acc" : [list(logging_original_test_em_acc_dict.values()), list(logging_test_em_acc_dict.values())]
                }
                
                for key, data in knowledge_collapse_data.items():
                    plt.figure(figsize=(6, 6))
                    plt.plot(data[0], data[1], label=key)
                    # plt.xlabel('Update Steps')
                    if "attr_loss" in key:
                        plt.xlabel('pre-train Attribute Loss')
                        plt.ylabel('fine-tune Attribute Loss')
                        plt.xscale('log')  # x축을 로그 스케일로 설정
                        plt.yscale('log')  # y축을 로그 스케일로 설정
                        # plt.yscale('log')  # y축을 로그 스케일로 설정
                    elif "em_acc" in key:
                        plt.xlabel('pre-train EM Accuracy')
                        plt.ylabel('fine-tune EM Accuracy')
                        # plt.ylim(0, 1)  # EM Accuracy 범위 설정
                    elif "loss" in key:
                        plt.ylabel('Loss')
                        plt.yscale('log')
                    plt.title(f'Knowledge Collapse - {key}')
                    plt.legend()
                    plt.savefig(os.path.join(args.output_dir, args.run_name, f"knowledge_collapse_{key}.png"))
                    plt.close()
                print(f"Model saved at step {i+1} and plots generated.")

            elif mode == "knowledge_deleting":
                plot_data_list = {
                    "deleting_train_loss_dict": logging_train_loss_dict,
                    "deleting_test_loss_dict": logging_test_loss_dict,
                    "deleting_test_attr_loss_dict": logging_test_attr_loss_dict,
                    "deleting_test_em_acc_dict": logging_test_em_acc_dict,
                    "deleting_original_test_loss_dict": logging_original_test_loss_dict,
                    "deleting_original_test_attr_loss_dict": logging_original_test_attr_loss_dict,
                    "deleting_original_test_em_acc_dict": logging_original_test_em_acc_dict
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
                

    return model



def train_with_soft_prompt(args, model, tokenizer, train_data, test_data, original_data=None, mode="train"):
    if args.use_kl:
        reference_model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path)
        reference_model.to(model.device)
        # Freeze original model parameters
        for param in reference_model.parameters():
            param.requires_grad = False

    print(f"Starting {mode}...")
    model.train()

    train_dataset = RandomBioDataset(
        data_path=train_data,
        template_path=args.template_path,
        tokenizer=tokenizer,
        mode="train",
        add_augmented_prefix=args.add_augmented_prefix,
        prefix_mode=args.prefix_mode
    )
    if args.mix_augmented_prefixed_data:
        tmp_train_dataset = RandomBioDataset(
            data_path=train_data,
            template_path=args.template_path,
            tokenizer=tokenizer,
            mode="train",
            add_augmented_prefix= not args.add_augmented_prefix,
            prefix_mode=args.prefix_mode
        )
        train_dataset = ConcatDataset([train_dataset, tmp_train_dataset])
    test_dataset = RandomBioDataset(
        data_path=test_data,
        template_path=args.template_path,
        tokenizer=tokenizer,
        mode="test"
    )
    
    # filtered_data = filtering_data(args, model, tokenizer, train_dataset)
    # train_dataset = RandomBioDataset(
    #     data_path=filtered_data,
    #     template_path=args.template_path,
    #     tokenizer=tokenizer,
    #     mode="train",
    #     add_augmented_prefix=args.add_augmented_prefix,
    #     prefix_mode=args.prefix_mode
    # )
    # tmp_train_dataset = RandomBioDataset(
    #     data_path=filtered_data,
    #     template_path=args.template_path,
    #     tokenizer=tokenizer,
    #     mode="train",
    # )
    # train_dataset = ConcatDataset([train_dataset, tmp_train_dataset])
    # test_dataset = RandomBioDataset(
    #     data_path=filtered_data,
    #     template_path=args.template_path,
    #     tokenizer=tokenizer,
    #     mode="test"
    # )

    original_test_dataset = RandomBioDataset(
        data_path=original_data,
        template_path=args.template_path,
        tokenizer=tokenizer,
        mode="test"
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Original Test dataset size: {len(original_test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    original_test_loader = DataLoader(original_test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=original_test_dataset.collate_fn)
    # only evaluate on 16k samples
    indices = torch.randperm(len(original_test_dataset))[:args.num_eval_data]
    if len(test_dataset) > args.num_eval_data:
        test_indices = torch.randperm(len(test_dataset))[:args.num_eval_data]
        test_loader = DataLoader(Subset(test_dataset,test_indices), batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    original_test_loader = DataLoader(Subset(original_test_dataset, indices), batch_size=args.eval_batch_size, shuffle=False, collate_fn=original_test_dataset.collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if mode == "train":
        # total_steps = args.num_steps * args.accumulation_steps
        total_steps = args.num_steps
    elif mode == "knowledge_deleting":
        total_steps = args.knowledge_deleting_steps if args.knowledge_deleting_finish_strategy == "step" else args.num_steps
        
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
    
    logging_original_test_loss = []
    logging_original_test_attr_loss = []
    logging_original_test_em_acc = []
    logging_original_test_loss_dict = {}
    logging_original_test_attr_loss_dict = {}
    logging_original_test_em_acc_dict = {}
    
    update_steps = 0
    save_first_all_correct = False
    
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    for i in tqdm(range(total_steps)):
        
        if i % args.eval_steps == 0:
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
                    test_labels = test_batch["custom_label"].to(model.device)

                    test_outputs = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    test_logits = test_outputs.logits

                    shift_test_logits = test_logits[..., :-1, :].contiguous()
                    shift_test_labels = test_labels[..., 1:].contiguous()
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

                all_original_test_loss = 0.0
                all_original_test_attr_loss = 0.0
                all_original_test_em_acc = []
                
                for original_test_batch in tqdm(original_test_loader):
                    original_test_input_ids = original_test_batch["input_ids"].to(model.device)
                    original_test_attention_mask = original_test_batch["attention_mask"].to(model.device)
                    original_test_name_mask = original_test_batch["name_mask"].to(model.device)
                    original_test_value_mask = original_test_batch["value_mask"].to(model.device)
                    original_test_custom_label = original_test_batch["custom_label"].to(model.device)

                    original_test_outputs = model(input_ids=original_test_input_ids, attention_mask=original_test_attention_mask)
                    original_test_logits = original_test_outputs.logits

                    shift_original_test_logits = original_test_logits[..., :-1, :].contiguous()
                    shift_original_test_labels = original_test_custom_label[..., 1:].contiguous()
                    shift_original_test_value_mask = original_test_value_mask[..., 1:].contiguous()
                    shift_original_test_labels[shift_original_test_labels == tokenizer.pad_token_id] = -100
                    
                    original_test_loss = loss_fct(shift_original_test_logits.view(-1, shift_original_test_logits.size(-1)), shift_original_test_labels.view(-1))
                    original_test_attr_loss = original_test_loss[shift_original_test_value_mask.view(-1) >= 1]
                    all_original_test_loss += original_test_loss.mean().item()
                    all_original_test_attr_loss += original_test_attr_loss.sum().item() / (len(ATTRIBUTE_LIST) * args.eval_batch_size)
                    original_test_predicted = shift_original_test_logits.argmax(dim=-1)
                    for b_i in range(shift_original_test_value_mask.size(0)):
                        one_data_em_acc = 0
                        for attr_i in range(len(ATTRIBUTE_LIST)):
                            label_value = tokenizer.decode(shift_original_test_labels[b_i][shift_original_test_value_mask[b_i] == (attr_i + 1)])
                            # print("Label Value:", label_value)
                            
                            # Get the predicted value for the attribute
                            predicted_value = tokenizer.decode(original_test_predicted[b_i][shift_original_test_value_mask[b_i] == (attr_i + 1)])
                            # print("Predict Value:", predicted_value)
                            # Check if the predicted value matches the actual value
                            if predicted_value.strip() == label_value.strip():
                                one_data_em_acc += 1
                        all_original_test_em_acc.append(one_data_em_acc / len(ATTRIBUTE_LIST))
                print(f"Original Test Loss: {all_original_test_loss / len(original_test_loader)}")
                print(f"Original Test Attribute Loss: {all_original_test_attr_loss / len(original_test_loader)}")
                print(f"Original Test EM Accuracy: {np.mean(all_original_test_em_acc)}")
                logging_original_test_loss_dict[update_steps] = all_original_test_loss / len(original_test_loader)
                logging_original_test_attr_loss_dict[update_steps] = all_original_test_attr_loss / len(original_test_loader)
                logging_original_test_em_acc_dict[update_steps] = np.mean(all_original_test_em_acc)
                logging_original_test_loss.append(all_original_test_loss / len(original_test_loader))
                logging_original_test_attr_loss.append(all_original_test_attr_loss / len(original_test_loader))
                logging_original_test_em_acc.append(np.mean(all_original_test_em_acc))
                
            model.train()
            
        batch = next(iter(train_loader))
        if batch is None:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
            batch = next(iter(train_loader))
            
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        name_mask = batch["name_mask"].to(model.device)
        value_mask = batch["value_mask"].to(model.device)
        labels = batch["custom_label"].to(model.device)
        
        shift_labels = labels[..., 1:].contiguous()
        shift_name_mask = name_mask[..., 1:].contiguous()
        shift_value_mask = value_mask[..., 1:].contiguous()
        tmp_shift_label = shift_labels.clone()
        #ignore padding tokens in loss calculation
        shift_labels[shift_labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss calculation
        
        #model freeze
        model.requires_grad_(False)
            
        # random learnable embedding shape batch, 1, hidden size
        if args.soft_prompt_type == "front":
            len_soft_prompt = 5
            soft_prompt = torch.nn.Parameter(torch.randn(input_ids.size(0), len_soft_prompt, model.config.hidden_size, device=model.device) * 0.02, requires_grad=True)
        elif args.soft_prompt_type == "add":
            # soft_prompt = torch.nn.Parameter(torch.randn(input_ids.size(0), input_ids.size(1), model.config.hidden_size, device=model.device) * 0.01, requires_grad=True)
            soft_prompt = torch.nn.Parameter(torch.zeros(input_ids.size(0), input_ids.size(1), model.config.hidden_size, device=model.device), requires_grad=True)
        soft_prompt_optimizer = torch.optim.Adam([soft_prompt], lr=1e-4)
        use_soft_prompt = True
        for j in range(10000):
            sp_input_embs = model.transformer.wte(input_ids)
            
            if args.soft_prompt_type == "front":
                sp_input_embs = torch.cat([soft_prompt, sp_input_embs], dim=1)  # Concatenate soft prompt to input embeddings
                sp_attention_mask = torch.cat([torch.ones(sp_input_embs.size(0), len_soft_prompt, device=sp_input_embs.device), attention_mask], dim=1)
                sp_outputs = model(inputs_embeds=sp_input_embs, attention_mask=sp_attention_mask)
            elif args.soft_prompt_type == "add":
                # print(f"sp_input_embs: {sp_input_embs.shape}, soft_prompt: {soft_prompt.shape}")
                sp_input_embs = sp_input_embs + soft_prompt
                sp_outputs = model(inputs_embeds=sp_input_embs, attention_mask=attention_mask)

            if args.soft_prompt_type == "front":
                sp_shift_logits = sp_outputs.logits[:, len_soft_prompt:-1, :].contiguous()  #exclude soft prompt's output
            elif args.soft_prompt_type == "add":
                sp_shift_logits = sp_outputs.logits[:, :-1, :].contiguous()  #exclude soft prompt's output

            soft_prompt_loss = loss_fct(sp_shift_logits.view(-1, sp_shift_logits.size(-1)), shift_labels.view(-1))
            # soft_prompt_loss = soft_prompt_loss.view(input_ids.size(0), -1)
            soft_prompt_attr_loss = soft_prompt_loss[shift_value_mask.view(-1) >= 1]
            # print(tokenizer.decode(shift_labels.view(-1)[shift_value_mask.view(-1) >= 1]))
            # print(tokenizer.decode(torch.argmax(sp_shift_logits, dim=-1).view(-1)[shift_value_mask.view(-1) >= 1]))
            # print(f"Step {i}")
            # print(f"Soft Prompt Loss: {soft_prompt_loss.mean().item()}")
            # print(f"Soft Prompt Attribute Loss: {soft_prompt_attr_loss.mean().item()}")
            # print(f"Max Soft Prompt Attribute Loss: {soft_prompt_attr_loss.max()}")
            # print(soft_prompt_attr_loss.mean())
            if soft_prompt_loss.mean() < 0.4:
                use_soft_prompt = j > 0
                
                print(f"Early stopping at soft prompt create step {j} use soft prompt {use_soft_prompt}")
                print(f"decoded: {tokenizer.batch_decode(torch.argmax(sp_shift_logits, dim=-1),skip_special_tokens=True)[0]}")
                print(f"decoded: {tokenizer.batch_decode(input_ids,skip_special_tokens=True)[0]}")
                break
            soft_prompt_loss = soft_prompt_loss.mean()
            soft_prompt_loss.backward()
            soft_prompt_optimizer.step()
            soft_prompt_optimizer.zero_grad()

        
        model.requires_grad_(True)
        soft_prompt.requires_grad_(False)  # Freeze soft prompt parameters after training

        if use_soft_prompt:
            sp_input_embs = model.transformer.wte(input_ids)
            if args.soft_prompt_type == "front":
                sp_input_embs = torch.cat([soft_prompt.detach(), sp_input_embs], dim=1)  # Concatenate soft prompt to input embeddings
                sp_attention_mask = torch.cat([torch.ones(sp_input_embs.size(0), len_soft_prompt, device=sp_input_embs.device), attention_mask], dim=1)
                outputs = model(inputs_embeds=sp_input_embs, attention_mask=sp_attention_mask)
            elif args.soft_prompt_type == "add":
                sp_input_embs = sp_input_embs + soft_prompt.detach()
                outputs = model(inputs_embeds=sp_input_embs, attention_mask=attention_mask)
            print(f"sp_input_embs: {sp_input_embs.shape}")
            if args.soft_prompt_type == "front":
                logits = outputs.logits[:, len_soft_prompt:, :].contiguous()  #exclude soft prompt's output
            else:
                logits = outputs.logits
        else:
            print(f"input_ids: {input_ids.shape}")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Shift logits and input_ids for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()

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
        
        if mode == "train":
            # loss = loss.mean() / args.accumulation_steps  # Normalize loss by accumulation steps
            # if getattr(args, "current_prob_based_loss", False) and args.current_prob_based_loss:
            # # if args.current_prob_based_loss:
            #     probs = torch.nn.functional.softmax(shift_logits, dim=-1)
            #     label_probs = torch.gather(probs, dim=-1, index=tmp_shift_label.unsqueeze(-1)).squeeze(-1).view(-1)
            #     # loss = (loss * label_probs.detach()).view(-1)
            #     loss = -torch.exp(torch.log(label_probs + 1e-10) - torch.log(label_probs.detach() + 1e-10))
            #     loss = loss[shift_labels.view(-1) != -100]

            loss = loss.mean()  
            # loss = attr_loss.mean() 
        elif mode == "knowledge_deleting":
            probs = torch.nn.functional.softmax(shift_logits, dim=-1)
            max_probs = probs.max(dim=-1)[0].view(-1)
            # attr_max_probs = max_probs[shift_value_mask.view(-1) >= 1]
            if args.knowledge_deleting_loss_type == "ppo":
                loss = torch.exp(torch.log(max_probs + 1e-10) - torch.log(max_probs.detach() + 1e-10))
            else:
                loss = max_probs
            loss = loss[shift_value_mask.view(-1) >= 1]
            loss = loss.sum() / (shift_value_mask.view(-1) >= 1).sum()

        if args.use_kl:
            with torch.no_grad():
                reference_outputs = reference_model(input_ids=input_ids, attention_mask=attention_mask)
                reference_shift_logits = reference_outputs.logits[..., :-1, :].contiguous()

            tmp_mask = shift_value_mask < 1
            tmp_mask = tmp_mask.unsqueeze(-1).expand_as(shift_logits)
            shift_logits = shift_logits.masked_fill(tmp_mask, 0.0)
            reference_shift_logits = reference_shift_logits.masked_fill(tmp_mask, 0.0)
            kl_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(shift_logits, dim=-1), torch.nn.functional.softmax(reference_shift_logits.detach(), dim=-1), reduction='mean')
            
            # kl_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(shift_logits, dim=-1), torch.nn.functional.softmax(reference_shift_logits.detach(), dim=-1), reduction='none').sum(dim=-1).view(-1)
            # kl_loss = kl_loss[shift_value_mask.view(-1) < 1]
            # kl_loss = kl_loss.sum() / shift_logits.size(0)
            print(f"Step {i+1}, KL Loss: {kl_loss.item()}")
            loss += kl_loss

        attr_loss = attr_loss.sum() / (len(ATTRIBUTE_LIST) * args.batch_size)  # Normalize attribute loss
        
        print(f"Step {i+1}, Loss: {loss.item()}")
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
            if mode == "train":
                model.save_pretrained(os.path.join(args.output_dir, args.run_name, f"model_step_{i+1}"))
                
                plot_data_list = {
                    "train_loss_dict": logging_train_loss_dict,
                    "train_attr_loss_dict": logging_train_attr_loss_dict,
                    "train_em_acc_dict": logging_train_em_acc_dict,
                    "test_loss_dict": logging_test_loss_dict,
                    "test_attr_loss_dict": logging_test_attr_loss_dict,
                    "test_em_acc_dict": logging_test_em_acc_dict,
                    "original_test_loss_dict": logging_original_test_loss_dict,
                    "original_test_attr_loss_dict": logging_original_test_attr_loss_dict,
                    "original_test_em_acc_dict": logging_original_test_em_acc_dict
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
                
                knowledge_collapse_data = {
                    # "loss" : [list(logging_train_loss_dict.values()), list(logging_test_loss_dict.values())],
                    "attr_loss" : [list(logging_original_test_attr_loss_dict.values()) ,list(logging_test_attr_loss_dict.values())],
                    "em_acc" : [list(logging_original_test_em_acc_dict.values()), list(logging_test_em_acc_dict.values())]
                }
                
                for key, data in knowledge_collapse_data.items():
                    plt.figure(figsize=(6, 6))
                    plt.plot(data[0], data[1], label=key)
                    # plt.xlabel('Update Steps')
                    if "attr_loss" in key:
                        plt.xlabel('pre-train Attribute Loss')
                        plt.ylabel('fine-tune Attribute Loss')
                        plt.xscale('log')  # x축을 로그 스케일로 설정
                        plt.yscale('log')  # y축을 로그 스케일로 설정
                        # plt.yscale('log')  # y축을 로그 스케일로 설정
                    elif "em_acc" in key:
                        plt.xlabel('pre-train EM Accuracy')
                        plt.ylabel('fine-tune EM Accuracy')
                        # plt.ylim(0, 1)  # EM Accuracy 범위 설정
                    elif "loss" in key:
                        plt.ylabel('Loss')
                        plt.yscale('log')
                    plt.title(f'Knowledge Collapse - {key}')
                    plt.legend()
                    plt.savefig(os.path.join(args.output_dir, args.run_name, f"knowledge_collapse_{key}.png"))
                    plt.close()
                print(f"Model saved at step {i+1} and plots generated.")

            elif mode == "knowledge_deleting":
                plot_data_list = {
                    "deleting_train_loss_dict": logging_train_loss_dict,
                    "deleting_test_loss_dict": logging_test_loss_dict,
                    "deleting_test_attr_loss_dict": logging_test_attr_loss_dict,
                    "deleting_test_em_acc_dict": logging_test_em_acc_dict,
                    "deleting_original_test_loss_dict": logging_original_test_loss_dict,
                    "deleting_original_test_attr_loss_dict": logging_original_test_attr_loss_dict,
                    "deleting_original_test_em_acc_dict": logging_original_test_em_acc_dict
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
                

    return model
