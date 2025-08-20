from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import transformers

from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
import torch

from tqdm import tqdm
import numpy as np

import random
import json
import os

import argparse

from tqdm import tqdm

ATTRIBUTE_LIST = ["birth_date", "birth_place", "university", "major", "company", "location"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train a plain transformer model for knowledge analysis")
    
    parser.add_argument("--run_name", type=str, default="test", help="Name of the run for logging purposes")
    
    parser.add_argument("--model_name", type=str, default="gpt2", help="Pretrained model name or path")
    parser.add_argument("--data_path", type=str, default="data/bio_data_30000_42_other_1764.json", help="Path to the training data")
    parser.add_argument("--template_path", type=str, default="bio_templates.json", help="Path to the templates JSON file")
    parser.add_argument("--output_dir", type=str, default="results_test/plain_transformer", help="Directory to save the model")
    # parser.add_argument("--num_eval_data", type=int, default=10000, help="Number of evaluation data points")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    # parser.add_argument("--num_steps", type=int, default=100000, help="Number of training epochs")
    # parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    # parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    # parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for the optimizer")
    # parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for the learning rate scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    # parser.add_argument("--logging_steps", type=int, default=50, help="Number of steps between logging")
    # parser.add_argument("--save_steps", type=int, default=10000, help="Number of steps between model saves")
    # parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps between evaluations")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use for training")
    
    # model configuration
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to a pretrained model to load")
    # parser.add_argument("--model_type", type=str, default="transformer", help="Type of model to use")
    # parser.add_argument("--n_layers", type=int, default=8, help="Number of layers in the transformer model")
    # parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads in the transformer model")
    # parser.add_argument("--d_model", type=int, default=512, help="Model dimension (residual stream)")
    # parser.add_argument("--d_hidden", type=int, default=2048, help="Hidden dimension of the multi-layer perceptron")
    # parser.add_argument("--key_size", type=int, default=64, help="Dimension of the key and values")
    # parser.add_argument("--sequence_mixer", type=str, default="attention", choices=["attention", "mlp"], help="Which sequence mixing block to use")
    # parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length for the model")
    
    return parser.parse_args()

class RandomBioDataset(Dataset):
    def __init__(self, data_path, template_path, tokenizer, mode="train"):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        with open(template_path, 'r') as f:
            self.templates = json.load(f)
            
        self.tokenizer = tokenizer
        
        self.len_temp = len(self.data[0]["train_tamplates"]["birth_date_template"])
        
        if mode == "train":
            self.temp_mask = 0
            self.use_temp_num = sum(self.data[0]["train_tamplates"]["birth_date_template"])
        elif mode == "test":
            self.temp_mask = 1
            self.use_temp_num = self.len_temp - sum(self.data[0]["train_tamplates"]["birth_date_template"])
            
        self.name_holder = tokenizer.convert_tokens_to_ids("[X]")
        self.value_holder = tokenizer.convert_tokens_to_ids("[Y]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        name = item['name']
        
        shuffled_attributes = ATTRIBUTE_LIST.copy()
        random.shuffle(shuffled_attributes)
        
        attr_values = []
        use_templates = []
        for attr in shuffled_attributes:
            attr_values.append(item[attr])
            while True:
                template_index = random.randint(0, self.len_temp - 1)
                if item["train_tamplates"][f"{attr}_template"][template_index] == self.temp_mask:
                    use_templates.append(self.templates[f"{attr}_template"][template_index])
                    break
                
        template = " ".join(use_templates)
        
        tokens = self.tokenizer(template)
        # print("before decoded input_ids:", self.tokenizer.convert_ids_to_tokens(tokens.input_ids))
        
        name_tokens = self.tokenizer(name, add_special_tokens=False)
        name_indices = []
        value_indices = []
        len_name_tokens = len(name_tokens.input_ids)
        for value in attr_values:
            attr_tokens = self.tokenizer(value, add_special_tokens=False)
            
            name_index = tokens.input_ids.index(self.name_holder)
            tokens.input_ids = tokens.input_ids[:name_index] + name_tokens.input_ids + tokens.input_ids[name_index + 1:]
            tokens.attention_mask = tokens.attention_mask[:name_index] + name_tokens.attention_mask + tokens.attention_mask[name_index + 1:]
            
            name_indices.append([name_index, name_index + len_name_tokens])
            
            value_index = tokens.input_ids.index(self.value_holder)
            len_value_tokens = len(attr_tokens.input_ids)
            tokens.input_ids = tokens.input_ids[:value_index] + attr_tokens.input_ids + tokens.input_ids[value_index + 1:]
            tokens.attention_mask = tokens.attention_mask[:value_index] + attr_tokens.attention_mask + tokens.attention_mask[value_index + 1:]
            
            value_indices.append([value_index, value_index + len_value_tokens])
        
        name_mask = torch.zeros(len(tokens.input_ids), dtype=torch.long)
        value_mask = torch.zeros(len(tokens.input_ids), dtype=torch.long)
        
        for i, (start, end) in enumerate(name_indices):
            name_mask[start:end] = i + 1
        
        for i, (start, end) in enumerate(value_indices):
            value_mask[start:end] = i + 1
            
        input_ids = torch.tensor(tokens.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(tokens.attention_mask, dtype=torch.long)
        
        # print("after decoded input_ids:", self.tokenizer.convert_ids_to_tokens(input_ids))
        
        # print("only name tokens:", self.tokenizer.decode(input_ids[name_mask == 1]))
        # print("only value tokens:", self.tokenizer.decode(input_ids[value_mask == 1]))
            
        return {"input_ids": input_ids, "attention_mask": attention_mask, "name_mask": name_mask, "value_mask": value_mask}

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        name_mask = [item["name_mask"] for item in batch]
        value_mask = [item["value_mask"] for item in batch]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        name_mask = torch.nn.utils.rnn.pad_sequence(name_mask, batch_first=True, padding_value=0)
        value_mask = torch.nn.utils.rnn.pad_sequence(value_mask, batch_first=True, padding_value=0)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "name_mask": name_mask, "value_mask": value_mask}
    
def main():
    args = parse_args()
    args.original_target_file = args.data_path.replace("_other"+args.data_path.split("_other")[-1], ".json")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model_run_name = f"{args.model_name.split('/')[-1]}"
    
    args.run_name = f"{args.run_name}_{args.data_path.split('/')[-1].split('.')[0]}_{model_run_name}"
    
    os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for GPT-2 compatibility
    tokenizer.add_special_tokens({"additional_special_tokens":["[X]", "[Y]"]})  # Add special tokens for placeholders
    
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path)
    
    model.to(device)
    
    test_dataset = RandomBioDataset(
        data_path=args.data_path,
        template_path=args.template_path,
        tokenizer=tokenizer,
        mode="test"
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
    model.eval()
    all_data = test_dataset.data
    with torch.no_grad():
        all_test_loss = 0.0
        all_test_attr_loss = 0.0
        all_test_em_acc = []
        all_test_attr_loss = []
        all_test_first_attr_loss = []
        all_test_confidence = []
        all_test_first_attr_confidence = []
        data_count = 0
        
        for test_batch in tqdm(test_loader):
            B, L = test_batch["input_ids"].shape
            test_input_ids = test_batch["input_ids"].to(model.device)
            test_attention_mask = test_batch["attention_mask"].to(model.device)
            test_name_mask = test_batch["name_mask"].to(model.device)
            test_value_mask = test_batch["value_mask"].to(model.device)

            test_outputs = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
            test_logits = test_outputs.logits

            shift_test_logits = test_logits[..., :-1, :].contiguous()
            shift_test_labels = test_input_ids[..., 1:].contiguous()
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
                all_data[data_count]["test_attr_loss"] = one_data_test_attr_loss
                all_data[data_count]["test_first_attr_loss"] = one_data_test_first_attr_loss
                all_data[data_count]["test_confidence"] = one_data_test_confidence
                all_data[data_count]["test_first_attr_confidence"] = one_data_test_first_attr_confidence
                
                data_count += 1
                
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
                all_test_attr_loss.append(one_data_test_attr_loss)
                all_test_first_attr_loss.append(one_data_test_first_attr_loss)
                all_test_confidence.append(one_data_test_confidence)
                all_test_first_attr_confidence.append(one_data_test_first_attr_confidence)
                

    # count histogram 0~1 each 0.1
    hist_datas = {
        "all_test_attr_loss": all_test_attr_loss,
        "all_test_first_attr_loss": all_test_first_attr_loss,
        "all_test_em_acc": all_test_em_acc,
        "all_test_confidence": all_test_confidence,
        "all_test_first_attr_confidence": all_test_first_attr_confidence,
    }
    for data_name, data in hist_datas.items():
        min_value = min(data)
        max_value = max(data)
        print(f"{data_name} min: {min_value}, max: {max_value}")
        hist, bin_edges = np.histogram(data, bins=np.arange(min_value, max_value, max_value / 10))
        print(f"{data_name} histogram:")
        for i in range(len(hist)):
            print(f"{bin_edges[i]:.4f} - {bin_edges[i+1]:.4f}: {hist[i]}")


    # print(f"Test Loss: {all_test_attr_loss}")
    # print(f"Test First Attribute Loss: {all_test_first_attr_loss}")
    # print(f"Test EM Accuracy: {all_test_em_acc}")
    
    # print(f"Average Test Loss: {np.mean(all_test_attr_loss)}")
main()