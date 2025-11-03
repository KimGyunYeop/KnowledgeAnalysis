
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import json
import random
import copy
from utils import ATTRIBUTE_LIST

class RandomBioDataset(Dataset):
    def __init__(self, data_path, template_path, tokenizer, mode="train", add_augmented_prefix=False, prefix_mode="repeat"):
        if type(data_path) is str:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        elif type(data_path) is list:
            self.data = data_path
            
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
        
        self.add_augmented_prefix = add_augmented_prefix
        self.prefix_mode = prefix_mode
        self.mode = mode

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
        custom_label = input_ids.clone()

        if self.add_augmented_prefix:
            if self.prefix_mode == "repeat":
                prefix_input_ids = input_ids.clone()
            if self.prefix_mode == "metadata":
                #without train_template in item
                prefix = copy.deepcopy(item)
                prefix.pop("train_tamplates")
                prefix = str(prefix)
                prefix_input_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
                prefix_input_ids = torch.tensor(prefix_input_ids, dtype=torch.long)

            input_ids = torch.cat([prefix_input_ids, input_ids], dim=-1)
            attention_mask = torch.cat([torch.ones_like(prefix_input_ids), attention_mask], dim=-1)
            custom_label = torch.cat([torch.ones_like(prefix_input_ids)*-100, custom_label], dim=-1)
            name_mask = torch.cat([torch.zeros_like(prefix_input_ids), name_mask], dim=-1)
            value_mask = torch.cat([torch.zeros_like(prefix_input_ids), value_mask], dim=-1)
            
            # input_ids = torch.cat([input_ids, prefix_input_ids], dim=-1)
            # attention_mask = torch.cat([attention_mask, torch.ones_like(prefix_input_ids)], dim=-1)
            # custom_label = torch.cat([custom_label, torch.ones_like(prefix_input_ids)*-100], dim=-1)
            # name_mask = torch.cat([name_mask, torch.zeros_like(prefix_input_ids)], dim=-1)
            # value_mask = torch.cat([value_mask, torch.zeros_like(prefix_input_ids)], dim=-1)

            # print(f"[{self.mode}]input_data: {self.tokenizer.decode(input_ids, skip_special_tokens=False)}")

        return {"input_ids": input_ids, "attention_mask": attention_mask, "name_mask": name_mask, "value_mask": value_mask, "custom_label": custom_label}

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        name_mask = [item["name_mask"] for item in batch]
        value_mask = [item["value_mask"] for item in batch]
        custom_label = [item["custom_label"] for item in batch]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        name_mask = torch.nn.utils.rnn.pad_sequence(name_mask, batch_first=True, padding_value=0)
        value_mask = torch.nn.utils.rnn.pad_sequence(value_mask, batch_first=True, padding_value=0)
        custom_label = torch.nn.utils.rnn.pad_sequence(custom_label, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "name_mask": name_mask, "value_mask": value_mask, "custom_label": custom_label}