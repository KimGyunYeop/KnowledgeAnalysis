import argparse
import os
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Graph Generation for Knowledge Analysis")

parser.add_argument("--output_dir", type=str, default="figure", help="Directory to save the output JSON file")
parser.add_argument("--target_folder", type=str, default="results_ft/plain_transformer", help="Path to the output JSON file")

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

all_folders = os.listdir(args.target_folder)
all_folders = sorted(all_folders)
base_exp_folders = []

for i in all_folders:
    if "-" not in i:
        base_exp_folders.append(i)

split_type = ["attr_loss", "first_attr_loss", "attr_confidence", "first_attr_confidence"]

for st in split_type:
    for i in base_exp_folders:
        with open(os.path.join(args.target_folder, i, "test_attr_loss_dict.json"), "r") as f:
            base_test_loss = json.load(f)
        with open(os.path.join(args.target_folder, i, "original_test_attr_loss_dict.json"), "r") as f:
            base_original_test_loss = json.load(f)
        
        # Plotting the graph
        plt.figure(figsize=(6, 6))
        plt.xlabel('pre-train Attribute Loss')
        plt.ylabel('fine-tune Attribute Loss')
        plt.xscale('log')  # x축을 로그 스케일로 설정
        plt.yscale('log')  # y축을 로그 스케일로 설정
        plt.title(f'Knowledge Collapse - split by {st}')
        plt.plot(list(base_original_test_loss.values())[1:], list(base_test_loss.values())[1:], label="base")
        
        save_flg = False
        for af in all_folders:
            if i+"_splited_"+st in af:
                with open(os.path.join(args.target_folder, af, "test_attr_loss_dict.json"), "r") as f:
                    test_loss = json.load(f)
                with open(os.path.join(args.target_folder, af, "original_test_attr_loss_dict.json"), "r") as f:
                    original_test_loss = json.load(f)

                # Plotting the graph
                plt.plot(list(original_test_loss.values())[1:], list(test_loss.values())[1:], label=af.split("_")[-1])
                save_flg = True
                
        if save_flg:
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, f"{i.split('/')[-1]}_{st}.png"))



