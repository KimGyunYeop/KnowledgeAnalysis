# KnowledgeAnalysis

### data processing
```
bash data_download_script.sh
python bid_data_generate.py
python bio_data_generate_other.py 
```
bash data_download_script.sh --> 바이오 데이터의 사전 데이터 수집
python bio_data_generate.py --> 사전학습 데이터셋 생성
python bio_data_generate_other.py --> fine-tuning 데이터셋 생성

## Pre-Train
```
bash train.sh
```
train.sh --> 사전학습

### fine-tuning
일반 fine-tuning
```
bash finetuning.sh
```

데이터 분리 후 fine-tuning (loss기준 5분리, conf기준 5분리 등)
```
bash finetuning_splited.sh
```
