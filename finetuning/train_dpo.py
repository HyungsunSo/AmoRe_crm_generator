"""
DPO (Direct Preference Optimization) 학습 스크립트
./finetuning_data의 cycle_00.csv 파일을 토대로 1 사이클 DPO 학습 이후
./checkpoints에 Trainer 등의 메타 데이터를 저장하고 이후 resume을 통해 추가 학습할 수 있도록 함.
adapter의 경우 ./adapters에 저장
"""
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOTrainer

# 모델 및 경로 설정
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
CACHE_DIR = "../models"
OUTPUT_DIR = "./checkpoints"

# 데이터셋 경로 설정
DATA_DIR = "./finetuning_data"
CSV_FILE = os.path.join(DATA_DIR, "cycle_01.csv")

# 하이퍼파라미터 설정
PROMPT_LENGTH = 1024
MAX_SEQ_LENGTH = 1512


def load_dpo_dataset(csv_path: str):
    """CSV 파일에서 DPO 형식의 데이터셋을 로드합니다.
    
    CSV 형식: prompt, chosen, rejected 컬럼을 가져야 합니다.
    
    Args:
        csv_path: CSV 파일 경로
        
    Returns:
        train_dataset, eval_dataset: 학습 및 평가 데이터셋
    """
    # CSV 파일 로드
    dataset = load_dataset("csv", data_files=csv_path)
    dataset = dataset['train']
    
    # train/test split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    return dataset['train'], dataset['test']


def main():
    """DPO 학습 메인 함수"""
    
    # 1. 토크나이저 로드
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
    )
    
    # pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 패딩 사이드 설정 (DPO 학습에 유리)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    
    # 2. 데이터셋 로드
    print(f"데이터셋 로드 중: {CSV_FILE}")
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {CSV_FILE}")
    
    train_dataset, eval_dataset = load_dpo_dataset(CSV_FILE)
    print(f"학습 데이터: {len(train_dataset)}개, 평가 데이터: {len(eval_dataset)}개")
    
    # 3. BitsAndBytesConfig 설정 (QLoRA)
    print("BitsAndBytesConfig 설정 중...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 4. Flash Attention 설정
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16
    
    # 5. 모델 로드
    print("모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        use_cache=False,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        quantization_config=bnb_config,
        cache_dir=CACHE_DIR,
    )
    
    # 6. PEFT (LoRA) 설정
    print("PEFT 설정 중...")
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    
    # 7. TrainingArguments 설정
    print("TrainingArguments 설정 중...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=5e-5,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=100,
        save_total_limit=20,
        evaluation_strategy="steps",
        eval_steps=30000,
        bf16=True,
        tf32=True,
        push_to_hub=False,
        report_to="none",  # mlflow 대신 none으로 설정 (필요시 변경)
    )
    
    # 8. DPO 설정
    dpo_args = {
        "beta": 0.1,
        "loss_type": "sigmoid"
    }
    
    # 9. DPOTrainer 초기화
    print("DPOTrainer 초기화 중...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT 사용 시 None으로 설정
        peft_config=peft_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=PROMPT_LENGTH,
        beta=dpo_args["beta"],
        loss_type=dpo_args["loss_type"],
    )
    
    # 10. 학습 시작
    print("학습 시작...")
    trainer.train(resume_from_checkpoint=True)
    
    # 11. 모델 저장
    print("모델 저장 중...")
    trainer.save_model()
    print(f"모델이 저장되었습니다: {OUTPUT_DIR}")
    
    

if __name__ == "__main__":
    main()
