import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

wandb.init(project='Hanghae99')
wandb.run.name = 'gpt-finetuning'

@dataclass
class Arguments:
    model_name_or_path: str = field(
        default="gpt2",
        metadata={"help": "HuggingFace hub에서 pre-trained 모델로 사용할 모델의 이름"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            'choices': ['auto', 'bfloat16', 'float16', 'float32'],
            "help": "우리 모델의 precision(data type이라고 이해하시면 됩니다)"
        }
    )

    dataset_name: str = field(
        default="wikitext",
        metadata={"help": "Fine-tuning으로 사용할 huggingface hub에서의 dataset 이름"}
    )
    dataset_config_name: str = field(
        default="wikitext-2-raw-v1",
        metadata={"help": "Fine-tuning으로 사용할 huggingface hub에서의 dataset configuration"}
    )
    block_size: int = field(
        default=1024,
        metadata={"help": "Fine-tuning에 사용할 input text의 길이"}
    )
    num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Data를 업로드하거나 전처리할 때 사용할 worker 숫자"}
    )
    validation_split_ratio: float = field(
        default=0.1,
        metadata={"help": "Validation 데이터 비율 (기본값: 10%)"}
    )

# 기본 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=1000,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    report_to=["wandb"],
)

# 커스텀 Arguments 파싱
parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

logger = logging.getLogger()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()  # log level을 INFO로 변경 

log_level = training_args.get_process_log_level()

# 우리가 가지고 있는 logger와 HuggingFace의 logger의 log level 설정
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

# 기타 HuggingFace logger option들을 설정
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.info(f"Training/evaluation parameters {training_args}")

# 데이터셋 로드 - 인자 확인 및 디버깅 추가
logger.info(f"Arguments: model_name_or_path={args.model_name_or_path}")
logger.info(f"Arguments: dataset_name={args.dataset_name}")
logger.info(f"Arguments: dataset_config_name={args.dataset_config_name}")

raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)

# 모델 및 토크나이저 로드
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output
    
# 데이터 토크나이징
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names
    )

max_pos_embeddings = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 1024
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)

def group_texts(examples):
    # 주어진 text들을 모두 concat 해줍니다. 
    # 예를 들어 examples = {'train': [['Hello!'], ['Yes, that is great!']]}이면 결과물은 {'train': ['Hello! Yes, that is great!']}가 됩니다.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    
    # 전체 길이를 측정합니다.
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    
    # block_size로 text를 쪼갭니다.
    # 예를 들어 block_size=3일 때 {'train': ['Hello! Yes, that is great!']}는
    # {'train': ['Hel', 'lo!', ' Ye', 's, ', 'tha', ...]}가 됩니다. 
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # Next token prediction이니 label은 자기 자신으로 설정합니다.
    result["labels"] = result["input_ids"].copy()
    return result
    
# 텍스트 그룹핑
with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )

# 데이터셋에서 train/validation split
train_dataset = lm_datasets["train"]

# Validation 데이터가 이미 있는지 확인하고, 없으면 train에서 split
if "validation" in lm_datasets:
    eval_dataset = lm_datasets["validation"]
    logger.info("Using existing validation split from dataset")
else:
    # train 데이터셋을 train/validation으로 분할
    split_dataset = train_dataset.train_test_split(
        test_size=args.validation_split_ratio,
        shuffle=True,
        seed=42
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    logger.info(f"Split train dataset: {len(train_dataset)} train samples, {len(eval_dataset)} validation samples")

# 평가 지표 정의
def compute_metrics(eval_pred):
    """평가 중 perplexity 계산"""
    predictions, labels = eval_pred
    # shift input_ids와 labels (causal language modeling)
    shift_predictions = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # loss 계산
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_predictions.view(-1, shift_predictions.size(-1)),
        shift_labels.view(-1)
    )
    
    # perplexity 계산
    perplexity = torch.exp(loss)
    
    return {"perplexity": perplexity.item()}

# Trainer 정의 (validation dataset 포함)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # validation dataset 추가
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,  # 평가 지표 추가
)

# 체크포인트 처리
checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)  # 만약 output_dir에 checkpoint가 남아있으면 이를 사용하고, 없으면 None이 return됩니다.
if training_args.resume_from_checkpoint is not None:  # output_dir이 아닌 다른 위치에서의 checkpoint를 resume_from_checkpoint로 지정할 수 있습니다.
    checkpoint = training_args.resume_from_checkpoint
else:  # 아니면 last_checkpoint로 checkpoint를 지정합니다.  
    checkpoint = last_checkpoint

# 훈련 시작
logger.info("Starting training...")
train_result = trainer.train(resume_from_checkpoint=checkpoint)

# 모델 저장
trainer.save_model()

# 훈련 메트릭 저장
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# 최종 평가 수행
logger.info("Running final evaluation...")
eval_result = trainer.evaluate()
trainer.log_metrics("eval", eval_result)
trainer.save_metrics("eval", eval_result)

# WandB에 최종 결과 로그
wandb.log({
    "final_train_loss": metrics.get("train_loss", 0),
    "final_eval_loss": eval_result.get("eval_loss", 0),
    "final_eval_perplexity": eval_result.get("eval_perplexity", 0),
})

logger.info("Training completed successfully!")
logger.info(f"Final train loss: {metrics.get('train_loss', 'N/A')}")
logger.info(f"Final eval loss: {eval_result.get('eval_loss', 'N/A')}")
logger.info(f"Final eval perplexity: {eval_result.get('eval_perplexity', 'N/A')}")