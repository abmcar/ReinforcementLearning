"""LoRA / QLoRA SFT training pipeline for JOB-13."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "lora_qwen7b.yaml"
DEFAULT_LOSS_FIGURE = PROJECT_ROOT / "docs" / "figures" / "JOB-13-loss.png"


@dataclass
class ModelSection:
    model_name_or_path: str
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    trust_remote_code: bool = True


@dataclass
class DataSection:
    train_file: str
    eval_file: str
    max_train_samples: Optional[int] = 1024
    max_eval_samples: Optional[int] = 256
    max_length: int = 1024
    preprocessing_num_workers: int = 4


@dataclass
class TrainingSection:
    output_dir: str
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    bf16: bool = True
    report_to: str = "none"
    seed: int = 42


@dataclass
class LoraSection:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(default_factory=list)


@dataclass
class SFTConfig:
    model: ModelSection
    data: DataSection
    training: TrainingSection
    lora: LoraSection


def _load_yaml_config(path: Path) -> SFTConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SFTConfig(
        model=ModelSection(**raw["model"]),
        data=DataSection(**raw["data"]),
        training=TrainingSection(**raw["training"]),
        lora=LoraSection(**raw["lora"]),
    )


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[key]


def _load_tokenizer(model_name_or_path: str, trust_remote_code: bool) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(config: SFTConfig) -> AutoModelForCausalLM:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    distributed = local_rank >= 0
    torch_dtype = _resolve_dtype(config.model.torch_dtype)
    quantization_config = None
    if config.model.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    device_map: str | dict[str, int]
    if distributed:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=config.model.trust_remote_code,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    model.config.use_cache = False
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if config.model.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
        target_modules=config.lora.target_modules,
    )
    return get_peft_model(model, lora_config)


def _jsonl_to_dataset(path: Path, sample_limit: Optional[int]) -> Dataset:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if sample_limit is not None and idx >= sample_limit:
                break
            records.append(json.loads(line))
    return Dataset.from_list(records)


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    system: str,
    user: str,
    response: Optional[str] = None,
) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=response is None,
    )


def _tokenize_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_text = _apply_chat_template(
        tokenizer,
        system=str(example.get("system", "")),
        user=str(example.get("user", "")),
        response=None,
    )
    full_text = _apply_chat_template(
        tokenizer,
        system=str(example.get("system", "")),
        user=str(example.get("user", "")),
        response=str(example["response"]),
    )
    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = list(full_tokens["input_ids"])
    attention_mask = list(full_tokens["attention_mask"])
    labels = input_ids.copy()
    prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SupervisedDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_masks.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _prepare_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    num_proc: int,
) -> Dataset:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        num_proc = 1
    mapped = dataset.map(
        lambda row: _tokenize_example(row, tokenizer, max_length),
        remove_columns=dataset.column_names,
        num_proc=max(1, num_proc),
        desc="Tokenizing SFT dataset",
    )
    return mapped


def _build_training_args(config: SFTConfig) -> TrainingArguments:
    training = config.training
    return TrainingArguments(
        output_dir=training.output_dir,
        num_train_epochs=training.num_train_epochs,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        warmup_ratio=training.warmup_ratio,
        lr_scheduler_type=training.lr_scheduler_type,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        gradient_checkpointing=training.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        logging_steps=training.logging_steps,
        evaluation_strategy="steps",
        eval_steps=training.eval_steps,
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        bf16=training.bf16,
        report_to=[] if training.report_to == "none" else [training.report_to],
        seed=training.seed,
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )


def _plot_loss_curves(log_history: list[dict[str, Any]], output_path: Path) -> None:
    train_points = [(row["step"], row["loss"]) for row in log_history if "loss" in row]
    eval_points = [(row["step"], row["eval_loss"]) for row in log_history if "eval_loss" in row]
    if not train_points and not eval_points:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if train_points:
        x, y = zip(*train_points)
        plt.plot(x, y, label="train_loss")
    if eval_points:
        x, y = zip(*eval_points)
        plt.plot(x, y, label="eval_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("JOB-13 LoRA SFT Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def train(config_path: Path) -> None:
    is_main_process = int(os.environ.get("RANK", "0")) == 0
    config = _load_yaml_config(config_path)
    tokenizer = _load_tokenizer(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    model = _load_model(config)

    train_dataset = _jsonl_to_dataset(
        PROJECT_ROOT / config.data.train_file,
        config.data.max_train_samples,
    )
    eval_dataset = _jsonl_to_dataset(
        PROJECT_ROOT / config.data.eval_file,
        config.data.max_eval_samples,
    )

    train_dataset = _prepare_dataset(
        train_dataset,
        tokenizer,
        config.data.max_length,
        config.data.preprocessing_num_workers,
    )
    eval_dataset = _prepare_dataset(
        eval_dataset,
        tokenizer,
        config.data.max_length,
        config.data.preprocessing_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=_build_training_args(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )

    output_dir = PROJECT_ROOT / config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.train()
    trainer.save_model()
    if is_main_process:
        tokenizer.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    if is_main_process:
        (output_dir / "eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _plot_loss_curves(trainer.state.log_history, DEFAULT_LOSS_FIGURE)


def _load_generation_model(
    base_model_path: str,
    adapter_path: Optional[str],
    trust_remote_code: bool,
) -> tuple[PreTrainedTokenizerBase, AutoModelForCausalLM]:
    tokenizer = _load_tokenizer(base_model_path, trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate(
    config_path: Path,
    prompt_text: str,
    adapter_path: Optional[str],
    baseline_only: bool,
) -> None:
    config = _load_yaml_config(config_path)
    tokenizer, model = _load_generation_model(
        config.model.model_name_or_path,
        None if baseline_only else adapter_path,
        config.model.trust_remote_code,
    )
    formatted = _apply_chat_template(tokenizer, "", prompt_text, None)
    encoded = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=16,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][encoded["input_ids"].shape[1] :]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JOB-13 LoRA / QLoRA SFT pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(PROJECT_ROOT)))

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(PROJECT_ROOT)))
    gen_parser.add_argument("--adapter-path", default=None)
    gen_parser.add_argument("--prompt-text", required=True)
    gen_parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = PROJECT_ROOT / args.config
    if args.command == "train":
        train(config_path)
        return
    generate(
        config_path=config_path,
        prompt_text=args.prompt_text,
        adapter_path=args.adapter_path,
        baseline_only=bool(args.baseline_only),
    )


if __name__ == "__main__":
    main()
