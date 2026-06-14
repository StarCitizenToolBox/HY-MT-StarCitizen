import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IGNORE_INDEX = -100


def resolve_project_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    candidate = ROOT / path
    if candidate.exists():
        return str(candidate.resolve())
    return value


def resolve_output_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


class SFTJsonlDataset(Dataset):
    def __init__(self, data_file: str, tokenizer: Any, max_seq_length: int):
        self.rows = [line for line in Path(data_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.assistant_id = tokenizer.convert_tokens_to_ids("<｜hy_Assistant｜>")
        self.eos_id = tokenizer.convert_tokens_to_ids("<｜hy_place▁holder▁no▁2｜>")
        if self.assistant_id is None or self.assistant_id < 0:
            raise ValueError("Cannot find Hy-MT2 assistant token in tokenizer.")
        if self.eos_id is None or self.eos_id < 0:
            raise ValueError("Cannot find Hy-MT2 EOS token in tokenizer.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = json.loads(self.rows[index])
        tokens = self.tokenizer.apply_chat_template(item["messages"], tokenize=True, return_dict=False)
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        input_ids = torch.tensor(tokens[: self.max_seq_length], dtype=torch.long)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        starts = (input_ids == self.assistant_id).nonzero(as_tuple=True)[0].tolist()
        ends = (input_ids == self.eos_id).nonzero(as_tuple=True)[0].tolist()
        for start, end in zip(starts, ends):
            if start <= end:
                labels[start : end + 1] = input_ids[start : end + 1]
        return {"input_ids": input_ids, "labels": labels}


@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(self, instances: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in instances],
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return {"input_ids": input_ids, "attention_mask": input_ids.ne(pad_token_id), "labels": labels}


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_tiny_config(model_dir: str):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 1
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.attention_head_dim = 16
    config.max_position_embeddings = 512
    config.torch_dtype = "float32"
    if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
        config.rope_parameters["rope_theta"] = config.rope_parameters.get("rope_theta", 10000.0)
    return config


def parse_args() -> argparse.Namespace:
    defaults = load_config(str(ROOT / "config" / "default.json"))
    parser = argparse.ArgumentParser(description="LoRA fine-tune Hy-MT2 for Star Citizen localization.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-name-or-path", default=resolve_project_path(defaults.get("model_name_or_path", "tencent/Hy-MT2-1.8B")))
    parser.add_argument("--train-file", default=resolve_project_path(defaults.get("train_file", "data/processed/train.zh-en.jsonl")))
    parser.add_argument("--eval-file", default=resolve_project_path(defaults.get("eval_file", "data/processed/eval.zh-en.jsonl")))
    parser.add_argument("--output-dir", default=resolve_output_path(defaults.get("output_dir", "outputs/hymt-starcitizen-lora")))
    parser.add_argument("--max-seq-length", type=int, default=defaults.get("max_seq_length", 512))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 1))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults.get("gradient_accumulation_steps", 8))
    parser.add_argument("--learning-rate", type=float, default=defaults.get("learning_rate", 2e-4))
    parser.add_argument("--max-steps", type=int, default=defaults.get("max_steps", 200))
    parser.add_argument("--warmup-ratio", type=float, default=defaults.get("warmup_ratio", 0.0))
    parser.add_argument("--lr-scheduler-type", default=defaults.get("lr_scheduler_type", "linear"))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 0.0))
    parser.add_argument("--logging-steps", type=int, default=defaults.get("logging_steps", 1))
    parser.add_argument("--eval-steps", type=int, default=defaults.get("eval_steps", 0))
    parser.add_argument("--save-steps", type=int, default=defaults.get("save_steps", 0))
    parser.add_argument("--save-total-limit", type=int, default=defaults.get("save_total_limit", 2))
    parser.add_argument("--dataloader-num-workers", type=int, default=defaults.get("dataloader_num_workers", 0))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--lora-rank", type=int, default=defaults.get("lora_rank", 16))
    parser.add_argument("--lora-alpha", type=int, default=defaults.get("lora_alpha", 32))
    parser.add_argument("--lora-dropout", type=float, default=defaults.get("lora_dropout", 0.05))
    parser.add_argument("--bf16", action="store_true", default=defaults.get("bf16", False))
    parser.add_argument("--gradient-checkpointing", action="store_true", default=defaults.get("gradient_checkpointing", False))
    parser.add_argument("--resume-from-checkpoint", default=defaults.get("resume_from_checkpoint"))
    parser.add_argument("--tiny-random-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                setattr(args, attr, value)
    args.model_name_or_path = resolve_project_path(args.model_name_or_path)
    args.train_file = resolve_project_path(args.train_file)
    args.eval_file = resolve_project_path(args.eval_file)
    args.output_dir = resolve_output_path(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.tiny_random_smoke:
        model = AutoModelForCausalLM.from_config(make_tiny_config(args.model_name_or_path), trust_remote_code=True)
    else:
        dtype = torch.bfloat16 if args.bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SFTJsonlDataset(args.train_file, tokenizer, args.max_seq_length)
    eval_dataset = SFTJsonlDataset(args.eval_file, tokenizer, args.max_seq_length) if Path(args.eval_file).exists() else None
    eval_steps = args.eval_steps if args.eval_steps and eval_dataset is not None else None
    save_steps = args.save_steps or max(args.max_steps, 1)
    warmup_steps = int(args.max_steps * args.warmup_ratio) if args.warmup_ratio else 0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
        remove_unused_columns=False,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
