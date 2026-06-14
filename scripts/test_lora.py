import argparse
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hymt_sc.data import prompt_for  # noqa: E402


def resolve_existing_or_project_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    candidate = ROOT / path
    if candidate.exists():
        return str(candidate.resolve())
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PEFT LoRA generation test for Hy-MT-StarCitizen.")
    parser.add_argument("--model-name-or-path", default="models/hy-mt2-model")
    parser.add_argument("--adapter-path", default="outputs/hymt-starcitizen-lora")
    parser.add_argument("--direction", choices=["zh-en", "en-zh"], default="zh-en")
    parser.add_argument("--text", default="在洛维尔哪里可以买到飞船武器？")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = resolve_existing_or_project_path(args.model_name_or_path)
    adapter_path = resolve_existing_or_project_path(args.adapter_path)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.to(device)
    model.eval()

    prompt = prompt_for(args.direction, args.text)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    seconds = time.time() - start
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(new_ids, skip_special_tokens=True))
    print(
        f"device={device} input_tokens={inputs['input_ids'].shape[1]} "
        f"new_tokens={new_ids.shape[0]} seconds={seconds:.3f}"
    )


if __name__ == "__main__":
    main()
