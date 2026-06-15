import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

EOS_TOKEN_ID = 120020
ROOT = Path(__file__).resolve().parents[1]


def resolve_existing_or_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def make_session(model_path: str, provider: str) -> ort.InferenceSession:
    provider_name = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}[
        provider
    ]
    available = ort.get_available_providers()
    if provider_name not in available:
        raise RuntimeError(f"{provider_name} not available. Available: {available}")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if provider == "dml":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(model_path, sess_options=options, providers=[provider_name])


def model_dims(session: ort.InferenceSession) -> tuple[int, int, int, np.dtype]:
    key_inputs = [item for item in session.get_inputs() if item.name.endswith(".key")]
    num_layers = len(key_inputs)
    shape = key_inputs[0].shape
    num_kv_heads = int(shape[1])
    head_size = int(shape[3])
    dtype = np.float16 if key_inputs[0].type == "tensor(float16)" else np.float32
    return num_layers, num_kv_heads, head_size, dtype


def empty_past(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    num_layers, num_kv_heads, head_size, dtype = model_dims(session)
    shape = (1, num_kv_heads, 0, head_size)
    past = {}
    for layer in range(num_layers):
        past[f"past_key_values.{layer}.key"] = np.zeros(shape, dtype=dtype)
        past[f"past_key_values.{layer}.value"] = np.zeros(shape, dtype=dtype)
    return past


def greedy_generate(session: ort.InferenceSession, input_ids: list[int], max_new_tokens: int) -> tuple[list[int], float]:
    num_layers, _, _, _ = model_dims(session)
    past = empty_past(session)
    token_chunk = np.asarray([input_ids], dtype=np.int64)
    past_length = 0
    generated = []
    start = time.time()

    for _ in range(max_new_tokens):
        sequence_length = token_chunk.shape[1]
        total_length = past_length + sequence_length
        feeds = {
            "input_ids": token_chunk,
            "attention_mask": np.ones((1, total_length), dtype=np.int64),
            "position_ids": np.arange(past_length, total_length, dtype=np.int64).reshape(1, sequence_length),
            **past,
        }
        outputs = session.run(None, feeds)
        next_token = int(np.argmax(outputs[0][0, -1, :]))
        generated.append(next_token)
        past = {}
        for layer in range(num_layers):
            past[f"past_key_values.{layer}.key"] = outputs[1 + layer]
            past[f"past_key_values.{layer}.value"] = outputs[1 + num_layers + layer]
        past_length = total_length
        token_chunk = np.asarray([[next_token]], dtype=np.int64)
        if next_token == EOS_TOKEN_ID:
            break

    return generated, time.time() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure ONNX Runtime inference for Hy-MT-StarCitizen.")
    parser.add_argument("--onnx-dir", default="outputs/onnx-q4acc4-b128")
    parser.add_argument("--filename", default="model_q4acc4_b128.onnx")
    parser.add_argument("--tokenizer-dir", default="models/hy-mt2-model")
    parser.add_argument("--provider", choices=["cpu", "cuda", "dml"], default="cpu")
    parser.add_argument("--direction", choices=["zh-en", "en-zh"], default="zh-en")
    parser.add_argument("--text", default="在洛维尔哪里可以买到飞船武器？")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_dir = resolve_existing_or_project_path(args.onnx_dir)
    tokenizer_dir = resolve_existing_or_project_path(args.tokenizer_dir)
    model_path = onnx_dir / args.filename
    session = make_session(str(model_path), args.provider)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    if args.direction == "zh-en":
        prompt = f"Translate the following Star Citizen localization text into English. Only output the translation:\n\n{args.text}"
    else:
        prompt = f"将以下《星际公民》本地化文本翻译为简体中文。只输出翻译结果：\n\n{args.text}"
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt_text, add_special_tokens=False)
    input_ids = [int(token_id) for token_id in inputs["input_ids"]]
    new_ids, seconds = greedy_generate(session, input_ids, args.max_new_tokens)
    print(tokenizer.decode(new_ids, skip_special_tokens=True))
    print(f"provider={session.get_providers()[0]} input_tokens={len(input_ids)} new_tokens={len(new_ids)} seconds={seconds:.3f}")


if __name__ == "__main__":
    main()
