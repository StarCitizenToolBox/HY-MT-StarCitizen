# HY-MT-StarCitizen

Hy-MT2 based Star Citizen zh/en localization fine-tuning and ONNX Runtime export project.

This project is intentionally self-contained for later Rust `ort` integration:

- Dataset update does not require `genai`.
- Training uses Hugging Face Transformers + PEFT LoRA.
- ONNX export defaults to the fastest local CPU graph tested: q4 `MatMulNBits` with `accuracy_level=4`, `block_size=128`, and standard ONNX `Attention`.
- Runtime inference uses plain `onnxruntime` only, not `onnxruntime-genai` or PyTorch.

## Sources

The dataset updater pulls from:

- English game localization: https://github.com/Dymerz/StarCitizen-Localization/blob/main/data/Localization/english/global.ini
- Simplified Chinese game localization: https://github.com/StarCitizenToolBox/LocalizationData/blob/main/chinese_(simplified)/global.ini
- ScWeb translations: https://github.com/CxJuice/ScWeb_Chinese_Translate

The category classifier is adapted from:

- https://github.com/StarCitizenToolBox/Opus-MT-StarCitizen

Priority categories are `location`, `vehicle`, `item`, `subtitle`, and `mission`; everything else is kept as `other`. ScWeb rows are marked as `scweb`.

## Layout

```text
HY-MT-StarCitizen/
  config/default.json
  data/raw/
  data/processed/
  models/
  scripts/download_model.py
  scripts/update_dataset.py
  scripts/train_lora.py
  scripts/test_lora.py
  scripts/export_onnx_q4f16.py
  scripts/convert_gqa_to_attention.py
  scripts/inspect_onnx.py
  scripts/ort_infer.py
  src/hymt_sc/
```

## Setup

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

The scripts resolve project-relative paths, so they can be called from either the workspace root or the project directory.

Download or refresh the base model inside the project:

```powershell
python scripts\download_model.py
```

The default model directory is `models\hy-mt2-model`. It is ignored by git, so the 4GB model weights stay local and are not committed.

## Update Dataset

```powershell
python scripts\update_dataset.py
```

Current generated dataset:

- `global_ini`: 78,268 valid aligned pairs
- `scweb`: 10,994 valid pairs
- total source pairs: 89,262
- per-direction train rows: 87,477
- per-direction eval rows: 1,785
- mixed bidirectional train rows: 174,954
- mixed bidirectional eval rows: 3,570

Outputs:

- `data/processed/train.zh-en.jsonl`
- `data/processed/eval.zh-en.jsonl`
- `data/processed/train.en-zh.jsonl`
- `data/processed/eval.en-zh.jsonl`
- `data/processed/train.mixed.jsonl`
- `data/processed/eval.mixed.jsonl`
- `data/processed/pairs.tsv`
- `data/processed/metadata.json`

Each JSONL row contains Hy-MT2 chat-style `messages`, plus `key`, `direction`, `category`, `source_type`, `source`, and `target`.

## Train LoRA

Tiny smoke test:

```powershell
python scripts\train_lora.py `
  --model-name-or-path models\hy-mt2-model `
  --train-file data\processed\train.zh-en.jsonl `
  --eval-file data\processed\eval.zh-en.jsonl `
  --output-dir outputs\tiny-lora-smoke `
  --tiny-random-smoke `
  --max-steps 1 `
  --max-seq-length 128 `
  --batch-size 1 `
  --gradient-accumulation-steps 1 `
  --lora-rank 4 `
  --lora-alpha 8
```

Real Hy-MT2 1-step smoke:

```powershell
python scripts\train_lora.py `
  --model-name-or-path models\hy-mt2-model `
  --train-file data\processed\train.zh-en.jsonl `
  --eval-file data\processed\eval.zh-en.jsonl `
  --output-dir outputs\hymt-starcitizen-lora-smoke `
  --max-steps 1 `
  --max-seq-length 128 `
  --batch-size 1 `
  --gradient-accumulation-steps 1 `
  --lora-rank 4 `
  --lora-alpha 8 `
  --bf16 `
  --gradient-checkpointing
```

Verified smoke result on RTX 4090:

- trainable params: 1,703,936
- all params: 1,792,784,384
- trainable ratio: 0.0950%
- 1-step train loss: 2.651

For normal fine-tuning, use `config/default.json` or override `--max-steps`, `--max-seq-length`, LoRA rank, and output path.

The checked default run uses mixed bidirectional data:

```powershell
python scripts\train_lora.py
```

Verified 6000-step run on RTX 4090:

- output: `outputs/hymt-starcitizen-lora`
- trainable params: 13,631,488
- all params: 1,804,711,936
- trainable ratio: 0.7553%
- final eval loss: 0.8874
- final checkpoint: `outputs/hymt-starcitizen-lora/checkpoint-6000`

The default training config uses LoRA rank 32, effective batch size 16, cosine learning-rate schedule, and checkpoint/eval every 500 steps.

## Test LoRA

```powershell
python scripts\test_lora.py `
  --model-name-or-path models\hy-mt2-model `
  --adapter-path outputs\hymt-starcitizen-lora `
  --direction en-zh `
  --text "Where can I buy ship weapons in Lorville?" `
  --max-new-tokens 64
```

Verified adapter outputs:

- zh-en: `Where can I buy ship weapons in Lorville?`
- en-zh: `我在罗威尔哪里能买到飞船武器？`
- zh-en long: `If your ship was destroyed near Orison, check your insurance status and then head to the nearest terminal to reapply for a vehicle.`
- en-zh long: `在接单之前，请确保你的飞船有足够的空间，并且目标前哨站目前没有敌对势力。`

## Export ONNX

Smoke export, using only one hidden layer:

```powershell
python scripts\export_onnx_q4f16.py `
  --model-dir models\hy-mt2-model `
  --output-dir outputs\onnx-q4acc4-b128-smoke `
  --num-hidden-layers 1
```

Full export with LoRA adapter. The default int4 export is `accuracy_level=4`, `block_size=128`, standard ONNX `Attention`, and writes `model_q4acc4_b128.onnx`:

```powershell
python scripts\export_onnx_q4f16.py `
  --model-dir models\hy-mt2-model `
  --adapter-path outputs\hymt-starcitizen-lora `
  --execution-provider cpu
```

The exporter uses `onnxruntime-genai` only as a conversion tool. The produced model is a standard ONNX graph that can be loaded with plain ONNX Runtime.

Inspect quantization:

```powershell
python scripts\inspect_onnx.py outputs\onnx-q4acc4-b128\model_q4acc4_b128.onnx
```

Expected default indicators include `MatMulNBits` nodes with `bits=4`, `block_size=128`, and `accuracy_level=4`.

Verified full adapter export:

- `model_q4acc4_b128.onnx`: 671,824 bytes
- `model_q4acc4_b128.onnx.data`: 1,101,842,432 bytes
- nodes: 1,075
- `MatMulNBits`: 481, with `bits=4`, `block_size=128`, `accuracy_level=4`
- `Attention`: 32
- `GroupQueryAttention`: 0

## Recommended ONNX Targets

Local benchmark hardware:

- CPU: AMD Ryzen 7 7800X3D, 8 cores / 16 threads
- GPU: NVIDIA GeForce RTX 4090, 24 GB VRAM, driver 596.49
- Runtime: ONNX Runtime 1.26.0 for CPU/CUDA tests; ONNX Runtime DirectML 1.24.4 for DirectML tests

CPU default:

- export args: `--execution-provider cpu --attention-op standard --int4-accuracy-level 4 --int4-block-size 128`
- output dir: `outputs/onnx-q4acc4-b128`
- filename: `model_q4acc4_b128.onnx`
- size: about 1.10 GB external data
- local CPU speed: about 27-33 tok/s on the tested translation prompts
- CUDA status: loads, but generates invalid repeated text, so use this target for CPU only

CUDA fastest correct quantized target:

- export args: `--execution-provider cuda --attention-op standard --int4-accuracy-level 2 --int4-block-size 32 --unquantized-lm-head`
- default filename: `model_q4f16_hybrid.onnx`
- graph shape: q4f16 `MatMulNBits` body, standard `Attention`, unquantized fp16 `lm_head`, and shared embeddings disabled
- local CUDA speed: about 61-69 tok/s on the tested translation prompts
- tradeoff: larger than the CPU default artifact, about 1.93 GB external data in the local test

DirectML:

- best correct q4 variants reached only about 2-3 tok/s locally
- `GroupQueryAttention` DML q4 variants generated invalid text in testing
- treat DML as a compatibility fallback, not the performance target

## Pure ORT CPU Inference

```powershell
python scripts\ort_infer.py `
  --onnx-dir outputs\onnx-q4acc4-b128 `
  --filename model_q4acc4_b128.onnx `
  --tokenizer-dir models\hy-mt2-model `
  --provider cpu `
  --direction zh-en `
  --text "在洛维尔哪里可以买到飞船武器？" `
  --max-new-tokens 64
```

The inference script:

- imports `onnxruntime`, not `onnxruntime-genai`
- does not require PyTorch
- manually manages KV cache inputs and outputs
- uses greedy decoding
- prints token count and elapsed seconds for TPS calculation

Verified CPU outputs with the default q4acc4/b128 model:

- zh-en short: `Where can I get ship weapons in Lorville?`, 11 tokens / 0.403s, about 27.3 tok/s
- en-zh short: `我在罗威尔哪里可以买到舰船武器？`, 11 tokens / 0.406s, about 27.1 tok/s
- zh-en long: `If your ship was destroyed near Orison, please check your insurance status and head to the nearest terminal to reapply for the vehicle.`, 28 tokens / 0.844s, about 33.2 tok/s
- en-zh long: `接受合约前，请确保你的飞船有足够的货舱空间，并且目标前哨站目前未被敌对势力控制。`, 24 tokens / 0.774s, about 31.0 tok/s

## Attention Op Notes

As of ONNX 1.22, the standard `Attention` op exists in the main domain and supports MHA, GQA, MQA, and KV-cache update cases. The default export monkey-patches the current ORT GenAI Hunyuan builder to emit standard-domain `Attention` instead of ORT contrib `GroupQueryAttention`.

For this model, standard `Attention` is the checked default because it is the only path that worked across the CPU q4acc4/b128 and CUDA hybrid experiments. `GroupQueryAttention` DML q4 testing generated invalid text.

## Quantization Notes

The ONNX export here targets ORT's weight-only int4 path (`MatMulNBits`). Stock ONNX Runtime has direct support for this style of 4-bit LLM weight quantization. Lower-than-q4 formats such as q2 or GGUF-style 1.25-bit are not a drop-in ORT ONNX export target; they would require a custom operator/runtime path or a different backend. For Rust `ort`, q4 `MatMulNBits` is the practical CPU-compatible target.
