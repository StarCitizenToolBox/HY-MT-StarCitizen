# HY-MT-StarCitizen

Hy-MT2 based Star Citizen zh/en localization fine-tuning and ONNX Runtime export project.

This project is intentionally self-contained for later Rust `ort` integration:

- Dataset update does not require `genai`.
- Training uses Hugging Face Transformers + PEFT LoRA.
- ONNX export produces a q4f16 graph.
- Runtime inference uses plain `onnxruntime` only, not `onnxruntime-genai`.

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

From `P:\AI_WorkSpace`:

```powershell
.\.venv_hymt2\Scripts\python.exe -m pip install -r HY-MT-StarCitizen\requirements.txt
```

The scripts resolve project-relative paths, so they can be called from either the workspace root or the project directory.

Download or refresh the base model inside the project:

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\download_model.py
```

The default model directory is `HY-MT-StarCitizen\models\hy-mt2-model`. It is ignored by git, so the 4GB model weights stay local and are not committed.

## Update Dataset

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\update_dataset.py
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
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\train_lora.py `
  --model-name-or-path models\hy-mt2-model `
  --train-file HY-MT-StarCitizen\data\processed\train.zh-en.jsonl `
  --eval-file HY-MT-StarCitizen\data\processed\eval.zh-en.jsonl `
  --output-dir HY-MT-StarCitizen\outputs\tiny-lora-smoke `
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
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\train_lora.py `
  --model-name-or-path models\hy-mt2-model `
  --train-file HY-MT-StarCitizen\data\processed\train.zh-en.jsonl `
  --eval-file HY-MT-StarCitizen\data\processed\eval.zh-en.jsonl `
  --output-dir HY-MT-StarCitizen\outputs\hymt-starcitizen-lora-smoke `
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
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\train_lora.py
```

Verified 6000-step run on RTX 4090:

- output: `outputs/hymt-starcitizen-lora`
- trainable params: 13,631,488
- all params: 1,804,711,936
- trainable ratio: 0.7553%
- final eval loss: 0.8874
- final checkpoint: `outputs/hymt-starcitizen-lora/checkpoint-6000`

The default config now targets a higher-quality q4f16 ONNX export: LoRA rank 32, effective batch size 16, cosine learning-rate schedule, and checkpoint/eval every 500 steps.

## Test LoRA

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\test_lora.py `
  --model-name-or-path models\hy-mt2-model `
  --adapter-path HY-MT-StarCitizen\outputs\hymt-starcitizen-lora `
  --direction en-zh `
  --text "Where can I buy ship weapons in Lorville?" `
  --max-new-tokens 64
```

Verified adapter outputs:

- zh-en: `Where can I buy ship weapons in Lorville?`
- en-zh: `我在罗威尔哪里能买到飞船武器？`
- zh-en long: `If your ship was destroyed near Orison, check your insurance status and then head to the nearest terminal to reapply for a vehicle.`
- en-zh long: `在接单之前，请确保你的飞船有足够的空间，并且目标前哨站目前没有敌对势力。`

## Export ONNX q4f16

Smoke export, using only one hidden layer:

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\export_onnx_q4f16.py `
  --model-dir models\hy-mt2-model `
  --output-dir outputs\onnx-q4f16-smoke `
  --num-hidden-layers 1
```

Full export with LoRA adapter:

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\export_onnx_q4f16.py `
  --model-dir models\hy-mt2-model `
  --adapter-path HY-MT-StarCitizen\outputs\hymt-starcitizen-lora `
  --output-dir outputs\onnx-q4f16 `
  --execution-provider cpu
```

The exporter uses `onnxruntime-genai` only as a conversion tool. The produced model is a standard ONNX graph that can be loaded with plain ONNX Runtime.

The same exporter can produce unquantized models by changing `--precision`:

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\export_onnx_q4f16.py `
  --model-dir models\hy-mt2-model `
  --adapter-path HY-MT-StarCitizen\outputs\hymt-starcitizen-lora `
  --output-dir outputs\onnx-fp16-attention-direct `
  --precision fp16 `
  --execution-provider cpu `
  --attention-op standard
```

For non-int4 exports, the default filename changes to `model_<precision>.onnx`, for example `model_fp16.onnx`.

Inspect quantization:

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\inspect_onnx.py HY-MT-StarCitizen\outputs\onnx-q4f16\model_q4f16.onnx
```

Expected q4f16 indicators include `MatMulNBits` nodes with `bits=4`.

Verified full adapter export:

- `model_q4f16.onnx`: 671,425 bytes
- `model_q4f16.onnx.data`: 1,262,667,776 bytes
- nodes: 1,075
- `MatMulNBits`: 481, with `bits=4`, `block_size=32`, `accuracy_level=2`
- `com.microsoft:GroupQueryAttention`: 32

## Standard ONNX Attention

ORT 1.26 can run standard-domain ONNX `Attention` on CPU for this graph shape. The exporter can directly generate standard `Attention` nodes instead of ORT `GroupQueryAttention` nodes:

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\export_onnx_q4f16.py `
  --model-dir models\hy-mt2-model `
  --adapter-path HY-MT-StarCitizen\outputs\hymt-starcitizen-lora `
  --output-dir outputs\onnx-q4f16-attention-direct `
  --execution-provider cpu `
  --attention-op standard
```

This path monkey-patches the current ORT GenAI Hunyuan builder at export time:

- the actual attention node is emitted as standard-domain `Attention`
- default ONNX opset is raised to 23
- q4f16 weight-only quantization still uses ORT contrib `MatMulNBits`

Inspect the direct standard Attention export:

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\inspect_onnx.py HY-MT-StarCitizen\outputs\onnx-q4f16-attention-direct\model_q4f16.onnx
```

Verified direct standard Attention graph:

- opset: `ai.onnx:23`, `com.microsoft:1`
- `Attention`: 32
- `GroupQueryAttention`: 0
- `MatMulNBits`: 481
- pure ORT CPU inference works with the existing `ort_infer.py`

Verified direct unquantized FP16 standard Attention graph:

- file: `outputs/onnx-fp16-attention-direct/model_fp16.onnx`
- external data: 3,662,946,304 bytes
- opset: `ai.onnx:23`, `com.microsoft:1`
- `Attention`: 32
- `GroupQueryAttention`: 0
- `MatMulNBits`: 0
- `MatMul`: 481

There is also a post-export converter for comparing an existing GQA export against the standard Attention graph:

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\convert_gqa_to_attention.py `
  --input-dir HY-MT-StarCitizen\outputs\onnx-q4f16 `
  --output-dir outputs\onnx-q4f16-attention `
  --opset 23
```

Verified converted graph:

- opset: `ai.onnx:23`, `com.microsoft:1`
- `Attention`: 32
- `GroupQueryAttention`: 0
- `MatMulNBits`: 481
- pure ORT CPU inference works with the existing `ort_infer.py`

Important limitation: both standard Attention paths map `Q,K,V,past_key,past_value` directly and set `is_causal=1`. They are validated for the single-sample, unpadded generation path used by `ort_infer.py`. Keep the original `GroupQueryAttention` export as the conservative default if you need padded batched prompts or want to exactly match ORT GenAI's export semantics.

## Pure ORT CPU Inference

```powershell
.\.venv_hymt2\Scripts\python.exe HY-MT-StarCitizen\scripts\ort_infer.py `
  --onnx-dir HY-MT-StarCitizen\outputs\onnx-q4f16 `
  --filename model_q4f16.onnx `
  --tokenizer-dir models\hy-mt2-model `
  --provider cpu `
  --direction zh-en `
  --text "在洛维尔哪里可以买到飞船武器？" `
  --max-new-tokens 64
```

The inference script:

- imports `onnxruntime`, not `onnxruntime-genai`
- manually manages KV cache inputs and outputs
- uses greedy decoding
- prints token count and elapsed seconds for TPS calculation

Verified CPU outputs with the exported adapter model:

- zh-en short: `Where can I get ship weapons in Lorville?`, 11 tokens / 0.672s, about 16.4 tok/s
- en-zh short: `我在罗威尔哪里能买到舰船武器？`, 11 tokens / 0.708s, about 15.5 tok/s
- zh-en long: `If your ship was destroyed near Orison, please check your insurance status and then head to the nearest terminal to reapply for a vehicle.`, 29 tokens / 1.483s, about 19.6 tok/s
- en-zh long: `在接单之前，请确认你的飞船有足够的空间，并且目的地的前哨站目前没有敌对势力。`, 23 tokens / 1.300s, about 17.7 tok/s
- zh-en mixed terms: `I need to get the quantum fuel to Crusader's showroom and then get back to New Babbage.`, 23 tokens / 1.187s, about 19.4 tok/s

Verified CPU outputs with the standard `Attention` converted model:

- en-zh short: `我在洛维尔哪里能买到舰船武器？`, 11 tokens / 0.607s, about 18.1 tok/s
- zh-en long: 45 tokens / 2.933s, about 15.3 tok/s
- en-zh long: 36 tokens / 2.576s, about 14.0 tok/s

Verified CPU output with the directly exported standard `Attention` model:

- en-zh short: `我在洛维尔哪里能买到舰船武器？`, 11 tokens / 0.735s, about 15.0 tok/s

Verified CPU outputs with the unquantized FP16 standard `Attention` model:

- en-zh short: `我在洛维尔哪里可以买到舰船武器？`, 11 tokens / 1.656s, about 6.6 tok/s
- zh-en long: 44 tokens / 13.530s, about 3.3 tok/s
- en-zh long: 36 tokens / 12.388s, about 2.9 tok/s

## Attention Op Notes

As of ONNX 1.22, the standard `Attention` op exists in the main domain and supports MHA, GQA, MQA, and KV-cache update cases. The current ORT GenAI builder export for Hy-MT2 already emits `com.microsoft:GroupQueryAttention`, which is an ORT contrib operator specifically for GQA with KV cache support and optional KV cache quantization.

For this model, keeping `GroupQueryAttention` is still the lower-risk default path:

- Hy-MT2 uses grouped-query attention.
- The generated graph already has one `GroupQueryAttention` per layer.
- The pure `onnxruntime` CPU session loads and runs the graph.
- The optional `convert_gqa_to_attention.py` path is available when you want to test the standard ONNX `Attention` op directly.
- For the highest ORT compatibility, use `--precision fp16 --attention-op standard`; it removes `MatMulNBits`, but CPU speed is much slower and model size is about 3.66 GB.

## Quantization Notes

The ONNX export here targets ORT's weight-only int4 path (`MatMulNBits`, q4f16). Stock ONNX Runtime has direct support for this style of 4-bit LLM weight quantization. Lower-than-q4 formats such as q2 or GGUF-style 1.25-bit are not a drop-in ORT ONNX export target; they would require a custom operator/runtime path or a different backend. For Rust `ort`, q4f16 is the practical CPU-compatible target.
