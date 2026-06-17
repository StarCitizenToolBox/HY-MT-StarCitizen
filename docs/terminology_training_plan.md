# Terminology Training Plan

## Why this is a model problem

Current failures are not isolated test cases. The same term can translate correctly when alone and drift in context:

- `小刀` -> `Cutter`, but `小刀是个非常不错的新手船` -> `Razor` in current Q4 ONNX.
- Broader merged-term evaluation shows failures across ship names and locations, not just Cutter.

The runtime prompt must stay the normal translation prompt. Terminology should be learned through fine-tuning data and model weights, then validated before and after quantization.

## Evidence from literature

- [Dinu et al., 2019, "Training Neural Machine Translation to Apply Terminology Constraints"](https://aclanthology.org/P19-1294/): training-time terminology injection avoids runtime constrained-decoding overhead and brittleness.
- [Chu and Wang, 2018, "A Survey of Domain Adaptation for Neural Machine Translation"](https://aclanthology.org/C18-1111/): domain-specific translation needs adaptation with in-domain data instead of relying on a generic model.
- [WMT 2024 terminology integration work](https://www2.statmt.org/wmt24/pdf/2024.wmt-1.51.pdf): extracting glossaries from training data and fine-tuning on terminology-focused data improves domain-specific terminology.
- 2025 low-resource MT augmentation analysis: synthetic data quality and how the synthetic data is mixed matter, not just quantity.

## Implemented data strategy

- Mine Star Citizen terms from `global.ini` keys, with caps and filters to avoid noisy UI/task-title terms.
- Keep a small manually curated seed list in `data/terms.zh-en.tsv`.
- Classify all focused ship-name key families as `vehicle`, including `vehicle_Name*`, `Event_ShipName*`, `Event_ShipTitle*`, bare `ShipName_*`, and suffix-style `*_ShipName` keys. A focused audit currently covers 801 short ship/name candidates with zero non-vehicle classifications.
- Generate terminology samples in three forms:
  - direct term pairs, e.g. `小刀 -> Cutter`;
  - contextual dialogue/use-case templates, e.g. buy/rent/fly/search/select/ATC/player text;
  - cross-term player dialogue templates combining locations and vehicles, including bounty, repair, hangar, explosion, crew, and long chat messages;
  - contrast samples, e.g. "I said Cutter, not Razor."
- Give manually curated seed terms higher direct-term repeat weight than mined terms so confirmed high-risk terms survive ONNX Q4 quantization.
- Filter ScWeb rows that conflict with known terminology before they enter training.
- Write the merged mined glossary to `data/processed/terms.merged.zh-en.tsv` for audit.

## Current baseline before the terminology pass

High-risk manual terms, 158 cases:

- LoRA reference: term accuracy `0.8924`, exact accuracy `0.3354`.
- Q4 ONNX: term accuracy `0.7975`, exact accuracy `0.3291`.
- Quantization delta: `-0.0949` term accuracy.

Merged glossary sample, 220 cases:

- LoRA reference: term accuracy `0.8409`, exact accuracy `0.4727`.
- Q4 ONNX: term accuracy `0.7545`, exact accuracy `0.4364`.
- Quantization delta: `-0.0864` term accuracy.

Conclusion: the previous model was undertrained for terminology before quantization, and Q4 added another significant loss.

## Term-focus result

The first `term_focus` checkpoint at step 1000 is already the best tested training point. A longer 14000-step run was started first, but throughput was too low on the local 4090 for this LoRA shape; checkpoint 1000 already exceeded the acceptance target and was used for quantization tests.

High-risk manual terms, 158 cases:

- LoRA checkpoint-1000: term accuracy `1.0000`, exact accuracy `0.9430`.
- Q4 checkpoint-1000: term accuracy `1.0000`, exact accuracy `0.9494`.
- Q4 hybrid checkpoint-1000: term accuracy `1.0000`, exact accuracy `0.9430`.

Merged glossary sample, 300 cases:

- LoRA checkpoint-1000: term accuracy `0.9467`, exact accuracy `0.9200`.
- Q4 checkpoint-1000: term accuracy `0.9133`, exact accuracy `0.8867`.
- Q4 hybrid checkpoint-1000: term accuracy `0.9267`, exact accuracy `0.9000`.

Quantization conclusion:

- Normal Q4 block-128 loses `-0.0333` term accuracy on merged terms, slightly worse than the `-0.03` acceptance line.
- Q4 block-128 with unquantized `lm_head` loses `-0.0200`, so it is the recommended shipping profile.
- The hybrid profile is larger, about `2.8GB`, but still much smaller than a full FP16 model and preserves terminology better.

LoRA FP32 experiment:

- Replacing quantized LoRA branches with FP32 MatMul branches was tested with `scripts/dequantize_lora_onnx.py`.
- Normal Q4 plus FP32 LoRA reached merged-term accuracy `0.9167`, only `+0.0034` over normal Q4 and still below the hybrid export.
- Hybrid Q4 plus FP32 LoRA reached merged-term accuracy `0.9267`, exactly matching the recommended hybrid export, while increasing size from about `2.801GB` to `3.031GB`.
- Conclusion: keeping LoRA unquantized is not a useful shipping tradeoff for terminology. Keeping `lm_head` unquantized is the meaningful quantization change.

## Dialogue v3 result

`term_focus_dialogue_v3` is the current recommended shipping model. It continues from the previous term-focus LoRA, expands player-dialogue training coverage, increases manual seed term weight, and keeps the same hybrid Q4 profile with unquantized `lm_head`.

Focus holdout for the newly reported player-chat failures, 9 cases:

- Previous Q4 hybrid checkpoint-1000: term accuracy `0.2222`, exact accuracy `0.2222`.
- LoRA dialogue v3: term accuracy `1.0000`, exact accuracy `0.8889`.
- Q4 hybrid dialogue v3: term accuracy `1.0000`, exact accuracy `0.8889`.

High-risk manual terms, 160 cases:

- LoRA dialogue v3: term accuracy `1.0000`, exact accuracy `1.0000`.
- Q4 hybrid dialogue v3: term accuracy `0.9938`, exact accuracy `0.9250`.

Merged glossary sample, 300 cases:

- LoRA dialogue v3: term accuracy `0.9933`, exact accuracy `0.9933`.
- Q4 hybrid dialogue v3: term accuracy `0.9700`, exact accuracy `0.9300`.
- Compared with the previous recommended Q4 hybrid checkpoint-1000, merged-term accuracy improves by `+0.0433`.

Known residual:

- Isolated `炽天使` can still quantize to `Ariel` in ONNX. The reported phrase `炽天使空间站` now maps to `Seraphim Station`, and the held-out sentence `炽天使空间站的北极星要爆炸了，快跑啊` is exact.

## Recommended training command

```powershell
P:\AI_WorkSpace\.venv_hymt2\Scripts\python.exe scripts\update_dataset.py --skip-download
P:\AI_WorkSpace\.venv_hymt2\Scripts\python.exe scripts\train_lora.py --config config\term_focus_dialogue_v3_continue.json
```

`config/term_focus_dialogue_v3_continue.json` continues from the previous dialogue LoRA and uses the expanded processed dataset:

- `adapter_init_path: outputs/hymt-starcitizen-term-focus-dialogue-v2-lora`, to preserve the earlier term-focus/dialogue improvements.
- `max_steps: 400`, because this is a stabilization pass over the expanded seed-weighted dataset.
- `learning_rate: 0.00002`, lower than the first terminology pass to avoid forgetting.
- LoRA targets attention and MLP projections: `q/k/v/o/gate/up/down`, rank 64, alpha 128.

## Required acceptance tests

After training:

```powershell
P:\AI_WorkSpace\.venv_hymt2\Scripts\python.exe scripts\eval_terms.py --backend lora --device cuda --adapter-path outputs\hymt-starcitizen-term-focus-dialogue-v3-lora --terms-file data\term_eval_focus.zh-en.tsv --examples-file data\term_eval_holdout.zh-en.tsv --max-terms 3 --max-context-cases 3 --max-cases 12 --output reports\eval_terms_lora_term_focus_dialogue_v3_holdout_seraphim_corsair.json
P:\AI_WorkSpace\.venv_hymt2\Scripts\python.exe scripts\eval_terms.py --backend lora --device cuda --adapter-path outputs\hymt-starcitizen-term-focus-dialogue-v3-lora --terms-file data\processed\terms.merged.zh-en.tsv --max-terms 120 --max-context-cases 360 --max-cases 300 --output reports\eval_terms_lora_term_focus_dialogue_v3_merged120.json
```

Recommended ONNX export:

```powershell
P:\AI_WorkSpace\.venv_hymt2\Scripts\python.exe scripts\export_onnx_q4f16.py --model-dir models\hy-mt2-model --adapter-path outputs\hymt-starcitizen-term-focus-dialogue-v3-lora --output-dir outputs\onnx-q4acc4-b128-hybrid-term-focus-dialogue-v3 --cache-dir outputs\ort-cache-term-focus-dialogue-v3-hybrid --precision int4 --execution-provider cpu --attention-op standard --attention-opset 23 --int4-accuracy-level 4 --int4-block-size 128 --unquantized-lm-head
```

After ONNX export and Q4 hybrid quantization:

```powershell
P:\AI_WorkSpace\.venv_hymt2\Scripts\python.exe scripts\eval_terms.py --backend onnx --provider cpu --onnx-dir outputs\onnx-q4acc4-b128-hybrid-term-focus-dialogue-v3 --filename model_q4acc4_b128_hybrid.onnx --tokenizer-dir outputs\onnx-q4acc4-b128-hybrid-term-focus-dialogue-v3 --terms-file data\term_eval_focus.zh-en.tsv --examples-file data\term_eval_holdout.zh-en.tsv --max-terms 3 --max-context-cases 3 --max-cases 12 --output reports\eval_terms_onnx_q4_hybrid_term_focus_dialogue_v3_holdout_seraphim_corsair.json
P:\AI_WorkSpace\.venv_hymt2\Scripts\python.exe scripts\eval_terms.py --backend onnx --provider cpu --onnx-dir outputs\onnx-q4acc4-b128-hybrid-term-focus-dialogue-v3 --filename model_q4acc4_b128_hybrid.onnx --tokenizer-dir outputs\onnx-q4acc4-b128-hybrid-term-focus-dialogue-v3 --terms-file data\processed\terms.merged.zh-en.tsv --max-terms 120 --max-context-cases 360 --max-cases 300 --output reports\eval_terms_onnx_q4_hybrid_term_focus_dialogue_v3_merged120.json
```

Acceptance target:

- LoRA manual-term accuracy should be at least `0.97`.
- LoRA merged-term accuracy should be at least `0.92`.
- Q4 term accuracy loss should be no worse than `-0.03`; the tested hybrid export meets this with `-0.0200`.
