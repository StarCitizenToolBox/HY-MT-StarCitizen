import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def resolve_existing_or_project_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    candidate = ROOT / path
    if candidate.exists():
        return str(candidate.resolve())
    return value


def resolve_project_output(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def patch_ortgenai_hunyuan_rope() -> None:
    import onnxruntime_genai.models.builder as builder

    original_init = builder.HunyuanDenseV1Model.__init__

    def patched_init(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        if not hasattr(config, "rope_theta"):
            rope_parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
            if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
                config.rope_theta = rope_parameters["rope_theta"]
            else:
                config.rope_theta = 10000.0
        return original_init(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    builder.HunyuanDenseV1Model.__init__ = patched_init


def patch_ortgenai_standard_attention(opset: int) -> None:
    import onnxruntime_genai.models.builder as builder

    model_base = builder.HunyuanDenseV1Model.__mro__[1]
    original_init = model_base.__init__
    original_save_model = model_base.save_model

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.graph.opset_imports[""] = max(self.graph.opset_imports.get("", 0), opset)

    def patched_save_model(self, out_dir):
        self.graph.opset_imports[""] = max(self.graph.opset_imports.get("", 0), opset)
        return original_save_model(self, out_dir)

    def make_standard_attention(self, name, **kwargs):
        inputs = [
            kwargs["q_path"],
            kwargs["k_path"],
            kwargs["v_path"],
            "",  # attn_mask
            kwargs.get("past_k", ""),
            kwargs.get("past_v", ""),
        ]
        if opset >= 24:
            inputs.append("")  # nonpad_kv_seqlen

        output = f"{name}/output_0"
        outputs = [output, kwargs.get("present_k", ""), kwargs.get("present_v", "")]
        attrs = {
            "q_num_heads": self.num_attn_heads,
            "kv_num_heads": self.num_kv_heads,
            "scale": self.attention_attrs["scale"],
            "is_causal": 1,
        }
        softcap = self.attention_attrs.get("softcap", 0.0)
        if softcap:
            attrs["softcap"] = softcap

        self.make_node("Attention", inputs=inputs, outputs=outputs, name=name, domain="", **attrs)
        self.make_value(
            output,
            self.io_dtype,
            shape=["batch_size", "sequence_length", self.head_size * self.num_attn_heads],
        )

    model_base.__init__ = patched_init
    model_base.save_model = patched_save_model
    model_base.is_gqa_supported = lambda self: True
    model_base.make_group_query_attention = make_standard_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Hy-MT-StarCitizen to ONNX Runtime.")
    parser.add_argument("--model-dir", default="models/hy-mt2-model")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter directory.")
    parser.add_argument("--output-dir", default="outputs/onnx-q4f16")
    parser.add_argument("--cache-dir", default="outputs/ort-cache")
    parser.add_argument("--precision", default="int4", choices=["int4", "fp16", "bf16", "fp32"])
    parser.add_argument("--execution-provider", default="cpu", choices=["cpu", "cuda", "dml", "webgpu"])
    parser.add_argument("--filename", default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=None, help="For tiny smoke exports only.")
    parser.add_argument("--attention-op", default="gqa", choices=["gqa", "standard"])
    parser.add_argument("--attention-opset", type=int, default=23, choices=[23, 24, 25])
    parser.add_argument("--no-prune-lm-head", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from onnxruntime_genai.models.builder import create_model

    patch_ortgenai_hunyuan_rope()
    if args.attention_op == "standard":
        patch_ortgenai_standard_attention(args.attention_opset)
    model_dir = resolve_existing_or_project_path(args.model_dir)
    adapter_path = resolve_existing_or_project_path(args.adapter_path)
    filename = args.filename
    if filename is None:
        filename = "model_q4f16.onnx" if args.precision == "int4" else f"model_{args.precision}.onnx"

    extra_options = {
        "shared_embeddings": "true",
        "filename": filename,
        "hf_remote": "false",
    }
    if args.precision == "int4":
        extra_options.update(
            {
                "int4_accuracy_level": "2",
                "int4_block_size": "32",
                "int4_is_symmetric": "true",
                "int4_algo_config": "rtn",
            }
        )
    if adapter_path:
        extra_options["adapter_path"] = adapter_path
    if args.num_hidden_layers is not None:
        extra_options["num_hidden_layers"] = str(args.num_hidden_layers)
    if not args.no_prune_lm_head:
        extra_options["prune_lm_head"] = "true"

    create_model(
        model_name=None,
        input_path=model_dir,
        output_dir=resolve_project_output(args.output_dir),
        precision=args.precision,
        execution_provider=args.execution_provider,
        cache_dir=resolve_project_output(args.cache_dir),
        **extra_options,
    )


if __name__ == "__main__":
    main()
