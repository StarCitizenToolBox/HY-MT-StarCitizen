import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]


ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}
ALL_MODULES = sorted(ATTN_MODULES | MLP_MODULES)


def resolve_existing_or_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (ROOT / path).resolve()


def resolve_project_output(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_adapter_config(adapter_path: Path) -> tuple[int, int, float]:
    config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    rank = int(config["r"])
    alpha = int(config["lora_alpha"])
    return rank, alpha, alpha / rank


def adapter_key(layer: int, module: str, suffix: str) -> str:
    block = "self_attn" if module in ATTN_MODULES else "mlp"
    return f"base_model.model.model.layers.{layer}.{block}.{module}.{suffix}.weight"


def onnx_prefix(layer: int, module: str) -> str:
    block = "attn" if module in ATTN_MODULES else "mlp"
    return f"model.layers.{layer}.{block}.{module}"


def node_prefix(layer: int, module: str) -> str:
    block = "attn" if module in ATTN_MODULES else "mlp"
    return f"/model/layers.{layer}/{block}/{module}"


def make_initializer(name: str, array: np.ndarray) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(array.astype(np.float32)), name=name)


def replace_lora_node(node: onnx.NodeProto, weight_name: str) -> None:
    source_input = node.input[0]
    del node.input[:]
    node.input.extend([source_input, weight_name])
    node.domain = ""
    node.op_type = "MatMul"
    del node.attribute[:]
    if node.name.endswith("_Q4"):
        node.name = node.name[:-3]


def remove_initializers(model: onnx.ModelProto, names: set[str]) -> int:
    kept = []
    removed = 0
    for initializer in model.graph.initializer:
        if initializer.name in names:
            removed += 1
        else:
            kept.append(initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def update_genai_config(source_dir: Path, output_dir: Path, filename: str) -> None:
    config_path = source_dir / "genai_config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["decoder"]["filename"] = filename
    (output_dir / "genai_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=4), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace quantized LoRA branches in an ORT ONNX export with FP32 LoRA MatMul branches.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-filename", default="model_q4acc4_b128.onnx")
    parser.add_argument("--output-filename", default="model_q4acc4_b128_lora_fp32.onnx")
    parser.add_argument("--num-hidden-layers", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_existing_or_project_path(args.input_dir)
    adapter_path = resolve_existing_or_project_path(args.adapter_path)
    output_dir = resolve_project_output(args.output_dir)
    input_model_path = input_dir / args.input_filename
    output_model_path = output_dir / args.output_filename
    output_dir.mkdir(parents=True, exist_ok=True)

    rank, alpha, scale = load_adapter_config(adapter_path)
    print(f"adapter rank={rank} alpha={alpha} scale={scale}")
    print(f"loading {input_model_path}")
    model = onnx.load_model(input_model_path, load_external_data=True)

    node_by_name = {node.name: node for node in model.graph.node}
    new_initializers: list[TensorProto] = []
    old_initializers: set[str] = set()
    replaced = 0

    with safe_open(adapter_path / "adapter_model.safetensors", framework="pt", device="cpu") as adapter:
        for layer in range(args.num_hidden_layers):
            for module in ALL_MODULES:
                prefix = onnx_prefix(layer, module)
                graph_prefix = node_prefix(layer, module)
                a_key = adapter_key(layer, module, "lora_A")
                b_key = adapter_key(layer, module, "lora_B")
                a = adapter.get_tensor(a_key).numpy()
                b = adapter.get_tensor(b_key).numpy()

                a_weight_name = f"{prefix}.lora_A.MatMul.weight_fp32"
                b_weight_name = f"{prefix}.lora_B.MatMul.weight_fp32"
                new_initializers.append(make_initializer(a_weight_name, a.T))
                new_initializers.append(make_initializer(b_weight_name, (b * scale).T))

                a_node = node_by_name[f"{graph_prefix}/lora_A/MatMul_Q4"]
                b_node = node_by_name[f"{graph_prefix}/lora_B/MatMul_Q4"]
                replace_lora_node(a_node, a_weight_name)
                replace_lora_node(b_node, b_weight_name)
                replaced += 2

                old_initializers.update(
                    {
                        f"{prefix}.lora_A.MatMul.weight_Q4G128",
                        f"{prefix}.lora_A.MatMul.weight_scale",
                        f"{prefix}.lora_B.MatMul.weight_Q4G128",
                        f"{prefix}.lora_B.MatMul.weight_scale",
                    }
                )

    removed = remove_initializers(model, old_initializers)
    model.graph.initializer.extend(new_initializers)
    print(f"replaced_nodes={replaced}")
    print(f"removed_initializers={removed}")
    print(f"new_initializers={len(new_initializers)}")
    onnx.save_model(
        model,
        output_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{args.output_filename}.data",
        size_threshold=1024,
        convert_attribute=False,
    )

    for name in ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]:
        source = input_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    update_genai_config(input_dir, output_dir, args.output_filename)
    print(f"output={output_model_path}")


if __name__ == "__main__":
    main()
