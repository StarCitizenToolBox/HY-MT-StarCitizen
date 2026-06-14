import argparse
import shutil
from pathlib import Path

import onnx
from onnx import helper


ROOT = Path(__file__).resolve().parents[1]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimentally rewrite ORT com.microsoft:GroupQueryAttention nodes to "
            "standard-domain ONNX Attention nodes."
        )
    )
    parser.add_argument("--input-dir", default="outputs/onnx-q4f16")
    parser.add_argument("--output-dir", default="outputs/onnx-q4f16-attention")
    parser.add_argument("--filename", default="model_q4f16.onnx")
    parser.add_argument("--opset", type=int, default=23, choices=[23, 24, 25])
    return parser.parse_args()


def set_default_opset(model: onnx.ModelProto, version: int) -> None:
    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, version)
            return
    model.opset_import.append(helper.make_opsetid("", version))


def convert_node(node: onnx.NodeProto, opset: int) -> onnx.NodeProto:
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    q_num_heads = attrs["num_heads"]
    kv_num_heads = attrs["kv_num_heads"]
    scale = attrs.get("scale")
    softcap = attrs.get("softcap", 0.0)

    inputs = [
        node.input[0],  # Q
        node.input[1],  # K
        node.input[2],  # V
        "",  # attn_mask
        node.input[3],  # past_key
        node.input[4],  # past_value
    ]
    if opset >= 24:
        inputs.append("")  # nonpad_kv_seqlen

    attention_attrs = {
        "q_num_heads": q_num_heads,
        "kv_num_heads": kv_num_heads,
        "is_causal": 1,
    }
    if scale is not None:
        attention_attrs["scale"] = float(scale)
    if softcap:
        attention_attrs["softcap"] = float(softcap)

    return helper.make_node(
        "Attention",
        inputs,
        list(node.output),
        name=node.name.replace("GroupQueryAttention", "Attention"),
        **attention_attrs,
    )


def main() -> None:
    args = parse_args()
    input_dir = resolve_existing_or_project_path(args.input_dir)
    output_dir = resolve_project_output(args.output_dir)
    input_model = input_dir / args.filename
    output_model = output_dir / args.filename

    if not input_model.exists():
        raise FileNotFoundError(input_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(input_model, load_external_data=False)
    converted = 0
    for index, node in enumerate(model.graph.node):
        if node.domain == "com.microsoft" and node.op_type == "GroupQueryAttention":
            model.graph.node[index].CopyFrom(convert_node(node, args.opset))
            converted += 1
    if converted == 0:
        raise RuntimeError("No com.microsoft:GroupQueryAttention nodes found.")

    set_default_opset(model, args.opset)
    onnx.save_model(model, output_model)

    external_data = input_model.with_suffix(input_model.suffix + ".data")
    if external_data.exists():
        shutil.copy2(external_data, output_model.with_suffix(output_model.suffix + ".data"))
    for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "genai_config.json"]:
        source = input_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

    print(f"converted={converted}")
    print(f"output={output_model}")
    print("mode=experimental; valid for batch/prompt paths without padded KV cache")


if __name__ == "__main__":
    main()
