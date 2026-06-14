import argparse
from collections import Counter
from pathlib import Path

import onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Hy-MT-StarCitizen ONNX graph.")
    parser.add_argument("model")
    args = parser.parse_args()

    model = onnx.load(args.model, load_external_data=False)
    ops = Counter(node.op_type for node in model.graph.node)
    domain_ops = Counter((node.domain or "") + ":" + node.op_type for node in model.graph.node)
    print(f"path={Path(args.model)}")
    print(f"nodes={len(model.graph.node)}")
    print(f"initializers={len(model.graph.initializer)}")
    print("opsets=" + ", ".join(f"{item.domain or 'ai.onnx'}:{item.version}" for item in model.opset_import))
    print("top_ops=" + ", ".join(f"{name}:{count}" for name, count in ops.most_common(20)))
    print("top_domain_ops=" + ", ".join(f"{name}:{count}" for name, count in domain_ops.most_common(20)))
    print(f"MatMulNBits={ops.get('MatMulNBits', 0)}")
    print(f"Attention={domain_ops.get(':Attention', 0)}")
    print(f"GroupQueryAttention={domain_ops.get('com.microsoft:GroupQueryAttention', 0)}")
    first = next((node for node in model.graph.node if node.op_type == "MatMulNBits"), None)
    if first:
        print("first_matmulnbits=" + str([(attr.name, onnx.helper.get_attribute_value(attr)) for attr in first.attribute]))


if __name__ == "__main__":
    main()
