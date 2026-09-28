#!/usr/bin/env python
"""patch_onnx_for_tract.py

Patch a PyTorch-exported ONNX graph so it loads in `tract` (a pure-Rust ONNX
inference engine: https://github.com/sonos/tract), by working around two of
tract's ONNX operator implementations being narrower than the spec / than
onnxruntime's:

日本語: PyTorchからエクスポートしたONNXグラフを、tract(純Rust製のONNX推論エンジン、
https://github.com/sonos/tract )で読み込めるよう自動修正するツールです。
onnxruntimeでは問題なく動くのに、tractの以下2つの演算子実装が仕様より狭いために
読み込みに失敗するモデル(Transformer/STFTを含むモデルで特によく見られる)向けです。

1. Pad: the ONNX spec allows the 3rd input (`constant_value`) to be omitted,
   but tract's type-inference rule requires exactly 3 inputs.
   -> Adds an explicit zero constant as the 3rd input wherever it's missing
   or given as an empty-string placeholder.
   (Pad: ONNX仕様では3番目の入力`constant_value`は省略可能だが、tractの型推論
   規則は厳密に3入力を要求する。→ 明示的な定数〈0〉を補う。)

2. Range: many exported graphs use `Range` to (re)compute values that only
   depend on the *shape* of a tensor (e.g. position indices for a Transformer,
   frame indices for manual STFT framing). If you fix the model's input
   shape(s), these are actually always the same values — but tract's type
   inference for `Range` can fail on the symbolic-dimension vs. concrete-int64
   distinction. This script runs the graph once via onnxruntime (with your
   fixed input shape(s)) to capture the *concrete* value of every `Range`
   node's output, then replaces each `Range` node with a `Constant` holding
   that captured value.
   (Range: 入力の形状を固定すると、本来は動的に計算される値〈Transformerの
   位置インデックス、STFTのフレーム分割用インデックス等〉も実は常に同じ値になる。
   tractはこの型推論〈symbolic dimension vs 具体的なint64〉で失敗することがあるため、
   onnxruntimeで実際に1回推論して値を捕捉し、Constantノードへ置き換える。)

**Trade-off / 制約**: after patching, the graph only accepts the exact input
shape(s) you fixed it to (the original dynamic-shape support is lost for
those inputs). For variable-length inputs (e.g. audio), the caller is
expected to process the data in fixed-size chunks matching the patched
shape. This is a deliberate, honest trade-off, not a bug.
(パッチ後のグラフは、固定した入力形状**専用**になる〈対象の入力について動的形状
対応が失われる〉。可変長の入力〈音声等〉を扱う場合は、呼び出し側でパッチ後の
固定長のチャンクに区切って処理すること。これはトレードオフであり不具合ではない。)

This tool does not modify tract or onnxruntime themselves, and does not
change the model's numerical behavior for the fixed input shape(s) (verified
against onnxruntime's own output on the same input in the accompanying
tests/README — see the project this was extracted from,
https://github.com/aon-co-jp/make-disk, for a real-world example patching
Meta's HT-Demucs audio source-separation model).

Usage / 使い方:
    pip install onnx onnxruntime numpy
    python patch_onnx_for_tract.py <input.onnx> <output_dir> \\
        --input mix=1,2,343980 \\
        [--input other_input_name=1,4] \\
        [--dtype mix=float32] \\
        [--output-name model_patched.onnx]

    Inputs whose shape is *already* fully static in the graph don't need
    `--input`; give it only for inputs with a dynamic ("symbolic") dimension
    that you want to fix to a concrete value.
    (グラフ内で既に完全に静的な形状を持つ入力には`--input`は不要。動的
    〈symbolic〉な次元を固定したい入力にだけ指定すればよい。)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

# ONNX TensorProto data_type -> numpy dtype (the subset this tool needs to
# synthesize dummy input tensors for; extend if you hit an unsupported dtype).
_ONNX_TO_NUMPY = {
    TensorProto.FLOAT: np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT64: np.int64,
    TensorProto.INT32: np.int32,
    TensorProto.BOOL: np.bool_,
}


def _parse_shape(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_kv_list(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected NAME=VALUE, got: {item!r}")
        name, value = item.split("=", 1)
        out[name] = value
    return out


def _static_shape_of(value_info) -> list[int] | None:
    """Returns the input's shape if every dimension is a concrete int, else None."""
    dims = value_info.type.tensor_type.shape.dim
    shape = []
    for d in dims:
        if d.HasField("dim_value"):
            shape.append(d.dim_value)
        else:
            return None
    return shape


def patch(input_path: str, output_dir: str, input_shapes: dict[str, str], dtypes: dict[str, str], output_name: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {input_path} ...")
    m = onnx.load(input_path)
    g = m.graph

    # ── resolve every graph input's shape & dtype (override, or already-static) ──
    resolved_inputs: dict[str, tuple[list[int], int]] = {}
    for vi in g.input:
        elem_type = vi.type.tensor_type.elem_type
        if vi.name in input_shapes:
            shape = _parse_shape(input_shapes[vi.name])
        else:
            shape = _static_shape_of(vi)
            if shape is None:
                raise SystemExit(
                    f"input {vi.name!r} has a dynamic shape and no --input override was given. "
                    f"Pass --input {vi.name}=dim0,dim1,... to fix it."
                )
        resolved_inputs[vi.name] = (shape, elem_type)
    print("resolved input shapes:", {k: v[0] for k, v in resolved_inputs.items()})

    # ── 1. Pad: add the missing/empty-placeholder 3rd input (constant_value) ──
    const_name = "_zero_pad_const_f32"
    g.initializer.append(numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=const_name))
    pad_fixed = 0
    for n in g.node:
        if n.op_type == "Pad":
            if len(n.input) == 2:
                n.input.append(const_name)
                pad_fixed += 1
            elif len(n.input) == 3 and n.input[2] == "":
                n.input[2] = const_name
                pad_fixed += 1
    print(f"patched {pad_fixed} Pad node(s)")

    # ── 2. Range: run once via onnxruntime (with the fixed input shapes) to
    #      capture every Range node's concrete output, then bake as Constant ──
    ranges = [n for n in g.node if n.op_type == "Range"]
    out_names = [n.output[0] for n in ranges]
    print(f"found {len(ranges)} Range node(s), capturing concrete values via onnxruntime...")

    producer = {o: n for n in g.node for o in n.output}

    def dtype_of(tensor_name: str) -> int | None:
        n = producer.get(tensor_name)
        if n is None:
            return None
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    return attr.t.data_type
        if n.op_type == "Cast":
            for attr in n.attribute:
                if attr.name == "to":
                    return attr.i
        return None

    if ranges:
        m2 = onnx.ModelProto()
        m2.CopyFrom(m)
        g2 = m2.graph
        for n in ranges:
            dt = dtype_of(n.input[0]) or TensorProto.INT64
            g2.output.append(helper.make_tensor_value_info(n.output[0], dt, None))

        sess = ort.InferenceSession(m2.SerializeToString(), providers=["CPUExecutionProvider"])
        feed = {}
        for name, (shape, elem_type) in resolved_inputs.items():
            np_dtype = _ONNX_TO_NUMPY.get(elem_type, np.float32)
            if name in dtypes:
                np_dtype = np.dtype(dtypes[name])
            feed[name] = np.zeros(shape, dtype=np_dtype)
        outs = sess.run(out_names, feed)
        data = dict(zip(out_names, outs))

        # Bake as real graph *initializers* (not Constant-node attributes): a
        # Constant node's tensor lives in the node's attribute, which
        # `save_as_external_data` does not externalize, so protobuf
        # serialization can hit its ~2GB single-message limit on large
        # models. An initializer with the same name as the Range node's
        # output is used automatically wherever that name is referenced,
        # so the Range node itself can simply be dropped.
        # (ノード属性ではなく本物のinitializerとして焼き込む。Constantノードの
        # 属性はsave_as_external_dataで外部データ化されず、モデルが大きいと
        # protobufの単一メッセージの上限〈約2GB〉に達することがあるため。
        # Rangeノードの出力名と同じ名前のinitializerは、その名前が参照されている
        # 箇所で自動的に使われるため、Rangeノード自体は削除するだけでよい。)
        range_names = {n.name for n in ranges}
        for n in ranges:
            arr = data[n.output[0]]
            g.initializer.append(numpy_helper.from_array(arr, name=n.output[0]))
        remaining = [n for n in g.node if n.name not in range_names]
        del g.node[:]
        g.node.extend(remaining)
        print(f"baked {len(ranges)} Range node(s) into initializers")

    out_onnx = out_dir / output_name
    out_data = output_name + ".data"
    # onnx.checker.check_model() tries to serialize the whole model as one
    # protobuf message and fails above ~2GB; skip it (correctness is instead
    # verified by actually running the patched model, see the README) and
    # externalize large tensors so the .onnx file itself stays small.
    onnx.save(m, str(out_onnx), save_as_external_data=True, all_tensors_to_one_file=True, location=out_data)
    print(f"saved {out_onnx} (+ {out_data})")
    return out_onnx


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_onnx")
    p.add_argument("output_dir")
    p.add_argument("--input", action="append", default=[], metavar="NAME=DIM0,DIM1,...", help="Fix a dynamic-shape input to a concrete shape (repeatable).")
    p.add_argument("--dtype", action="append", default=[], metavar="NAME=numpy_dtype", help="Override the numpy dtype used for an input's dummy tensor (repeatable, e.g. mix=float32).")
    p.add_argument("--output-name", default="patched.onnx", help="Output .onnx filename (default: patched.onnx).")
    args = p.parse_args(argv)

    input_shapes = _parse_kv_list(args.input)
    dtypes = _parse_kv_list(args.dtype)
    patch(args.input_onnx, args.output_dir, input_shapes, dtypes, args.output_name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
