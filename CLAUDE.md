# CLAUDE.md — tract-onnx-patch

## 役割 / Role
**日本語**: PyTorchからエクスポートしたONNXの`Pad`(3番目の入力省略)と`Range`(形状依存値)を、純Rustの推論エンジン`tract`で読める形に直す1ファイルのPythonスクリプト`patch_onnx_for_tract.py`。実モデル(HT-Demucs)で検証済み。入力形状を固定する割り切り(可変長はチャンク処理)。
**English**: A single-file Python script, `patch_onnx_for_tract.py`, that rewrites PyTorch-exported ONNX `Pad` (omitted 3rd input) and `Range` (shape-dependent values) so the pure-Rust `tract` engine can load them. Verified on the real HT-Demucs model. The input shape is fixed (variable-length inputs are chunked).

## HANDOFF 2026-11-18
- 2GB超のprotobuf上限は、Constantノードではなくinitializer(外部データ)として焼き込むことで回避済み(.onnx本体は約148MB)。
- v0.1.0を公開済み。ライセンスはMIT OR Apache-2.0。文書はREADME(日英)・本ファイル・PORTING.md。
- Protobuf >2 GB limit is avoided by baking values as initializers with external data (the .onnx body is ~148 MB). v0.1.0 is released; MIT OR Apache-2.0.
