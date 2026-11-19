# CLAUDE.md — tract-onnx-patch

<!-- plain-language-2026-11-19 -->
## やさしい説明(次に作業する人・AI へ)

**1 つのファイル `patch_onnx_for_tract.py` だけの、小さな道具です。** 守ってほしいこと:

1. **直したら、必ず本物のモデルで読み込みを確かめる。**(tract で読めるか、答えが元のモデルと同じか)
2. **日本語を含むファイルを PowerShell で書き換えない。** Python か編集ツールを使う。
3. **対応していないモデルは「できない」と書く。** 無理に直したことにしない。

### 専門家向けの要点
入力の形は `--input NAME=1,2,343980` のように固定して指定する。Python 3.14 では onnxruntime が入らないことがあるので 3.13 などの仮想環境を使う。

## In plain words (for the next person or AI)

**A small tool: just one file, `patch_onnx_for_tract.py`.** Please keep these rules:

1. **After fixing, always check loading with the real model** (does tract read it, and are the answers the same as the original?).
2. **Do not edit files containing Japanese with PowerShell.** Use Python or the editor tools.
3. **If a model is not supported, say "cannot".** Do not pretend it was fixed.

### Key points for experts
Fix the input shape with `--input NAME=1,2,343980`. onnxruntime may not install on Python 3.14, so use a 3.13 virtual environment.

## 役割 / Role
**日本語**: PyTorchからエクスポートしたONNXの`Pad`(3番目の入力省略)と`Range`(形状依存値)を、純Rustの推論エンジン`tract`で読める形に直す1ファイルのPythonスクリプト`patch_onnx_for_tract.py`。実モデル(HT-Demucs)で検証済み。入力形状を固定する割り切り(可変長はチャンク処理)。
**English**: A single-file Python script, `patch_onnx_for_tract.py`, that rewrites PyTorch-exported ONNX `Pad` (omitted 3rd input) and `Range` (shape-dependent values) so the pure-Rust `tract` engine can load them. Verified on the real HT-Demucs model. The input shape is fixed (variable-length inputs are chunked).

## HANDOFF 2026-11-18
- 2GB超のprotobuf上限は、Constantノードではなくinitializer(外部データ)として焼き込むことで回避済み(.onnx本体は約148MB)。
- v0.1.0を公開済み。ライセンスはMIT OR Apache-2.0。文書はREADME(日英)・本ファイル・PORTING.md。
- Protobuf >2 GB limit is avoided by baking values as initializers with external data (the .onnx body is ~148 MB). v0.1.0 is released; MIT OR Apache-2.0.
