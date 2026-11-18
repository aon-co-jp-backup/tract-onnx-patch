# PORTING.md — tract-onnx-patch

## 再開用メッセージ / Resume message
**日本語**: 「tract-onnx-patch(`F:\tract-onnx-patch`、aon-co-jp/tract-onnx-patch、v0.1.0公開済み)を再開。次は (1) HT-Demucs以外のモデル(別のTransformer系・STFT系)での検証、(2) スクリプトをRustのコマンドにするか判断、(3) `open-media`のGPU/CPU出力一致の測定に使う。入力は`python patch_onnx_for_tract.py`(使い方はREADME)。」
**English**: "Resume tract-onnx-patch (`F:\tract-onnx-patch`, aon-co-jp/tract-onnx-patch, v0.1.0 released). Next: (1) try models other than HT-Demucs (other Transformer/STFT models), (2) decide whether to port the script to a Rust command, (3) use it for the GPU/CPU output comparison in `open-media`. Usage is in the README."

## 注意 / Notes
- 日本語を含むファイルはPowerShellで編集しない(UTF-8のEdit/Writeを使う)。
- Do not edit Japanese text with PowerShell; use UTF-8 Edit/Write.
