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

## Versionless API の方針 / Versionless API policy (2026-11-25)
**日本語**: このリポジトリのAPIは **Versionless**(URL・パッケージ名に `v1` 等を付けず、互換性を壊さず足すだけで進化させる)。欄は足すだけ・足す欄は省略可能・知らない欄は無視・名前変更は新名を足して旧を非推奨に(期限は既定で6か月以上、使用0を確認してから削除)・enumは未知の値を許容・安全上の緊急修正は例外。既存の `.v1` は呼び出し側を壊さないよう当面残し、版なしの別名を追加して段階的に畳む。詳細・出典は [`runo/API-POLICY.md`](https://github.com/aon-co-jp/runo/blob/main/API-POLICY.md) 末尾。適用はこれから順に行う。
**English**: This repository's APIs are **Versionless**: no `v1` in URLs or package names; evolve by adding only, without breaking compatibility. Only add fields, new fields are optional, ignore unknown fields, rename by adding the new name and deprecating the old (default deadline 6 months or more, remove only after confirming zero use), tolerate unknown enum values, emergency security fixes are the exception. Existing `.v1` names stay for now so callers do not break; a versionless alias is added and `.v1` is retired in stages. Details and sources: the end of [`runo/API-POLICY.md`](https://github.com/aon-co-jp/runo/blob/main/API-POLICY.md). Rollout is step by step.

## API 方針: REST 全廃・gRPC / Connect へ
**日本語**: このリポジトリでは **REST API を新規に作らない**。ブラウザ・`curl` から呼ぶ API は **Connect の単純呼び出し**(`POST /<パッケージ>.<Service>/<Method>`、`content-type: application/json`)、サービス間は **gRPC / gRPC-web**。GraphQL は読み取り専用。既存の REST は触るときに移し、移したら旧経路は消す。JSON は RS-JSON(`from_str_secure` / `parse_secure`)で読む。例外は相手の規格が形を決めるもの(ACME、OAuth/OIDC のコールバック、第三者の Webhook、ヘルスチェック、静的ファイル、OTLP)のみ。迷ったら Connect。正本は [`runo/API-POLICY.md`](https://github.com/aon-co-jp/runo/blob/main/API-POLICY.md)。
**English**: This repository **does not add new REST APIs**. APIs called from browsers or `curl` use **Connect unary calls** (`POST /<package>.<Service>/<Method>`, `content-type: application/json`); service-to-service calls use **gRPC / gRPC-web**. GraphQL is read-only. Existing REST is migrated when touched and the old route is removed. Read JSON with RS-JSON (`from_str_secure` / `parse_secure`). Exceptions are only shapes dictated by someone else's standard (ACME, OAuth/OIDC callbacks, third-party webhooks, health checks, static files, OTLP). When in doubt, use Connect. Source of truth: the link above.

## 完成まで作り切る方針 / Finish everything policy (2026-11-27)
**日本語**: 全リポジトリで、**未確認は自動で確認し、未実施は実施し、未了は完成させる**(確認を求めず進める)。使用頻度が低いと判断される機能や、**1から作り直す大規模開発になる内容でも、やめずに最後まで開発・TEST・DEBUGする**(別の方式で済ませない、後回しにしない)。進め方: 調査(英語・日本語の検索とGitHub) → 設計 → 実装 → 実機/実データで確認 → 失敗は原因を直して再試験 → README・CLAUDE・PORTING・POLICY を日英で更新 → push。「できた」と書くのは実測した範囲だけ、確認できなかった所は「未確認」と正直に書く。例外(確認を取る): 元に戻せない削除、秘密の公開、所有者の認証・決済・実機操作が要るもの、利用者のシステム設定の変更。
**English**: In every repository, **verify what is unverified, do what is not yet done, and finish what is unfinished** automatically, without asking. Even features that look rarely used, or work that means **a large rewrite from scratch, are developed, tested and debugged to the end** (no substituting another approach, no postponing). Method: research (English and Japanese search plus GitHub) -> design -> implement -> verify on real devices or data -> fix causes and retest on failure -> update README, CLAUDE, PORTING and POLICY in Japanese and English -> push. Write "done" only for what was measured, and say "unverified" honestly for the rest. Exceptions that still need confirmation: irreversible deletions, publishing secrets, anything needing the owner's authentication, payment or real-device actions, and changes to the user's system settings.