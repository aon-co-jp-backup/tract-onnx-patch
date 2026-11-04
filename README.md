# tract-onnx-patch

[日本語](#日本語) / English

A small command-line tool that patches PyTorch-exported ONNX graphs so they
load in [`tract`](https://github.com/sonos/tract) (a pure-Rust ONNX inference
engine), by working around two of tract's ONNX operator implementations
being narrower than the spec (or than onnxruntime's behavior).

## Why

`tract` is a great choice for shipping AI features in a self-contained Rust
binary (no Python, no C++ runtime dependency) — but its ONNX operator
coverage is narrower than onnxruntime's in a couple of specific ways that
show up often in real exported models, especially ones with Transformer
blocks or manual STFT framing:

1. **`Pad`**: the ONNX spec allows the 3rd input (`constant_value`) to be
   omitted, but tract's type-inference rule requires exactly 3 inputs.
   → This tool adds an explicit zero constant wherever it's missing or given
   as an empty-string placeholder.
2. **`Range`**: many exported graphs use `Range` to (re)compute values that
   only depend on a tensor's *shape* (e.g. Transformer position indices,
   manual STFT frame indices). If you fix the model's input shape(s), these
   are always the same values — but tract's type inference for `Range` can
   fail on the symbolic-dimension vs. concrete-`int64` distinction. This tool
   runs the graph once via `onnxruntime` (with your fixed input shape) to
   capture the *concrete* value of every `Range` node's output, then replaces
   each `Range` node with a `Constant` holding that value.

**Trade-off**: the patched graph only accepts the exact input shape(s) you
fixed it to. For variable-length inputs (e.g. audio), the caller processes
data in fixed-size chunks matching the patched shape. This is a deliberate
trade-off, not a bug — and for many real models (audio/vision models with a
natural "chunk size"), it's a perfectly reasonable one.

## Real-world result

Used to patch Meta's **HT-Demucs** (music source separation) ONNX export
([demucs-onnx](https://github.com/StemSplit/demucs-onnx)) so it runs in
[make-disk](https://github.com/aon-co-jp/make-disk)'s AI upmix feature —
verified working end to end with `tract` (load → optimize → real inference,
correct output shape) on 2026-09-28. See that project's
`tools/demucs-onnx-patch/` for the exact invocation used.

## Usage

```bash
pip install onnx onnxruntime numpy
python patch_onnx_for_tract.py <input.onnx> <output_dir> \
    --input mix=1,2,343980 \
    [--input other_input_name=1,4] \
    [--dtype mix=float32] \
    [--output-name model_patched.onnx]
```

Inputs whose shape is *already* fully static in the graph don't need
`--input` — give it only for inputs with a dynamic ("symbolic") dimension
you want to fix to a concrete value.

The patched model is saved with external data (`<output-name>.data`) so the
`.onnx` file itself stays small even for large models.

## What this tool does *not* do

- It doesn't change the model's numerical behavior for the fixed input
  shape (the `Range` values it bakes in are the model's own real
  onnxruntime-computed values, not approximations).
- It doesn't patch every possible tract/ONNX incompatibility — only these
  two specific, common ones. If you hit a different unsupported op, this
  tool won't fix it (but the same technique — inspect with `onnx`, patch
  the graph, re-verify with `onnxruntime` — often generalizes).
- It requires Python only at model-preparation time. The patched `.onnx`
  file itself is then loaded by `tract` at runtime — no Python needed there.

## License

MIT OR Apache-2.0

---

## 日本語

PyTorchからエクスポートしたONNXグラフを、[`tract`](https://github.com/sonos/tract)
(純Rust製のONNX推論エンジン)で読み込めるよう自動修正する、小さなコマンドラインツールです。
tractのONNX演算子の実装が、仕様(やonnxruntimeの挙動)より狭い箇所を2つ、機械的に
回避します。

## なぜ必要か

`tract`は、Pythonも C++ ランタイムも不要な自己完結型Rustバイナリで AI 機能を配布するのに
優れた選択肢です。しかし、その ONNX 演算子の対応範囲は onnxruntime より狭い箇所があり、
特に Transformer ブロックや手動 STFT フレーム分割を含む実際のエクスポート済みモデルで
よく問題になります:

1. **`Pad`**: ONNX 仕様では3番目の入力(`constant_value`)は省略可能ですが、tract の
   型推論規則は厳密に3入力を要求します。→ 欠けている/空文字列のプレースホルダになっている
   箇所へ、明示的な定数(0)を補います。
2. **`Range`**: 多くのエクスポート済みグラフは、テンソルの**形状**だけに依存する値
   (Transformer の位置インデックス、手動 STFT のフレーム分割用インデックス等)を
   再計算するために `Range` を使います。モデルの入力形状を固定すると、これらは常に
   同じ値になりますが、tract の `Range` に対する型推論は symbolic dimension と
   具体的な `int64` の区別で失敗することがあります。このツールは、固定した入力形状で
   `onnxruntime` を使って実際に1回推論を実行し、全ての `Range` ノードの出力の
   **具体的な値**を捕捉した上で、各 `Range` ノードをその値を持つ `Constant` へ
   置き換えます。

**トレードオフ**: パッチ後のグラフは、固定した入力形状**専用**になります。可変長の
入力(音声等)を扱う場合は、呼び出し側でパッチ後の固定長のチャンクに区切って処理する
ことになります。これは意図したトレードオフであり不具合ではありません。多くの実際の
モデル(音声・映像モデルなど、自然な「チャンク単位」がある場合)では十分合理的な
妥協点です。

## 実績

Meta 製の音源分離モデル**HT-Demucs**の ONNX 変換版
([demucs-onnx](https://github.com/StemSplit/demucs-onnx))をパッチし、
[make-disk](https://github.com/aon-co-jp/make-disk)の AI アップミックス機能で使用。
2026-09-28 に `tract` での読み込み→最適化→実際の推論実行(正しい出力 shape)まで
実機で確認済み。実際の呼び出し方法は同プロジェクトの `tools/demucs-onnx-patch/` 参照。

## 使い方

```bash
pip install onnx onnxruntime numpy
python patch_onnx_for_tract.py <元のonnx> <出力先ディレクトリ> \
    --input mix=1,2,343980 \
    [--input 別の入力名=1,4] \
    [--dtype mix=float32] \
    [--output-name パッチ後のファイル名.onnx]
```

グラフ内で既に完全に静的な形状を持つ入力には `--input` は不要です。動的
(symbolic)な次元を固定したい入力にだけ指定してください。

パッチ後のモデルは外部データ形式(`<出力名>.data`)で保存されるため、モデルが
大きい場合でも `.onnx` ファイル自体は小さいままです。

## このツールがしないこと

- 固定した入力形状に対するモデルの数値的な挙動は変えません(焼き込む `Range` の値は、
  モデル自身が onnxruntime で実際に計算した本物の値であり、近似ではありません)。
- あらゆる tract/ONNX 非互換をパッチするわけではなく、この2つの特定の・よくある
  ケースだけを扱います。別の非対応演算子に当たった場合はこのツールでは解決しません
  (ただし同じ手法──`onnx`で調べる→グラフをパッチする→`onnxruntime`で再検証する──は
  他のケースにも応用できることが多いです)。
- Python が必要なのはモデル準備の時だけです。パッチ済みの `.onnx` ファイル自体は
  実行時に `tract` が読み込むだけで、そこに Python は不要です。

## ライセンス

MIT OR Apache-2.0

## API 方針: REST 全廃・gRPC / Connect へ (2026-11-04)

### 日本語
- **REST は全廃する。新しい API は REST で作らない。** ブラウザや `curl` から呼ぶ API は **Connect の単純呼び出し**(`POST /<パッケージ>.<Service>/<Method>`、`content-type: application/json`、本文は JSON)、サービス間は **gRPC / gRPC-web**(RPoem の管理API `open_runo.admin.v1.Admin` と同じ流儀)。ストリーミングは、必要になってから。
- **既存の REST は、触るときに順に移す。** 移したら旧 REST の経路は消す。移行が終わるまで残るが、**新規には追加しない**。
- **JSON は RS-JSON に統一**する(読み込みは `from_str_secure` / `parse_secure` でサイズ・深さ・重複キーの上限つき、書き出しは `rust_json::fast`)。ブラウザの WASM に `prost` は入れない(サイズが増えるため)。
- **例外(自分たちの API ではなく、相手の規格が決めているもの)**: ACME、OAuth/OIDC のコールバック、第三者(GitHub など)からの Webhook、ヘルスチェック、静的ファイル配信、OTLP など。これらは規格どおりの HTTP を受ける。
- 参考実装: `RPoem`(`open_runo.admin.v1.Admin`)、`open-easy-web`(`server/src/connect.rs`、`proto/open_easy_web/admin/v1/admin.proto`)。
- ※ このリポジトリのコードが既に方針どおりかどうかは、各リポジトリの現状による。この節は方針の記録であり、コードの変更を意味しない。

### English
- **REST is being abolished. New APIs are not built as REST.** APIs called from browsers or `curl` use **Connect unary calls** (`POST /<package>.<Service>/<Method>`, `content-type: application/json`, JSON body); service-to-service calls use **gRPC / gRPC-web** (the same style as RPoem's admin API `open_runo.admin.v1.Admin`). Streaming only when it is actually needed.
- **Existing REST endpoints are migrated one by one, when they are touched.** The old REST route is removed once migrated. Until migration is finished they remain, but **no new ones are added**.
- **JSON is unified on RS-JSON**: reading with `from_str_secure` / `parse_secure` (size, depth and duplicate-key limits), writing with `rust_json::fast`. `prost` is not put into browser WASM (it would grow the binary).
- **Exceptions (shapes dictated by someone else's standard, not our own API)**: ACME, OAuth/OIDC callbacks, webhooks from third parties (e.g. GitHub), health checks, static file serving, OTLP, and so on. These accept plain HTTP as the standard requires.
- Reference implementations: `RPoem` (`open_runo.admin.v1.Admin`) and `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Note: whether the code in a given repository already follows this policy depends on that repository. This section records the policy; it does not mean the code has been changed.

### Deutsch (KI-Übersetzung, noch nicht von Muttersprachlern geprüft)
- **REST wird abgeschafft. Neue APIs werden nicht als REST gebaut.** APIs, die aus dem Browser oder mit `curl` aufgerufen werden, nutzen **Connect-Unary-Aufrufe** (`POST /<Paket>.<Service>/<Methode>`, `content-type: application/json`, JSON-Body); zwischen Diensten gilt **gRPC / gRPC-web** (wie bei der Admin-API von RPoem, `open_runo.admin.v1.Admin`). Streaming erst, wenn es wirklich nötig ist.
- **Bestehende REST-Endpunkte werden einzeln migriert, sobald man sie anfasst.** Nach der Migration wird die alte REST-Route entfernt. Bis dahin bleiben sie bestehen, es werden aber **keine neuen hinzugefügt**.
- **JSON wird auf RS-JSON vereinheitlicht**: Lesen mit `from_str_secure` / `parse_secure` (Grenzen für Größe, Tiefe und doppelte Schlüssel), Schreiben mit `rust_json::fast`. `prost` kommt nicht in Browser-WASM (die Datei würde größer).
- **Ausnahmen (Formate, die ein fremder Standard vorgibt, nicht unsere eigene API)**: ACME, OAuth/OIDC-Callbacks, Webhooks von Dritten (z. B. GitHub), Health-Checks, statische Dateien, OTLP usw. Diese nehmen schlichtes HTTP entgegen, wie es der Standard verlangt.
- Referenzimplementierungen: `RPoem` (`open_runo.admin.v1.Admin`) und `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Hinweis: Ob der Code eines Repositorys diese Richtlinie schon befolgt, hängt vom jeweiligen Repository ab. Dieser Abschnitt hält die Richtlinie fest; er bedeutet nicht, dass der Code bereits geändert wurde.

### Русский (машинный перевод, носителями языка ещё не проверен)
- **REST полностью отменяется. Новые API не создаются как REST.** API, вызываемые из браузера или через `curl`, используют **унарные вызовы Connect** (`POST /<пакет>.<Service>/<Method>`, `content-type: application/json`, тело — JSON); между сервисами — **gRPC / gRPC-web** (так же, как административный API RPoem, `open_runo.admin.v1.Admin`). Потоковая передача — только когда она действительно нужна.
- **Существующие REST-эндпоинты переносятся по одному, когда до них доходят руки.** После переноса старый REST-маршрут удаляется. До завершения переноса они остаются, но **новые не добавляются**.
- **JSON унифицируется на RS-JSON**: чтение через `from_str_secure` / `parse_secure` (ограничения на размер, глубину и повторяющиеся ключи), запись через `rust_json::fast`. `prost` не добавляется в браузерный WASM (размер вырос бы).
- **Исключения (форматы, заданные чужим стандартом, а не наш собственный API)**: ACME, обратные вызовы OAuth/OIDC, вебхуки сторонних сервисов (например, GitHub), проверки работоспособности, раздача статических файлов, OTLP и т. п. Они принимают обычный HTTP, как требует стандарт.
- Эталонные реализации: `RPoem` (`open_runo.admin.v1.Admin`) и `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Примечание: соответствует ли код конкретного репозитория этой политике, зависит от самого репозитория. Этот раздел фиксирует политику и не означает, что код уже изменён.
