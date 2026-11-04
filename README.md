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

### Deutsch (KI-Übersetzung, redaktionell überarbeitet; noch nicht von Muttersprachlern geprüft)
- **REST wird vollständig abgeschafft; neue APIs entstehen nicht mehr als REST.** APIs, die aus dem Browser oder per `curl` aufgerufen werden, verwenden **unäre Connect-Aufrufe** (`POST /<Paket>.<Service>/<Methode>`, `content-type: application/json`, JSON-Body). Zwischen Diensten kommen **gRPC bzw. gRPC-web** zum Einsatz, im selben Stil wie die Admin-API von RPoem (`open_runo.admin.v1.Admin`). Streaming wird erst ergänzt, wenn es tatsächlich gebraucht wird.
- **Bestehende REST-Endpunkte werden schrittweise umgestellt, und zwar immer dann, wenn man ohnehin an ihnen arbeitet.** Sobald ein Endpunkt umgestellt ist, wird die alte REST-Route entfernt. Bis zum Abschluss der Migration bleiben die übrigen bestehen, es werden aber **keine neuen mehr hinzugefügt**.
- **JSON wird einheitlich über RS-JSON verarbeitet**: Gelesen wird mit `from_str_secure` bzw. `parse_secure` (mit Obergrenzen für Größe, Verschachtelungstiefe und doppelte Schlüssel), geschrieben mit `rust_json::fast`. `prost` wird nicht in das Browser-WASM aufgenommen, da es die Datei vergrößern würde.
- **Ausnahmen (Schnittstellen, deren Form ein fremder Standard vorgibt und die nicht zu unserer eigenen API gehören)**: ACME, OAuth/OIDC-Callbacks, Webhooks von Drittanbietern (z. B. GitHub), Health-Checks, die Auslieferung statischer Dateien, OTLP und Ähnliches. Diese nehmen, wie vom jeweiligen Standard verlangt, gewöhnliches HTTP entgegen.
- Referenzimplementierungen: `RPoem` (`open_runo.admin.v1.Admin`) und `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Hinweis: Ob der Code eines einzelnen Repositorys diese Richtlinie bereits erfüllt, hängt vom jeweiligen Repository ab. Dieser Abschnitt dokumentiert lediglich die Richtlinie; er besagt nicht, dass der Code schon angepasst wurde.

### Русский (перевод выполнен с помощью ИИ и отредактирован; носителями языка пока не проверен)
- **От REST мы полностью отказываемся; новые API больше не создаются в стиле REST.** API, к которым обращаются из браузера или через `curl`, используют **унарные вызовы Connect** (`POST /<пакет>.<Service>/<Method>`, `content-type: application/json`, тело запроса — JSON). Для взаимодействия между сервисами применяются **gRPC и gRPC-web** — в том же стиле, что и административный API RPoem (`open_runo.admin.v1.Admin`). Потоковая передача данных добавляется лишь тогда, когда в ней действительно возникает необходимость.
- **Существующие REST-эндпоинты переводятся поэтапно — по мере того, как с ними приходится работать.** После перевода старый REST-маршрут удаляется. Пока миграция не завершена, оставшиеся эндпоинты сохраняются, но **новые не добавляются**.
- **Работа с JSON унифицирована на основе RS-JSON**: чтение — через `from_str_secure` или `parse_secure` (с ограничениями на размер, глубину вложенности и повторяющиеся ключи), запись — через `rust_json::fast`. `prost` в браузерный WASM не включается, поскольку это увеличило бы размер файла.
- **Исключения (интерфейсы, форму которых определяет чужой стандарт, а не наш собственный API)**: ACME, обратные вызовы OAuth/OIDC, вебхуки сторонних сервисов (например, GitHub), проверки работоспособности, раздача статических файлов, OTLP и тому подобное. Такие интерфейсы принимают обычный HTTP, как того требует соответствующий стандарт.
- Эталонные реализации: `RPoem` (`open_runo.admin.v1.Admin`) и `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Примечание: соответствует ли код конкретного репозитория этой политике, зависит от самого репозитория. Данный раздел лишь фиксирует политику и не означает, что код уже изменён.

## API 方針の補足: 用語の整理と使い分け (2026-11-04)

### 日本語
**1つに統一するものではなく、別の層の話です。** 「REST の代わり」になるのは Connect・gRPC・GraphQL だけで、WASM は画面を作る技術です。

| 用語 | 何か | REST の代わりか |
|---|---|---|
| REST | URL と HTTP メソッドで操作を表す API の作り方 | 全廃する側 |
| **Connect** | `POST /サービス名/メソッド名` に JSON を送る形式。`curl` でもブラウザの `fetch` でも呼べる | はい(**採用**) |
| **gRPC** | 同じ「サービス/メソッド」の考え方を、バイナリ(protobuf)と HTTP/2 で行う形式。速く、ストリーミングも得意 | はい(**サービス同士に採用**) |
| gRPC-web | gRPC のブラウザ向けの変形版 | はい(必要なら) |
| GraphQL | 欲しい項目を問い合わせる形式 | 一部だけ(**問い合わせ(読み取り)専用**) |
| WASM | Rust をブラウザの中で動かす仕組み | いいえ(画面を作る技術) |

- **ブラウザの画面や `curl` から呼ぶ API は Connect**(POST、JSON を RS-JSON で読み書き)。ブラウザの WASM は素の gRPC を話せず、Connect なら `fetch` のままで、`prost` を足さずに済む(画面のサイズも増えない)。
- **サービス同士の通信(サーバー間、重い処理、ストリーミング)は gRPC。** 多くは Connect で足りるので、ストリーミングが必要になった所だけ gRPC に上げる。
- **GraphQL は、まとめて問い合わせたいときの読み取り専用。** 書き込み(操作)には使わない。
- **WASM は画面を作る技術として、そのまま使う。** 画面から呼ぶ API が Connect になっただけ。
- 外部の規格が形を決めるもの(ACME、OAuth のコールバック、第三者の Webhook、ヘルスチェック、静的ファイル、OTLP)は、規格どおりの HTTP のまま。
- **迷ったら Connect。**

### English
**These are not alternatives to unify; they sit at different layers.** Only Connect, gRPC and GraphQL can replace REST; WASM is a way to build the UI.

| Term | What it is | Replaces REST? |
|---|---|---|
| REST | API style that expresses operations through URLs and HTTP methods | The thing being abolished |
| **Connect** | `POST /Service/Method` with a JSON body; callable from `curl` and from a browser `fetch` | Yes (**adopted**) |
| **gRPC** | The same service/method idea over binary (protobuf) and HTTP/2; fast, good at streaming | Yes (**for service-to-service**) |
| gRPC-web | A browser-friendly variant of gRPC | Yes (if needed) |
| GraphQL | A query language: ask for exactly the fields you want | Partly (**read-only queries**) |
| WASM | Runs Rust inside the browser | No (a UI technology) |

- **APIs called from browser screens or `curl` use Connect** (POST, JSON read and written with RS-JSON). Browser WASM cannot speak plain gRPC; with Connect it stays a `fetch`, and `prost` is not needed (the binary does not grow).
- **Service-to-service calls (server to server, heavy work, streaming) use gRPC.** Connect is enough in most places; move a call to gRPC only where streaming is actually needed.
- **GraphQL is read-only**, for queries that gather several pieces of data. It is not used for writes (operations).
- **WASM stays as the way to build the UI.** The API it calls simply became Connect.
- Interfaces whose shape an outside standard dictates (ACME, OAuth callbacks, third-party webhooks, health checks, static files, OTLP) stay plain HTTP.
- **When in doubt, use Connect.**

### Deutsch (KI-Übersetzung, sprachlich überarbeitet; noch nicht von Muttersprachlern geprüft)
**Es geht nicht darum, eine einzige Technik auszuwählen – die Begriffe liegen auf unterschiedlichen Ebenen.** Als Ersatz für REST kommen nur Connect, gRPC und GraphQL in Frage; WASM dagegen ist eine Technik, mit der die Benutzeroberfläche gebaut wird.

| Begriff | Was ist das? | Ersatz für REST? |
|---|---|---|
| REST | Ein API-Stil, bei dem Operationen über URLs und HTTP-Methoden ausgedrückt werden | wird abgeschafft |
| **Connect** | `POST /Dienst/Methode` mit JSON im Body; aufrufbar mit `curl` ebenso wie mit `fetch` im Browser | ja (**gewählt**) |
| **gRPC** | Dasselbe Prinzip aus Dienst und Methode, aber über binäres protobuf und HTTP/2; schnell und gut für Streaming geeignet | ja (**zwischen Diensten**) |
| gRPC-web | Eine für den Browser angepasste Variante von gRPC | ja (bei Bedarf) |
| GraphQL | Abfragesprache, mit der man genau die benötigten Felder anfordert | teilweise (**nur für lesende Abfragen**) |
| WASM | Führt Rust-Code direkt im Browser aus | nein (Technik für die Oberfläche) |

- **APIs, die von Browseroberflächen oder per `curl` aufgerufen werden, verwenden Connect** (POST; JSON wird mit RS-JSON gelesen und geschrieben). WASM im Browser kann kein reines gRPC sprechen. Mit Connect bleibt es bei einem gewöhnlichen `fetch`, und `prost` ist nicht nötig – die WASM-Datei wird dadurch nicht größer.
- **Für die Kommunikation zwischen Diensten (Server zu Server, rechenintensive Aufgaben, Streaming) wird gRPC eingesetzt.** In den meisten Fällen reicht Connect aus; auf gRPC wird nur dort umgestellt, wo Streaming tatsächlich gebraucht wird.
- **GraphQL dient ausschließlich lesenden Abfragen**, wenn mehrere Informationen gebündelt abgefragt werden sollen. Für Schreibzugriffe (Operationen) wird es nicht verwendet.
- **WASM bleibt die Technik für die Oberfläche.** Geändert hat sich lediglich, dass die aufgerufene API nun auf Connect beruht.
- Schnittstellen, deren Form durch einen externen Standard festgelegt ist (ACME, OAuth-Callbacks, Webhooks von Drittanbietern, Health-Checks, statische Dateien, OTLP), bleiben gewöhnliches HTTP.
- **Im Zweifel: Connect.**

### Русский (перевод выполнен с помощью ИИ и литературно отредактирован; носителями языка пока не проверен)
**Это не варианты, из которых нужно выбрать один, а технологии разных уровней.** Заменить REST могут только Connect, gRPC и GraphQL; WASM же — это способ создания пользовательского интерфейса.

| Термин | Что это | Заменяет REST? |
|---|---|---|
| REST | Стиль построения API, при котором операции выражаются через URL и HTTP-методы | отменяется |
| **Connect** | `POST /Сервис/Метод` с телом в формате JSON; вызывается как из `curl`, так и из браузерного `fetch` | да (**выбран**) |
| **gRPC** | Тот же принцип «сервис — метод», но поверх двоичного формата protobuf и HTTP/2; работает быстро и хорошо подходит для потоковой передачи | да (**для связи между сервисами**) |
| gRPC-web | Адаптированный для браузера вариант gRPC | да (при необходимости) |
| GraphQL | Язык запросов: клиент запрашивает ровно те поля, которые ему нужны | частично (**только запросы на чтение**) |
| WASM | Позволяет выполнять код на Rust прямо в браузере | нет (технология для интерфейса) |

- **API, которые вызываются из браузерных интерфейсов или через `curl`, строятся на Connect** (POST; JSON читается и записывается через RS-JSON). Браузерный WASM не умеет работать с «чистым» gRPC. С Connect достаточно обычного `fetch`, а `prost` не требуется — размер WASM-файла не увеличивается.
- **Для обмена между сервисами (сервер — сервер, ресурсоёмкие задачи, потоковая передача) применяется gRPC.** В большинстве случаев достаточно Connect; на gRPC переходят только там, где потоковая передача действительно необходима.
- **GraphQL используется только для чтения**, когда нужно получить сразу несколько связанных данных. Для записи (операций) он не применяется.
- **WASM по-прежнему остаётся технологией для интерфейса.** Изменилось лишь то, что вызываемый им API теперь построен на Connect.
- Интерфейсы, форма которых задана внешним стандартом (ACME, обратные вызовы OAuth, вебхуки сторонних сервисов, проверки работоспособности, статические файлы, OTLP), остаются обычным HTTP.
- **Если есть сомнения — выбирайте Connect.**
