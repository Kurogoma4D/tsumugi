---
name: run-oneshot
description: |
  tsumugiをワンショットモードで実行し、動作確認を行う。
  --event-log でイベントを記録し、stdout出力とログの両方から動作を検証する。
  `/run-oneshot` で起動。引数にプロンプト文字列を指定可能。
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# tsumugi ワンショット実行スキル

tsumugiバイナリ (`tmg`) をワンショットモード (`--prompt`) で実行し、`--event-log` による構造化イベントログで動作を検証する。

## 前提条件の確認

### 1. バイナリのビルド

```bash
cargo build --workspace 2>&1
```

ビルドに失敗した場合はエラーを報告して終了。

### 2. LLMサーバーの稼働確認

設定ファイルまたは環境変数からエンドポイントを取得し、疎通を確認する。

```bash
# 設定ファイルの確認 (優先順: プロジェクトローカル → グローバル)
cat .tsumugi/tsumugi.toml 2>/dev/null || cat ~/.config/tsumugi/tsumugi.toml 2>/dev/null || echo "no config"

# 環境変数の確認
echo "TMG_LLM_ENDPOINT=${TMG_LLM_ENDPOINT:-未設定}"
echo "TMG_LLM_MODEL=${TMG_LLM_MODEL:-未設定}"
```

エンドポイントが判明したら疎通チェック:

```bash
curl -s --max-time 5 <endpoint>/health || curl -s --max-time 5 <endpoint>/v1/models
```

**サーバーが起動していない場合**はユーザーに案内して終了する。

### 3. ワンショット実行

イベントログを有効にして実行する。プロンプトが引数で指定されていればそれを使い、なければデフォルトを使用。

```bash
timeout 120 ./target/debug/tmg --prompt "<prompt>" --event-log /tmp/tmg-oneshot.jsonl 2>&1
```

**デフォルトのテストプロンプト:**
- 基本応答: `"Hello, what is 2+2?"`

### 4. 出力とイベントログの検証

**stdout 出力の確認:**
- レスポンスが空でないこと
- `[done: ...]` が出力されていること
- パニックやエラーがないこと

**イベントログの確認:**

```
Read /tmp/tmg-oneshot.jsonl
```

イベントログは JSON Lines 形式:
```jsonl
{"elapsed_ms":0,"event":{"type":"thinking","token":"..."}}
{"elapsed_ms":150,"event":{"type":"token","token":"..."}}
{"elapsed_ms":200,"event":{"type":"tool_call","name":"file_read","arguments":"{...}"}}
{"elapsed_ms":1200,"event":{"type":"tool_result","name":"file_read","output":"...","is_error":false}}
{"elapsed_ms":3000,"event":{"type":"done"}}
```

### 5. レポート

```
## ワンショット実行レポート

- エンドポイント: <endpoint>
- モデル: <model>
- プロンプト: "<prompt>"

### 結果: PASS / FAIL

### stdout 出力 (抜粋)
<出力の要点>

### イベントログサマリ
- 総イベント数: N
- token イベント: N
- tool_call: N
- 合計所要時間: Nms

### 問題点 (あれば)
<問題の詳細>
```

## ルール

- コードの変更は行わない (read-only)
- LLMサーバーが未起動の場合は実行せず案内のみ行う
- 出力が非常に長い場合は要点を抜粋する
- タイムアウトは 2 分
