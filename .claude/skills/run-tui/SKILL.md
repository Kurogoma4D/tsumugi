---
name: run-tui
description: |
  tsumugiをTUIモードで起動し、対話的な動作確認を行う。
  --event-log でイベントログを出力し、Claudeがログファイルを読んで動作を検証する。
  `/run-tui` で起動。引数にテストしたいプロンプトを指定可能。
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# tsumugi TUI 実行スキル

tsumugiバイナリ (`tmg`) をTUIモードで起動し、`--event-log` による構造化イベントログで動作を検証する。

## 前提条件の確認

### 1. バイナリのビルド

```bash
cargo build --workspace 2>&1
```

ビルドに失敗した場合はエラーを報告して終了。

### 2. LLMサーバーの稼働確認

設定ファイルまたは環境変数からエンドポイントを取得し、疎通を確認する。

```bash
# 設定ファイルの確認
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

### 3. TUI起動

TUIはインタラクティブなので、ユーザーに `!` コマンドでの起動を案内する。
**必ず `--event-log` フラグを付ける:**

> 以下のコマンドでTUIを起動してください:
> ```
> ! ./target/debug/tmg --event-log /tmp/tmg-events.jsonl
> ```
> TUI操作後、Ctrl+C で終了してください。

### 4. イベントログの検証

ユーザーがTUIを終了したら、イベントログを読んで動作を検証する:

```bash
# Read tool で /tmp/tmg-events.jsonl を読む
```

イベントログは JSON Lines 形式で、各行が1イベント:

```jsonl
{"elapsed_ms":0,"event":{"type":"thinking","token":"..."}}
{"elapsed_ms":150,"event":{"type":"token","token":"..."}}
{"elapsed_ms":200,"event":{"type":"tool_call","name":"file_read","arguments":"{...}"}}
{"elapsed_ms":1200,"event":{"type":"tool_result","name":"file_read","output":"...","is_error":false}}
{"elapsed_ms":3000,"event":{"type":"done"}}
```

**イベントタイプ:**
- `thinking` — LLMの思考トークン
- `token` — LLMの応答テキストトークン
- `tool_call` — ツール呼び出し (name + arguments)
- `tool_result` — ツール実行結果 (name + output + is_error)
- `done` — ターン完了
- `warning` — 非致命的警告

### 5. 検証観点

イベントログから以下を確認する:

- **レスポンス生成**: `token` イベントが存在するか
- **ツール実行**: `tool_call` → `tool_result` のペアが正しいか
- **エラー**: `is_error: true` のイベントがないか
- **完了**: `done` イベントで正常終了しているか
- **レイテンシ**: `elapsed_ms` の値からレスポンス速度が妥当か

## テストシナリオ

ユーザーに以下のシナリオでの操作を提案する:

### 基本応答テスト
> TUIで「Hello」と入力 → ログに `token` イベントが記録されることを確認

### ツール実行テスト
> TUIで「Cargo.toml を読んで」と入力 → `tool_call` (file_read) と `tool_result` が記録されることを確認

### 日本語テスト
> TUIで日本語の指示を入力 → `token` イベントで日本語が正しく出力されることを確認

## レポートフォーマット

```
## TUI 動作確認レポート

- エンドポイント: <endpoint>
- モデル: <model>
- イベントログ: /tmp/tmg-events.jsonl

| 確認項目 | 結果 |
|----------|------|
| TUI起動・終了 | PASS / FAIL |
| LLMレスポンス | PASS / FAIL |
| ツール呼び出し | PASS / FAIL / N/A |
| エラーなし | PASS / FAIL |

### イベントサマリ
- 総イベント数: N
- token イベント: N
- tool_call / tool_result: N
- 合計所要時間: Nms

### 問題点 (あれば)
<問題の詳細>
```

## ルール

- コードの変更は行わない (read-only)
- LLMサーバーが未起動の場合は実行せず案内のみ行う
- TUI起動はユーザーに `!` コマンドで委ねる
- 検証はイベントログファイルの `Read` で行う (画面キャプチャは使わない)
