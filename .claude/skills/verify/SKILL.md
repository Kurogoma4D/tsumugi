---
name: verify
description: |
  tsumugiプロジェクトの動作確認を行う。ビルド、lint、フォーマット、テスト、
  バイナリのスモークテストを順に実行し、結果をレポートする。
  `/verify` で起動。
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# tsumugi 動作確認スキル

tsumugi (local-LLM-powered coding agent) プロジェクトの品質ゲートを一括で実行し、結果をレポートする。

## 実行手順

以下のステップを順に実行し、各ステップの成否を記録する。
**いずれかのステップが失敗しても、残りのステップは続行する。**

### Step 1 — ビルド

```bash
cargo build --workspace --all-targets 2>&1
```

- 成功: `Compiling` / `Finished` で終了
- 失敗: コンパイルエラーを記録

### Step 2 — Clippy (lint)

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings 2>&1
```

- 成功: warning なし
- 失敗: 警告/エラーを記録

### Step 3 — フォーマットチェック

```bash
cargo fmt --all -- --check 2>&1
```

- 成功: 差分なし
- 失敗: フォーマット違反のファイル一覧を記録

### Step 4 — テスト

```bash
cargo test --workspace 2>&1
```

- 成功: 全テスト pass
- 失敗: 失敗したテスト名とエラーメッセージを記録

### Step 5 — バイナリ スモークテスト

ビルドが成功している場合のみ実行:

```bash
./target/debug/tmg --help 2>&1
```

- 成功: ヘルプテキストが出力される
- 失敗: 実行時エラーを記録

### Step 6 — 依存関係チェック (任意)

`cargo-deny` がインストールされている場合のみ:

```bash
command -v cargo-deny && cargo deny check 2>&1 || echo "cargo-deny not installed, skipping"
```

## レポートフォーマット

全ステップ完了後、以下の形式でサマリを出力する:

```
## 動作確認レポート

| ステップ | 結果 |
|----------|------|
| ビルド | PASS / FAIL |
| Clippy | PASS / FAIL |
| フォーマット | PASS / FAIL |
| テスト | PASS / FAIL (N passed, M failed) |
| スモークテスト | PASS / FAIL / SKIP |
| 依存関係チェック | PASS / FAIL / SKIP |

### 問題の詳細
(失敗したステップがあれば、エラー内容と修正の方針を記載)
```

## ルール

- 各コマンドのタイムアウトは 5 分とする
- 失敗したステップがあっても残りは必ず実行する (fail-forward)
- エラー出力が長い場合は要点のみ抜粋する
- 修正が必要な場合は、具体的なファイルパスと修正方針を提案する
- このスキルはコードの変更を行わない (read-only)
