# tsumugi

ローカル LLM で動くコーディングエージェント。llama-server (llama.cpp) と連携し、TUI 上でファイル操作・コード探索・シェル実行などを自律的に行います。

## 特徴

- **完全ローカル動作** - 外部 API 不要。llama.cpp の OpenAI 互換エンドポイントで動作
- **TUI インターフェース** - ratatui ベースのチャット UI、ツール実行ログ、コンテキスト使用量のリアルタイム表示
- **ビルトインツール** - `file_read`, `file_write`, `file_patch`, `grep_search`, `list_dir`, `shell_exec`
- **Subagent** - explore (読み取り専用探索)、worker (汎用)、plan (計画立案) の 3 種 + TOML で独自定義可能
- **Skills** - `SKILL.md` による拡張可能な知識・ワークフロー注入
- **サンドボックス** - Landlock LSM によるファイルシステムアクセス制御 (Linux)
- **コンテキスト圧縮** - トークン使用量の監視と自動圧縮で限られた context window を有効活用
- **Prompt-based tool calling** - native function calling 非対応モデル向けのフォールバック
- **コネクションプール** - 複数の llama-server でリクエストを分散、Subagent の並列実行を効率化

## 必要なもの

- Rust 1.85.0+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) の `llama-server`

## ビルド

```bash
cargo build --release
```

バイナリは `target/release/tmg` に生成されます。

## 使い方

### llama-server の起動

```bash
llama-server -m model.gguf -c 8192 --port 8080
```

### TUI モード

```bash
tmg --endpoint http://localhost:8080 --model model-name
```

### ワンショットモード

```bash
tmg --endpoint http://localhost:8080 --model model-name --prompt "src/main.rs を読んで要約して"
```

## 設定

`tsumugi.toml` で設定を管理できます。探索順序:

1. `~/.config/tsumugi/tsumugi.toml` (グローバル)
2. `.tsumugi/tsumugi.toml` (プロジェクトローカル、優先マージ)

```toml
[llm]
endpoint = "http://localhost:8080"
model = "qwen3-8b"
tool_calling = "auto"  # "native" | "prompt_based" | "auto"

[llm.subagent_pool]
endpoints = [
    "http://localhost:8080",
    "http://localhost:8081",
]
strategy = "round_robin"  # "round_robin" | "random"

[sandbox]
mode = "workspace_write"  # "read_only" | "workspace_write" | "full"

[skills]
discovery_paths = ["./my-skills"]
compat_claude = true
compat_agent_skills = true
```

環境変数 (`TMG_LLM_ENDPOINT`, `TMG_LLM_MODEL` など) や CLI オプションでもオーバーライドできます。

優先順位: CLI > 環境変数 > プロジェクトローカル > グローバル

## カスタム Subagent

`.tsumugi/agents/*.toml` に TOML ファイルを配置して独自の Subagent を定義できます。

```toml
name = "reviewer"
description = "コードレビューを行う Subagent"
instructions = "与えられたコードを品質・バグ・設計の観点でレビューしてください。"

[tools]
allow = ["file_read", "grep_search", "list_dir"]
```

## プロジェクト構成

```
tsumugi/
  tmg/          CLI エントリポイント・設定読み込み
  crates/
    tmg-core/       エージェントループ・コンテキスト管理
    tmg-llm/        LLM クライアント (SSE streaming)・コネクションプール
    tmg-tools/      ビルトインツール (file_read, shell_exec, ...)
    tmg-skills/     Skill 探索・読み込み・use_skill ツール
    tmg-agents/     Subagent 管理・spawn_agent ツール
    tmg-sandbox/    サンドボックス (Landlock)
    tmg-tui/        TUI (ratatui)
```

## ライセンス

MIT
