了解です！以下に、ここまでの内容をまとめた README.md をご用意しました。プロジェクトの概要、セットアップ手順、使い方、YAML 設定、curl実行方法、構成などを含んでいます。

⸻

✅ README.md

# 🔎 FAISS HTML Knowledge Crawler

開発ナレッジなどの **HTMLページを再帰的にクロール**し、**テキスト抽出・埋め込み・FAISSによる検索インデックス化**までを一気通貫で実行できるエージェントフレームワークです。

---

## 🚀 機能概要

- 指定されたURL（HTML）を再帰的にクロール
- metaタグやLast-Modifiedヘッダによる更新検知
- テキストコンテンツ抽出・ページタイトル・画像URLの収集
- 変更のあったページのみをベクトル化しFAISSに登録
- バックアップとしてFAISS indexとページ情報(JSON)を保存
- 複数サイトのクロール設定に対応（YAMLで定義）
- FastAPI + Uvicorn ベースで提供（API化可能）

---

## 🛠️ 環境構築

このプロジェクトでは [Poetry](https://python-poetry.org/) を利用します。

### 1. リポジトリをクローン

```bash
git clone <your-repo-url>
cd faiss-index-sample

2. Python 環境の準備

poetry env use python3.10  # 例: Python 3.10を使用

3. 依存ライブラリをインストール

poetry install



⸻

📁 ディレクトリ構成（例）

faiss-index-sample/
├── app/                    # FastAPI アプリケーション
│   └── main.py            # メインのエントリポイント
├── sites.yaml             # クロール対象サイトの設定
├── .env                   # グローバル環境変数
├── pyproject.toml         # Poetry 設定ファイル
├── README.md              # このファイル
└── backup/                # ページとインデックスのバックアップ保存先



⸻

🧪 実行方法

FastAPI サーバーを起動

poetry run uvicorn app.main:app --reload

curl コマンドでクロールを開始

curl -X GET http://localhost:8000/crawl

クロールはバックグラウンドで実行されます。

⸻

🧩 .env の例

BACKUP_DIR=backup
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CONCURRENCY=10
EMBEDDING_MAX_LENGTH=512
EMBEDDING_BATCH_SIZE=32
FAISS_INDEX_TYPE=flat_l2
SITES_YAML=sites.yaml



⸻

📜 sites.yaml の例（複数サイト対応）

sites:
  - name: "LangChain API Reference"
    root_url: "https://python.langchain.com/api_reference/reference.html"
    max_depth: 3
    pattern: "^https://python\\.langchain\\.com/api_reference/.*\\.html$"
    excluded_extensions: ".png,.jpg,.jpeg,.gif,.svg,.pdf,.zip,.mp4,.ico"

  - name: "Example Site"
    root_url: "https://example.com/knowledge/"
    max_depth: 2
    pattern: "^https://example\\.com/knowledge/.*\\.html$"
    excluded_extensions: ".png,.jpg,.jpeg"



⸻

🧠 補足
	•	クロール結果は backup/ に保存されます（サイトごとにサブディレクトリ分け）。
	•	既にバックアップされていて更新がないページはスキップされ、FAISSの再インデックス対象になりません。
	•	テキスト抽出は BeautifulSoup、埋め込みは HuggingFace Transformers を使用。

⸻

✅ 開発予定 / TODO
	•	クロール結果の差分検知による無駄な再埋め込みの回避
	•	複数サイト設定（YAML）
	•	FAISS + LangChain Retriever 統合
	•	RAG（Retrieval Augmented Generation）対応

⸻

📄 ライセンス

MIT

---

必要があればこの README に合わせてディレクトリ構成やコード部分の微修正・ファイル配置もご提案できます！他にも欲しいセクションがあれば言ってくださいね。