import os
import pandas as pd
from langchain_community.document_loaders import GitLoader, GithubFileLoader, ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import all_

# === ハードコードされたURL ===
DOC_PATHS = [
    "https://docs.langchain.com/docs",
    "https://docs.langgraph.dev/docs"
]

GIT_REPOS = [
    "https://github.com/langchain-ai/langchain",
    "https://github.com/langchain-ai/langgraph"
]


# === GitHubのリポジトリをクローン＆読み込み ===
all_docs = []
for repo_url in GIT_REPOS:
    loader = GithubFileLoader(
        repo=repo_url,
        access_token=os.getenv("GITHUB_TOKEN"),
        branch="main",
        file_filter=lambda f: f.endswith(".md") or "README" in f or f.endswith(".py") or f.endswith(".ipynb") or f.endswith(".mdx"),
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {repo_url}")
    # all_docs.extend(docs)
    all_docs.extend(docs)

# === 分割 ===
print("🧩 Splitting documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
split_docs = splitter.split_documents(all_docs)

# === prompt / completion に変換 ===
print("📄 Generating prompt/completion pairs...")
df = pd.DataFrame([
    {
        "prompt": f"以下の文章を要約してください:\n\n{doc.page_content}",
        "completion": doc.page_content
    }
    for doc in split_docs
])

# === JSONLで保存 ===
df.to_json("train.jsonl", orient="records", lines=True)
print("✅ train.jsonl を保存しました（件数: {}）".format(len(df)))