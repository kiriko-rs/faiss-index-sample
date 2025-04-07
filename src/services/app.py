import os
import asyncio
import logging
import json
from pathlib import Path
import re
import uuid
from urllib.parse import urljoin, urlparse
from typing import Optional, List, Dict

from bs4 import BeautifulSoup
import faiss
import httpx
import numpy as np
from pydantic_settings import BaseSettings
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import yaml

# プロジェクトのルートディレクトリを基準に解決
PROJECT_ROOT = Path(__file__).resolve().parent.parent

##########################################
# Global Configuration (サイト固有設定は YAML で管理)
##########################################
class Settings(BaseSettings):
    # .env から読み込まれる値
    backup_dir: Path = Field(default=Path("backup"), description="FAISS index と JSON バックアップの保存先")
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="埋め込みモデル名")
    concurrency: int = Field(default=10, description="並列クロールの数")
    embedding_max_length: int = Field(default=512, description="埋め込み時の最大トークン数")
    embedding_batch_size: int = Field(default=32, description="埋め込み時のバッチサイズ")
    faiss_index_type: str = Field(default="flat_l2", description="FAISS index のタイプ（flat_l2 など）")
    sites_yaml: str = Field(default="sites.yaml", description="クロール対象サイト設定ファイル名")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def project_root(self) -> Path:
        # プロジェクトルートの推定（config.py が app/config.py にある場合）
        return Path(__file__).resolve().parent.parent

    @property
    def sites_yaml_path(self) -> Path:
        return self.project_root / self.sites_yaml

    @property
    def backup_path(self) -> Path:
        return self.project_root / self.backup_dir

config = Settings()

##########################################
# Logging Setup
##########################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

##########################################
# Data Model
##########################################
class PageInfo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Page unique ID")
    url: str = Field(..., description="Page URL")
    text_content: Optional[str] = Field(default=None, description="Extracted text content from the page")
    title: str = Field(..., description="Page title")
    page_links: Optional[List[str]] = Field(default=[], description="Internal links on the page")
    image_links: Optional[List[str]] = Field(default=[], description="Image URLs in the page")
    last_modified: Optional[str] = Field(default=None, description="Last modified time from header or meta tag")

##########################################
# Asynchronous Web Crawler (KnowledgeClient)
##########################################
class KnowledgeClient:
    def __init__(self, root_url: str, max_depth: int, pattern: re.Pattern, excluded_extensions: tuple):
        self.root_url = root_url
        self.max_depth = max_depth
        self.pattern = pattern
        self.excluded_extensions = excluded_extensions
        self.visited = set()
        self.pages: List[PageInfo] = []
        self.queue = asyncio.Queue()
        self.queue.put_nowait((self.root_url, 0))

    def normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def is_valid_link(self, url: str) -> bool:
        normalized_url = self.normalize_url(url)
        if not self.pattern.match(normalized_url):
            return False
        if normalized_url.lower().endswith(self.excluded_extensions):
            return False
        return True

    async def worker(self, client: httpx.AsyncClient):
        while not self.queue.empty():
            try:
                url, depth = await self.queue.get()
            except asyncio.QueueEmpty:
                break

            if depth > self.max_depth:
                self.queue.task_done()
                continue

            normalized_url = self.normalize_url(url)
            if normalized_url in self.visited:
                self.queue.task_done()
                continue

            logger.info(f"Crawling: {normalized_url} (depth {depth})")
            self.visited.add(normalized_url)

            try:
                response = await client.get(
                    normalized_url,
                    headers={"Accept": "text/html"},
                    follow_redirects=True,
                    timeout=10.0
                )
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    logger.info(f"Skipping non-HTML content: {normalized_url} ({content_type})")
                    self.queue.task_done()
                    continue
            except Exception as e:
                logger.error(f"Error fetching {normalized_url}: {e}")
                self.queue.task_done()
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            text_content = soup.get_text(separator=" ", strip=True)

            # 更新日時の取得（HTTP ヘッダーまたは meta タグ）
            last_modified = response.headers.get("Last-Modified")
            meta_tag = soup.find("meta", {"property": "article:modified_time"})
            if meta_tag:
                meta_modified = meta_tag.get("content")
                if meta_modified:
                    last_modified = last_modified or meta_modified

            page_links = []
            for a in soup.find_all("a", href=True):
                next_url = urljoin(normalized_url, a["href"])
                next_url = self.normalize_url(next_url)
                if self.is_valid_link(next_url) and next_url not in self.visited:
                    page_links.append(next_url)
                    await self.queue.put((next_url, depth + 1))

            image_links = [urljoin(normalized_url, img["src"]) for img in soup.find_all("img", src=True)]

            page_info = PageInfo(
                url=normalized_url,
                title=title,
                text_content=text_content,
                page_links=page_links,
                image_links=image_links,
                last_modified=last_modified
            )
            self.pages.append(page_info)
            self.queue.task_done()

    async def crawl(self) -> List[PageInfo]:
        async with httpx.AsyncClient() as client:
            tasks = [asyncio.create_task(self.worker(client)) for _ in range(config.concurrency)]
            await self.queue.join()
            for task in tasks:
                task.cancel()
            return self.pages

##########################################
# Backup Functions
##########################################
def save_pages_backup(pages: List[PageInfo], backup_dir: str, filename: str = "pages_backup.json"):
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, filename)
    pages_data = [page.dict() for page in pages]
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Pages backup saved to {backup_path}")

def save_faiss_index_backup(index: faiss.IndexFlatL2, backup_dir: str, filename: str = "faiss_index.index"):
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, filename)
    faiss.write_index(index, backup_path)
    logger.info(f"FAISS index backup saved to {backup_path}")

def load_pages_backup(backup_dir: str, filename: str = "pages_backup.json") -> Dict[str, Optional[str]]:
    backup_path = os.path.join(backup_dir, filename)
    if not os.path.exists(backup_path):
        return {}
    with open(backup_path, "r", encoding="utf-8") as f:
        pages_data = json.load(f)
    mapping = {}
    for page in pages_data:
        mapping[page["url"]] = page.get("last_modified")
    return mapping

##########################################
# Text Embedding & FAISS Indexing (Batch Processing)
##########################################
def get_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    logger.info(f"Loading tokenizer and model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    max_length = config.embedding_max_length
    batch_size = config.embedding_batch_size
    logger.info(f"Tokenizing texts in batches of {batch_size} with max_length {max_length}...")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.numpy())
    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings.astype("float32")

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    embedding_dim = embeddings.shape[1]
    index_type = config.faiss_index_type.lower()
    if index_type == "flat_l2":
        index = faiss.IndexFlatL2(embedding_dim)
    elif index_type == "flat_ip":
        index = faiss.IndexFlatIP(embedding_dim)
    else:
        logger.error(f"Unsupported FAISS index type: {index_type}. Falling back to flat_l2.")
        index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

##########################################
# YAML 設定読み込み
##########################################
def load_sites_config(yaml_path: str) -> List[dict]:
    if not os.path.exists(yaml_path):
        raise ValueError(f"Sites configuration file '{yaml_path}' not found.")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    sites = data.get("sites")
    if not sites:
        raise ValueError("No site configurations found in YAML.")
    # 各サイトエントリに必須項目があるか検証
    for site in sites:
        if "root_url" not in site or "max_depth" not in site or "pattern" not in site or "excluded_extensions" not in site:
            raise ValueError(f"Site configuration incomplete: {site}")
    return sites

##########################################
# FastAPI App
##########################################
app = FastAPI(title="KnowledgeClient API")

@app.get("/crawl")
async def crawl_endpoint(background_tasks: BackgroundTasks):
    """
    /crawl エンドポイントにアクセスすると、バックグラウンドで YAML に定義された複数サイトに対してクロールおよび更新チェック、
    新規または更新されたページのみをベクトル化・FAISS インデックス構築し、バックアップを保存します。
    """
    background_tasks.add_task(run_crawl_process)
    return {"message": "Crawl process started in background."}

async def run_crawl_process():
    sites_config = load_sites_config(config.sites_yaml)
    logger.info(f"Loaded site configurations for {len(sites_config)} sites.")
    tasks = []
    for site in sites_config:
        task = asyncio.create_task(process_site(site))
        tasks.append(task)
    await asyncio.gather(*tasks)

async def process_site(site: dict):
    site_name = site.get("name", "unknown_site")
    root_url = site.get("root_url")
    max_depth = site.get("max_depth")
    pattern_str = site.get("pattern")
    pattern = re.compile(pattern_str)
    excluded_ext_str = site.get("excluded_extensions")
    excluded_extensions = tuple(ext.strip() for ext in excluded_ext_str.split(",") if ext.strip())

    logger.info(f"Starting crawl for site: {site_name} ({root_url})")
    client = KnowledgeClient(root_url, max_depth, pattern, excluded_extensions)
    pages = await client.crawl()
    logger.info(f"[{site_name}] Total pages crawled: {len(pages)}")
    
    backup_dir_site = os.path.join(config.backup_dir, site_name.replace(" ", "_"))
    prev_backup = load_pages_backup(backup_dir_site)
    logger.info(f"[{site_name}] Loaded backup for {len(prev_backup)} pages.")

    filtered_pages = [page for page in pages if page.text_content and len(page.text_content) > 0]
    
    updated_pages = []
    for page in filtered_pages:
        if page.last_modified and page.url in prev_backup:
            if prev_backup[page.url] == page.last_modified:
                logger.info(f"[{site_name}] Page not updated: {page.url} (last_modified: {page.last_modified}), skipping embedding.")
                continue
        updated_pages.append(page)
    
    if not updated_pages:
        logger.error(f"[{site_name}] No new or updated pages with text content found.")
        return

    texts = [page.text_content for page in updated_pages]
    logger.info(f"[{site_name}] Computing embeddings for {len(texts)} updated pages using model '{config.model_name}'...")
    embeddings = get_embeddings(texts, config.model_name)
    logger.info(f"[{site_name}] Embeddings shape: {embeddings.shape}")

    faiss_index = build_faiss_index(embeddings)
    logger.info(f"[{site_name}] FAISS index built with {faiss_index.ntotal} vectors.")

    save_pages_backup(pages, backup_dir=backup_dir_site, filename="pages_backup.json")
    save_faiss_index_backup(faiss_index, backup_dir=backup_dir_site, filename="faiss_index.index")
    
    mapping = {i: page for i, page in enumerate(updated_pages)}
    logger.info(f"[{site_name}] Mapping entries: {len(mapping)}")
    
    query_text = "サンプルの問い合わせ内容"
    query_embedding = get_embeddings([query_text], config.model_name)
    distances, indices = faiss_index.search(query_embedding, k=1)
    nearest_page = mapping.get(indices[0][0])
    logger.info(f"[{site_name}] Nearest page for query:")
    logger.info(nearest_page.json(indent=2))
