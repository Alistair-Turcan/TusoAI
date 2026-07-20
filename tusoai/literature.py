from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
MAX_PAPER_SEARCHES = 20
FIELDS = (
    "title,year,abstract,citationCount,isOpenAccess,openAccessPdf,venue,authors,url,paperId"
)


def search_papers_page(
    query: str,
    api_key: str = "",
    *,
    limit: int = 100,
    offset: int = 0,
    require_open_access_pdf: bool = True,
) -> Tuple[List[dict], Optional[int], Optional[int]]:
    import requests
    params: Dict[str, Any] = {
        "query": query,
        "limit": limit,
        "offset": offset,
        "fields": FIELDS,
    }
    if require_open_access_pdf:
        params["openAccessPdf"] = ""

    headers = None

    total_wait_s = 0
    retry_count = 0
    step_s = 5
    max_total_wait_s = 120
    retryable_statuses = {408, 429, 500, 502, 503, 504}

    while True:
        try:
            resp = requests.get(
                SEMANTIC_SCHOLAR_API, params=params, headers=headers, timeout=30
            )

            if resp.status_code == 200:
                payload = resp.json()
                data = payload.get("data", []) or []

                nxt_raw = payload.get("next", None)
                tot_raw = payload.get("total", None)

                def _to_int_or_none(x: Any) -> Optional[int]:
                    try:
                        return int(x) if x is not None else None
                    except Exception:
                        return None

                return data, _to_int_or_none(nxt_raw), _to_int_or_none(tot_raw)

            if resp.status_code in retryable_statuses:
                if total_wait_s >= max_total_wait_s:
                    resp.raise_for_status()

                retry_after = resp.headers.get("Retry-After")
                wait_s = step_s
                if retry_after and retry_after.isdigit():
                    wait_s = max(wait_s, int(retry_after))

                wait_s = min(wait_s, max_total_wait_s - total_wait_s)
                retry_count += 1
                print(
                    f"[SemanticScholar] Temporary error ({resp.status_code}); retry {retry_count} in {wait_s}s..."
                )
                time.sleep(wait_s)
                total_wait_s += wait_s
                continue

            resp.raise_for_status()

        except Exception:
            if total_wait_s >= max_total_wait_s:
                raise
            wait_s = min(step_s, max_total_wait_s - total_wait_s)
            retry_count += 1
            print(f"[SemanticScholar] Request failed; retry {retry_count} in {wait_s}s...")
            time.sleep(wait_s)
            total_wait_s += wait_s


def safe_filename(title: str) -> str:
    base = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    return (base[:80] or "paper") + ".pdf"


def download_pdf(pdf_dir: Path, title: str, url: str) -> Optional[Path]:
    import requests
    out_path = pdf_dir / safe_filename(title)
    if out_path.exists():
        return out_path

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        it = resp.iter_content(8192)
        first_chunk = next(it, b"")
        if not first_chunk:
            return None

        is_pdf = url.lower().endswith(".pdf") or first_chunk.lstrip().startswith(b"%PDF")
        if not is_pdf:
            return None

        with open(out_path, "wb") as f:
            f.write(first_chunk)
            for chunk in it:
                if chunk:
                    f.write(chunk)

        return out_path
    except Exception:
        return None


def run_download_top_pdfs(
    function_name: str,
    task: str,
    folder: str,
    api_key: str = "",
    top_n: int = 5,
    clear: bool = True,
    *,
    page_size: int = 100,
    max_search_results: int = 1000,
    require_open_access_pdf: bool = True,
) -> List[Dict[str, Any]]:
    top_n = min(top_n, MAX_PAPER_SEARCHES)

    search_dir = Path(folder) / "papers"
    pdf_dir = search_dir / "pdfs"

    if clear and search_dir.exists():
        for path in search_dir.glob("**/*"):
            if path.is_file():
                path.unlink()

    pdf_dir.mkdir(parents=True, exist_ok=True)

    all_papers: List[Dict[str, Any]] = []
    selected: List[Dict[str, Any]] = []

    offset = 0
    attempts = 0

    while len(selected) < top_n and offset is not None and offset < max_search_results:
        try:
            papers, next_offset, total = search_papers_page(
                task,
                api_key=api_key,
                limit=page_size,
                offset=offset,
                require_open_access_pdf=require_open_access_pdf,
            )
        except Exception as exc:
            print(f"[SemanticScholar] Search failed after retries; continuing without PDFs ({type(exc).__name__}: {exc})")
            break

        all_papers.extend(papers)
        offset = next_offset
        attempts += 1

        for paper in papers:
            if len(selected) >= top_n:
                break

            title = paper.get("title", "")
            open_access = paper.get("openAccessPdf") or {}
            pdf_url = open_access.get("url", "")
            if not pdf_url:
                continue

            pdf_path = download_pdf(pdf_dir, title, pdf_url)
            if not pdf_path:
                continue

            paper = dict(paper)
            paper["local_pdf"] = str(pdf_path)
            paper["pdf_path"] = str(pdf_path)
            selected.append(paper)

        if total is not None and offset is None:
            break

        if not papers:
            break

    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "raw_search.json").write_text(
        json.dumps(all_papers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (search_dir / "selected.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return selected


__all__ = [
    "download_pdf",
    "MAX_PAPER_SEARCHES",
    "run_download_top_pdfs",
    "safe_filename",
    "search_papers_page",
]
