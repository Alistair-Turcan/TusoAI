import re
import json
import requests
import fitz
from pathlib import Path
from typing import List, Optional, Dict

# ------------- CONFIG -------------------------------------------------- #
SEMANTIC_SCHOLAR_API_KEY = ""  # <-- add this

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,abstract,citationCount,isOpenAccess,openAccessPdf,venue"
PDF_DIR = Path("pdfs"); PDF_DIR.mkdir(exist_ok=True, parents=True)

NATURE_VENUES = {
    "NATURE", "NATURE AFRICA", "NATURE AGING", "NATURE ASTRONOMY",
    "NATURE BIOMEDICAL ENGINEERING", "NATURE BIOTECHNOLOGY", "NATURE CANCER",
    "NATURE CARDIOVASCULAR RESEARCH", "NATURE CATALYSIS", "NATURE CELL BIOLOGY",
    "NATURE CHEMICAL BIOLOGY", "NATURE CHEMICAL ENGINEERING", "NATURE CHEMISTRY",
    "NATURE CITIES", "NATURE CLIMATE CHANGE", "NATURE COMMUNICATIONS",
    "NATURE COMPUTATIONAL SCIENCE", "NATURE DIGEST", "NATURE ECOLOGY & EVOLUTION",
    "NATURE ELECTRONICS", "NATURE ENERGY", "NATURE FOOD", "NATURE GENETICS",
    "NATURE GEOSCIENCE", "NATURE HEALTH", "NATURE HUMAN BEHAVIOUR",
    "NATURE IMMUNOLOGY", "NATURE INDIA", "NATURE ITALY",
    "NATURE MACHINE INTELLIGENCE", "NATURE MATERIALS", "NATURE MEDICINE",
    "NATURE MENTAL HEALTH", "NATURE METABOLISM", "NATURE METHODS",
    "NATURE MICROBIOLOGY", "NATURE NANOTECHNOLOGY", "NATURE NEUROSCIENCE",
    "NATURE PHOTONICS", "NATURE PHYSICS", "NATURE PLANTS", "NATURE PROTOCOLS",
    "NATURE SENSORS", "NATURE STRUCTURAL & MOLECULAR BIOLOGY",
    "NATURE SUSTAINABILITY", "NATURE SYNTHESIS", "NATURE WATER"
}

# ------------- SEMANTIC SCHOLAR --------------------------------------- #
def search_top_cited(query: str, top_n: int = 5, overshoot: int = 100) -> List[dict]:
    params = {"query": query, "limit": overshoot, "fields": FIELDS}
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    resp = requests.get(SEMANTIC_SCHOLAR_API, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    def ok(p: dict) -> bool:
        venue = (p.get("venue") or "").upper()
        return any(n in venue for n in NATURE_VENUES) and bool(p.get("openAccessPdf", {}).get("url"))

    filtered = [p for p in data if ok(p)]
    filtered.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
    #print(filtered)
    return filtered[:top_n]

def load_or_fetch_metadata(query: str, METADATA_PATH: Path, top_n: int = 5) -> List[dict]:
    """
    Cache metadata along with the query and top_n.
    Invalidate cache if query or top_n changes.
    """
    if METADATA_PATH.exists():
        cached = json.loads(METADATA_PATH.read_text())
        if cached.get("query") == query and cached.get("top_n") == top_n:
            print("[CACHE] Loading metadata")
            return cached["papers"]
    # Otherwise fetch fresh
    print("[API] Fetching metadata from Semantic Scholar")
    papers = search_top_cited(query, top_n=top_n, overshoot=100)
    METADATA_PATH.write_text(json.dumps({
        "query": query,
        "top_n": top_n,
        "papers": papers
    }, indent=2))
    return papers

# ------------- PDF HELPERS -------------------------------------------- #
def safe_filename(title: str) -> str:
    fn = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    return (fn[:60] or "paper") + ".pdf"

def download_pdf(title: str, url: str) -> Optional[Path]:
    out = PDF_DIR / safe_filename(title)
    if out.exists():
        print(f"[SKIP] PDF already downloaded: {out.name}")
        return out
    print(f"[DOWNLOADING] {title}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(out, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return out
    except Exception as e:
        print(f"[WARN] Download failed for '{title}': {e}")
        return None

# ------------- FITZ TEXT LINES ---------------------------------------- #
def pdf_lines(pdf_path: Path) -> List[str]:
    """
    Extract text lines in reading order via PyMuPDF,
    joining hyphen‑split words.
    """
    doc = fitz.open(pdf_path)
    blocks = []
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for ln in b["lines"]:
                text = "".join(sp["text"] for sp in ln["spans"]).strip()
                if text:
                    blocks.append((page.number, ln["bbox"][1], ln["bbox"][0], text))
    doc.close()

    blocks.sort(key=lambda x: (x[0], x[1], x[2]))
    lines, buf = [], ""
    for _, _, _, txt in blocks:
        if txt.endswith("-"):
            buf += txt[:-1]
        else:
            line = (buf + txt).strip()
            lines.append(line)
            buf = ""
    return lines

# ------------- METHODS EXTRACTION ------------------------------------- #
def extract_methods(lines: List[str]) -> List[str]:
    """
    Return lines between exact 'Methods' and 'Data availability' headers (case-sensitive).
    """
    try:
        start = next(i for i, l in enumerate(lines) if l == "Methods")
    except StopIteration:
        return []
    try:
        end = next(i for i, l in enumerate(lines[start+1:], start+1) if l == "Data availability")
    except StopIteration:
        end = len(lines)
    segment = lines[start:end]
    return segment if len(segment) >= 5 else []

# ------------- SPLIT BY PERIOD ---------------------------------------- #
def split_by_period(method_lines: List[str]) -> List[str]:
    """
    Chunk lines into sections: collect lines until one ends with a period,
    then start a new section.
    """
    sections, current = [], []
    for ln in method_lines:
        current.append(ln)
        if ln.rstrip().endswith("."):
            sections.append(" ".join(current).strip())
            current = []
    if current:
        sections.append(" ".join(current).strip())
    return sections

# ------------- DRIVER ------------------------------------------------- #
def run_extraction(task: str, api_key='', top_n: int = 5) -> List[Dict]:
    query = f'{task}'
    METADATA_PATH = Path(f"{task}.json")
    SEMANTIC_SCHOLAR_API_KEY = api_key
    papers = load_or_fetch_metadata(query, METADATA_PATH, top_n)
    results = []
    #print(papers)
    for p in papers:
        title = p["title"]
        abstract = p.get("abstract", "")
        pdf_path = download_pdf(title, p["openAccessPdf"]["url"])
        if not pdf_path:
            continue

        lines = pdf_lines(pdf_path)
        method_lines = extract_methods(lines)
        sections = split_by_period(method_lines)

        results.append({
            "title": title,
            "abstract": abstract,
            "method_lines": method_lines,
            "sections": sections
        })
    return results