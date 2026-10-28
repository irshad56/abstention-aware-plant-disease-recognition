"""Evidence corpus, ingestion, search and citation checks.

The evidence the app shows comes only from pages listed in
data/evidence/sources.yaml:

    python scripts/build_evidence.py     -> fetches the pages, stores their real
                                            text in data/evidence/corpus/documents.jsonl
    EvidenceIndex.from_config(config)    -> TF-IDF search over that stored text
    build_citations(hits)                -> numbered citations with real URLs

Nothing is generated here. A citation always points at a stored document; if no
evidence exists for a disease the index returns nothing and the caller says so.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import yaml

from plant_disease.config import get_logger, resolve_path

log = get_logger("retrieval")

SOURCE_TYPES = {"government", "university_extension", "plant_pathology_society", "peer_reviewed", "other"}
MIN_DOCUMENT_CHARS = 300
MIN_PASSAGE_CHARS = 80
MIN_QUOTE_WORDS = 5
CITATION_MARKER = re.compile(r"\[(\d+)\]")


class RetrievalError(Exception):
    """A problem with sources, fetching or the corpus."""


# ==========================================================================
# Data classes
# ==========================================================================
@dataclass
class SourceEntry:
    id: str
    url: str
    publisher: str
    source_type: str
    classes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    title: str = ""
    local_file: str = ""


@dataclass
class EvidenceDoc:
    source_id: str
    url: str
    title: str
    publisher: str
    source_type: str
    retrieved_utc: str
    origin: str                      # "web" or "local_file"
    classes: List[str]
    keywords: List[str]
    text: str
    content_sha256: str


@dataclass
class Passage:
    passage_id: str
    source_id: str
    url: str
    title: str
    publisher: str
    source_type: str
    retrieved_utc: str
    origin: str
    classes: List[str]
    keywords: List[str]
    text: str


@dataclass
class EvidenceHit:
    passage: Passage
    score: float
    match: str                       # "class_tag" or "keyword"


@dataclass
class Citation:
    number: int
    source_id: str
    url: str
    title: str
    publisher: str
    source_type: str
    retrieved_utc: str
    origin: str


# ==========================================================================
# Small helpers
# ==========================================================================
def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def is_allowed_url(url: str, allowed_domains: Sequence[str]) -> bool:
    """True for http(s) URLs whose host matches an allowed domain or suffix."""
    parsed = urlparse(str(url))
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    for rule in allowed_domains:
        rule = str(rule).lower().strip()
        if not rule:
            continue
        if rule.startswith("."):
            if host.endswith(rule):
                return True
        elif host == rule or host.endswith("." + rule):
            return True
    return False


def split_class_name(class_name: str) -> Tuple[str, str]:
    """'Apple__Apple Scab' -> ('Apple', 'Apple Scab')."""
    parts = [p.strip() for p in re.split(r"_{2,}", class_name) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return "", class_name.strip()


def is_healthy_class(class_name: str) -> bool:
    return "healthy" in class_name.lower()


# ==========================================================================
# Sources file
# ==========================================================================
def load_sources(path: Path, allowed_domains: Sequence[str]) -> Tuple[List[SourceEntry], List[str]]:
    """Read sources.yaml. Returns (valid entries, problems). Bad entries are reported, not used."""
    path = Path(path)
    if not path.is_file():
        raise RetrievalError(f"Sources file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RetrievalError(f"Could not parse {path.name}: {exc}") from exc
    raw = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise RetrievalError(f"{path.name} must contain a 'sources:' list.")

    entries: List[SourceEntry] = []
    problems: List[str] = []
    seen: set = set()
    for i, item in enumerate(raw, 1):
        label = f"entry {i}"
        if not isinstance(item, dict):
            problems.append(f"{label}: not a mapping")
            continue
        sid = str(item.get("id", "")).strip()
        label = f"'{sid}'" if sid else label
        missing = [k for k in ("id", "url", "publisher", "source_type") if not str(item.get(k, "")).strip()]
        if missing:
            problems.append(f"{label}: missing {missing}")
            continue
        if sid in seen:
            problems.append(f"{label}: duplicate id")
            continue
        if item["source_type"] not in SOURCE_TYPES:
            problems.append(f"{label}: source_type must be one of {sorted(SOURCE_TYPES)}")
            continue
        if not is_allowed_url(item["url"], allowed_domains):
            problems.append(f"{label}: URL domain not in retrieval.allowed_domains: {item['url']}")
            continue
        seen.add(sid)
        entries.append(SourceEntry(
            id=sid, url=str(item["url"]).strip(), publisher=str(item["publisher"]).strip(),
            source_type=item["source_type"],
            classes=[str(c) for c in item.get("classes") or []],
            keywords=[str(k) for k in item.get("keywords") or []],
            title=str(item.get("title") or "").strip(),
            local_file=str(item.get("local_file") or "").strip(),
        ))
    return entries, problems


# ==========================================================================
# HTML text extraction (standard library only)
# ==========================================================================
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "li", "ul", "ol", "table", "tr", "br", "blockquote",
    "pre", "dd", "dt", "dl", "h1", "h2", "h3", "h4", "h5", "h6", "figure", "figcaption", "td", "th",
}
_ALWAYS_SKIP = {"script", "style", "noscript", "svg", "template"}
_BOILERPLATE_SKIP = _ALWAYS_SKIP | {"nav", "header", "footer", "aside", "form", "button", "select", "iframe"}


class _TextExtractor(HTMLParser):
    def __init__(self, skip_tags: set) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_tags = skip_tags
        self.skip_depth = 0
        self.in_title = False
        self.main_depth = 0
        self.title_parts: List[str] = []
        self.all_parts: List[str] = []
        self.main_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.skip_tags:
            self.skip_depth += 1
            return
        if tag == "title":
            self.in_title = True
        if tag in ("main", "article"):
            self.main_depth += 1
        if tag in _BLOCK_TAGS and not self.skip_depth:
            self.all_parts.append("\n")
            if self.main_depth:
                self.main_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.skip_tags:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if tag == "title":
            self.in_title = False
        if tag in ("main", "article"):
            self.main_depth = max(0, self.main_depth - 1)
        if tag in _BLOCK_TAGS and not self.skip_depth:
            self.all_parts.append("\n")
            if self.main_depth:
                self.main_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
            return
        if self.skip_depth:
            return
        self.all_parts.append(data)
        if self.main_depth:
            self.main_parts.append(data)


def _tidy(parts: List[str]) -> str:
    text = "".join(parts).replace("\r", "")
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.split("\n")]
    out: List[str] = []
    for ln in lines:
        if len(ln) < 3:
            if out and out[-1] != "":
                out.append("")                       # keep one blank line as paragraph break
            continue
        if out and out[-1] == ln:
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def extract_html(html: str) -> Tuple[str, str]:
    """Return (title, main text) from an HTML page, dropping menus, scripts and footers."""
    best_title, best_text = "", ""
    for skip in (_BOILERPLATE_SKIP, _ALWAYS_SKIP):            # strict first, then relaxed
        parser = _TextExtractor(skip)
        try:
            parser.feed(html)
            parser.close()
        except Exception:  # noqa: BLE001 - malformed HTML
            continue
        title = re.sub(r"\s+", " ", "".join(parser.title_parts)).strip()
        main_text = _tidy(parser.main_parts)
        text = main_text if len(main_text) >= 400 else _tidy(parser.all_parts)
        if len(text) > len(best_text):
            best_title, best_text = title, text
        if len(text) >= MIN_DOCUMENT_CHARS:
            return title, text
    return best_title, best_text


# ==========================================================================
# Fetching and ingestion
# ==========================================================================
def _robots_allows(url: str, user_agent: str, timeout: float) -> bool:
    import requests

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        resp = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=timeout)
        if resp.status_code != 200:
            return True
        rp = RobotFileParser()
        rp.parse(resp.text.splitlines())
        return rp.can_fetch(user_agent, url)
    except Exception:  # noqa: BLE001 - robots.txt unreachable: cannot tell, proceed
        return True


def fetch_page(
    url: str,
    allowed_domains: Sequence[str],
    timeout: float = 20,
    max_bytes: int = 5 * 1024 * 1024,
    user_agent: str = "PlantDiseaseEvidenceBot/0.1",
    respect_robots: bool = True,
) -> Tuple[str, str]:
    """Download one HTML/text page. Returns (text, final_url). Raises RetrievalError."""
    import requests

    if not is_allowed_url(url, allowed_domains):
        raise RetrievalError("domain not allowed")
    if respect_robots and not _robots_allows(url, user_agent, timeout):
        raise RetrievalError("disallowed by the site's robots.txt")
    headers = {"User-Agent": user_agent}
    current = url
    resp = None
    for _ in range(6):                      # follow redirects by hand: every hop must be allowed
        try:
            resp = requests.get(current, headers=headers, timeout=timeout, stream=True,
                                allow_redirects=False)
        except requests.RequestException as exc:
            raise RetrievalError(f"request failed: {type(exc).__name__}: {str(exc)[:120]}") from exc
        if resp.status_code in (301, 302, 303, 307, 308):
            target = urljoin(current, resp.headers.get("Location", ""))
            resp.close()
            if not is_allowed_url(target, allowed_domains):
                raise RetrievalError(f"redirected to a domain that is not allowed: {target}")
            current = target
            continue
        break
    else:
        raise RetrievalError("too many redirects")
    final_url = current
    if resp.status_code != 200:
        raise RetrievalError(f"HTTP {resp.status_code}")
    ctype = resp.headers.get("Content-Type", "").lower()
    if "html" not in ctype and "text/plain" not in ctype:
        raise RetrievalError(
            f"unsupported content type '{ctype or 'unknown'}' (PDFs: save as text and use local_file)"
        )
    chunks, size = [], 0
    for block in resp.iter_content(chunk_size=65536):
        size += len(block)
        if size > max_bytes:
            raise RetrievalError(f"page larger than {max_bytes // (1024 * 1024)} MB")
        chunks.append(block)
    raw = b"".join(chunks)
    encoding = resp.encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace"), final_url
    except LookupError:
        return raw.decode("utf-8", errors="replace"), final_url


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ingest_sources(
    config: Dict[str, Any],
    entries: Sequence[SourceEntry],
    delay_s: float = 1.0,
    respect_robots: bool = True,
    fetcher: Optional[Callable[[str], Tuple[str, str]]] = None,
) -> Tuple[List[EvidenceDoc], List[Dict[str, Any]]]:
    """Fetch (or read) every source. Returns (documents, per-source report).

    A failed source is reported and skipped; it never produces a document.
    `fetcher(url) -> (text, final_url)` can be injected for testing.
    """
    rc = config["retrieval"]
    allowed = rc["allowed_domains"]
    max_bytes = int(rc.get("max_download_mb", 5)) * 1024 * 1024
    if fetcher is None:
        def fetcher(url: str) -> Tuple[str, str]:          # noqa: E306
            return fetch_page(url, allowed, float(rc.get("request_timeout_s", 20)), max_bytes,
                              rc.get("user_agent", "PlantDiseaseEvidenceBot/0.1"), respect_robots)

    docs: List[EvidenceDoc] = []
    report: List[Dict[str, Any]] = []
    first_web = True
    for entry in entries:
        row: Dict[str, Any] = {"id": entry.id, "url": entry.url, "status": "failed", "reason": "",
                               "title": "", "chars": 0, "origin": "local_file" if entry.local_file else "web"}
        try:
            if entry.local_file:
                path = resolve_path(entry.local_file)
                if path is None or not path.is_file():
                    raise RetrievalError(f"local file not found: {entry.local_file}")
                raw = path.read_text(encoding="utf-8", errors="replace")
                if path.suffix.lower() in (".html", ".htm"):
                    title, text = extract_html(raw)
                else:
                    title, text = "", re.sub(r"\n{3,}", "\n\n", raw).strip()
            else:
                if not first_web and delay_s > 0:
                    time.sleep(delay_s)                         # be polite to the sites
                first_web = False
                html, final_url = fetcher(entry.url)
                title, text = extract_html(html) if "<" in html[:2000] else ("", html.strip())
            title = entry.title or title or entry.id
            if len(text) < MIN_DOCUMENT_CHARS:
                raise RetrievalError(
                    f"only {len(text)} characters of text extracted (page may need JavaScript or be empty)"
                )
            docs.append(EvidenceDoc(
                source_id=entry.id, url=entry.url, title=title, publisher=entry.publisher,
                source_type=entry.source_type, retrieved_utc=_now(), origin=row["origin"],
                classes=entry.classes, keywords=entry.keywords, text=text,
                content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            ))
            row.update(status="ok", title=title, chars=len(text))
        except RetrievalError as exc:
            row["reason"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            row["reason"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        report.append(row)
    return docs, report


def save_corpus(path: Path, docs: Sequence[EvidenceDoc]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_corpus(path: Path) -> List[EvidenceDoc]:
    """Read the stored corpus; an absent file means 'no evidence yet' (empty list)."""
    path = Path(path)
    if not path.is_file():
        return []
    docs: List[EvidenceDoc] = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(EvidenceDoc(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise RetrievalError(f"{path.name} line {n} is not a valid document: {exc}") from exc
    return docs


# ==========================================================================
# Passages and search
# ==========================================================================
def chunk_text(text: str, chunk_chars: int = 900, overlap: int = 150) -> List[str]:
    """Split text into passages of about `chunk_chars`, breaking at paragraph/sentence ends."""
    chunk_chars = max(200, int(chunk_chars))
    overlap = max(0, min(int(overlap), chunk_chars // 2))
    units: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip().replace("\n", " ")
        if not para:
            continue
        if len(para) <= chunk_chars:
            units.append(para)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            while len(sentence) > chunk_chars:                       # very long run-on text
                cut = sentence.rfind(" ", 0, chunk_chars)
                cut = cut if cut > chunk_chars // 2 else chunk_chars
                units.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                units.append(sentence)

    chunks: List[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > chunk_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if tail and " " in tail else tail
            current = (tail + " " + unit).strip() if tail else unit
        else:
            current = (current + " " + unit).strip() if current else unit
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= MIN_PASSAGE_CHARS]


def build_passages(docs: Sequence[EvidenceDoc], chunk_chars: int = 900, overlap: int = 150) -> List[Passage]:
    passages: List[Passage] = []
    for d in docs:
        for k, chunk in enumerate(chunk_text(d.text, chunk_chars, overlap)):
            passages.append(Passage(
                passage_id=f"{d.source_id}#{k}", source_id=d.source_id, url=d.url, title=d.title,
                publisher=d.publisher, source_type=d.source_type, retrieved_utc=d.retrieved_utc,
                origin=d.origin, classes=list(d.classes), keywords=list(d.keywords), text=chunk,
            ))
    return passages


class EvidenceIndex:
    """TF-IDF search restricted to passages relevant to the predicted class.

    A passage is a candidate only if its source lists the predicted class, or one of
    its keywords clearly names the disease. Ranking inside the candidates uses TF-IDF
    similarity, and anything below `min_score` is dropped.
    """

    def __init__(self, passages: Sequence[Passage], notice: str = "") -> None:
        self.passages: List[Passage] = list(passages)
        self.notice = notice
        self._vectorizer = None
        self._matrix = None
        if self.passages:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                                   sublinear_tf=True)
                self._matrix = self._vectorizer.fit_transform([p.text for p in self.passages])
            except ImportError as exc:
                raise RetrievalError("scikit-learn is required for evidence search.") from exc
            except ValueError:                       # empty vocabulary
                self._vectorizer = self._matrix = None
                self.notice = "The evidence corpus has no searchable text."

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EvidenceIndex":
        corpus_path = config["paths"]["evidence_corpus"]
        rc = config["retrieval"]
        docs = load_corpus(corpus_path)
        if not docs:
            return cls([], notice="No evidence corpus found. Run: python scripts\\build_evidence.py")
        return cls(build_passages(docs, int(rc.get("chunk_chars", 900)), int(rc.get("chunk_overlap", 150))))

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    def stats(self) -> Dict[str, Any]:
        sources = sorted({p.source_id for p in self.passages})
        classes = sorted({c for p in self.passages for c in p.classes})
        return {"passages": len(self.passages), "sources": sources, "classes_covered": classes,
                "ready": self.ready, "notice": self.notice}

    def has_evidence_for(self, class_name: str) -> bool:
        return bool(self._candidates(class_name))

    def _candidates(self, class_name: str) -> List[Tuple[int, str]]:
        _, disease = split_class_name(class_name)
        disease_norm = _norm(disease)
        found: List[Tuple[int, str]] = []
        for i, p in enumerate(self.passages):
            if class_name in p.classes:
                found.append((i, "class_tag"))
                continue
            for kw in p.keywords:
                kn = _norm(kw)
                if len(kn) >= 8 and disease_norm and (kn in disease_norm or disease_norm in kn):
                    found.append((i, "keyword"))
                    break
        return found

    def search(
        self,
        class_name: str,
        top_k: int = 4,
        min_score: float = 0.05,
        max_per_source: int = 2,
        query: Optional[str] = None,
    ) -> List[EvidenceHit]:
        """Best passages for a predicted class, or [] when no real evidence is available."""
        if not self.ready or is_healthy_class(class_name):
            return []
        cands = self._candidates(class_name)
        if not cands:
            return []
        crop, disease = split_class_name(class_name)
        q = query or " ".join(dict.fromkeys(
            [disease, crop, "symptoms", "cause", "management", "control", "treatment"]))
        qvec = self._vectorizer.transform([q])
        idx = [i for i, _ in cands]
        scores = (self._matrix[idx] @ qvec.T).toarray().ravel()
        order = sorted(range(len(idx)), key=lambda j: -scores[j])
        hits: List[EvidenceHit] = []
        per_source: Dict[str, int] = {}
        for j in order:
            score = float(scores[j])
            if score < min_score:
                break
            passage = self.passages[idx[j]]
            if per_source.get(passage.source_id, 0) >= max_per_source:
                continue
            per_source[passage.source_id] = per_source.get(passage.source_id, 0) + 1
            hits.append(EvidenceHit(passage, score, cands[j][1]))
            if len(hits) >= top_k:
                break
        return hits


# ==========================================================================
# Citations and checks
# ==========================================================================
def build_citations(hits: Sequence[EvidenceHit]) -> Tuple[List[Citation], List[int]]:
    """Numbered citations (one per source, in order of first appearance).

    Returns (citations, number_for_each_hit).
    """
    citations: List[Citation] = []
    by_source: Dict[str, int] = {}
    numbers: List[int] = []
    for h in hits:
        p = h.passage
        if p.source_id not in by_source:
            by_source[p.source_id] = len(citations) + 1
            citations.append(Citation(
                number=by_source[p.source_id], source_id=p.source_id, url=p.url, title=p.title,
                publisher=p.publisher, source_type=p.source_type,
                retrieved_utc=p.retrieved_utc, origin=p.origin))
        numbers.append(by_source[p.source_id])
    return citations, numbers


def validate_hits(hits: Sequence[EvidenceHit], allowed_domains: Sequence[str]) -> Tuple[List[EvidenceHit], List[str]]:
    """Keep only hits with a real, allowed URL, a title and non-empty text."""
    good: List[EvidenceHit] = []
    problems: List[str] = []
    for h in hits:
        p = h.passage
        if not is_allowed_url(p.url, allowed_domains):
            problems.append(f"{p.source_id}: URL is missing or not on an allowed domain")
        elif not p.text.strip() or not p.publisher.strip():
            problems.append(f"{p.source_id}: passage or publisher is empty")
        else:
            good.append(h)
    return good, problems


def citation_marker_problems(text: str, n_citations: int) -> List[str]:
    """Problems with [n] markers: none present, or numbers outside 1..n_citations."""
    markers = [int(m) for m in CITATION_MARKER.findall(text)]
    problems = []
    if n_citations and not markers:
        problems.append("the explanation contains no citation markers")
    bad = sorted({m for m in markers if not 1 <= m <= n_citations})
    if bad:
        problems.append(f"citation marker(s) {bad} do not match any source")
    return problems


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_unsupported_quotes(text: str, passages: Sequence[Passage]) -> List[str]:
    """Quoted phrases (5+ words) that do not appear in any retrieved passage."""
    haystack = " ".join(_squash(p.text) for p in passages)
    quotes = re.findall(r"[\"\u201c]([^\"\u201d]{10,400})[\"\u201d]", text)
    bad = []
    for q in quotes:
        squashed = _squash(q)
        if len(squashed.split()) >= MIN_QUOTE_WORDS and squashed not in haystack:
            bad.append(q.strip())
    return bad
