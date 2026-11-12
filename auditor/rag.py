"""Policy index: the eight project clauses once, plus the official EBA guideline."""

import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
import pymupdf
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from auditor import log

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "policies" / "credit_risk_eba_guidelines_rag.txt"
EBA_PATH = ROOT / "data" / "EBA_GL_2020_06.pdf"
CHROMA_PATH = ROOT / "data" / "chroma"
MANIFEST_PATH = ROOT / "data" / "policy_manifest.json"
COLLECTION_NAME = "credit_policies_v1"
CHUNKER_VERSION = 1

SOURCE_NOTE = (
    "The uploaded text file and the uploaded PDF are the same eight clauses. "
    "The PDF adds a trailing semicolon on the annex references and a shorter closing line on clause P08. "
    "The rules do not contradict each other. Indexing both would duplicate every clause, "
    "so the store keeps the text file once and adds the official 89-page EBA/GL/2020/06 guideline."
)

PARA = re.compile(r"^(\d{1,3})\.\s+\S")
SECTION = re.compile(r"^\d+(?:\.\d+){1,3}\s+\S")
ANNEX = re.compile(r"^Annex\b", re.I)


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_project_rules(text):
    matches = list(re.finditer(r"(?m)^\[(P0[1-8])\]\s+", text))
    ids = [match.group(1) for match in matches]
    if ids != [f"P{number:02d}" for number in range(1, 9)]:
        raise ValueError(f"Expected policy ids P01–P08, found {ids}.")

    rules = []
    feature_to_rules = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        rule = text[match.start() : end]
        rule = rule.split("[[PAGE BREAK]]")[0].split("\nSource PDF:")[0].strip()
        rule_id = match.group(1)
        mapping = re.search(
            r"(?m)^Project feature mapping \(not EBA text\):\s*(.*)$",
            rule,
        )
        features = []
        if mapping:
            features = [name.strip() for name in mapping.group(1).split(";") if name.strip()]
            for name in features:
                feature_to_rules.setdefault(name, [])
                if rule_id not in feature_to_rules[name]:
                    feature_to_rules[name].append(rule_id)
        rules.append({"id": rule_id, "text": rule, "features": features})
    return rules, feature_to_rules


def mapped_ids(feature, value, feature_to_rules):
    if feature == "credit_history":
        return ["P04"] if str(value) == "A33" else ["P03"]
    return list(feature_to_rules.get(feature, []))


def _is_toc(lines):
    content = [
        line
        for line in lines
        if line and not line.startswith("FINAL REPORT") and not re.fullmatch(r"\d{1,3}", line)
    ]
    if len(content) < 8:
        return False
    ending_with_page = sum(1 for line in content if re.search(r"\s\d{1,3}$", line))
    return ending_with_page / len(content) > 0.5


def _page_lines(page):
    lines = []
    previous = ""
    for raw in page.get_text("text").splitlines():
        line = raw.strip()
        if not line or line.startswith("FINAL REPORT") or re.fullmatch(r"\d{1,3}", line):
            continue
        if previous.endswith("-") and line[:1].islower():
            previous = previous[:-1] + line
            continue
        if previous:
            lines.append(previous)
        previous = line
    if previous:
        lines.append(previous)
    return lines


def _split_long(text, limit=1400):
    parts = []
    remaining = text.strip()
    while len(remaining) > limit:
        cut = remaining.rfind(". ", 400, limit)
        if cut < 400:
            cut = limit
        else:
            cut += 1
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def chunk_guideline(path):
    document = pymupdf.open(path)
    paragraphs = []
    current = None
    try:
        for index in range(document.page_count):
            lines = _page_lines(document[index])
            if _is_toc(lines):
                continue
            page = index + 1
            for line in lines:
                starts = bool(PARA.match(line) or SECTION.match(line) or ANNEX.match(line))
                if current is None:
                    current = {"page": page, "text": line}
                elif starts and len(current["text"]) >= 80:
                    paragraphs.append(current)
                    current = {"page": page, "text": line}
                else:
                    current["text"] = f"{current['text']} {line}"
        if current:
            paragraphs.append(current)
    finally:
        document.close()

    merged = []
    buffer = None
    for paragraph in paragraphs:
        if buffer is None:
            buffer = paragraph
            continue
        combined = len(buffer["text"]) + len(paragraph["text"])
        if len(buffer["text"]) < 250 and combined < 1600:
            buffer["text"] = f"{buffer['text']} {paragraph['text']}"
        else:
            merged.append(buffer)
            buffer = paragraph
    if buffer:
        merged.append(buffer)

    chunks = []
    counts = {}
    for paragraph in merged:
        for part in _split_long(paragraph["text"]):
            if len(part) < 180:
                continue
            page = paragraph["page"]
            counts[page] = counts.get(page, 0) + 1
            chunks.append(
                {
                    "id": f"EBA-p{page:03d}-{counts[page]:02d}",
                    "text": part,
                    "page": page,
                }
            )
    if len(chunks) < 40:
        raise ValueError(f"Guideline chunking produced only {len(chunks)} chunks.")
    return chunks


def _manifest():
    return {
        "chunker": CHUNKER_VERSION,
        "policy_sha256": _sha256(POLICY_PATH),
        "eba_sha256": _sha256(EBA_PATH),
    }


def open_store():
    if not POLICY_PATH.is_file():
        raise FileNotFoundError(POLICY_PATH)
    if not EBA_PATH.is_file():
        raise FileNotFoundError(EBA_PATH)

    rules, feature_to_rules = parse_project_rules(POLICY_PATH.read_text(encoding="utf-8"))
    expected = _manifest()
    log("loading the policy embedding model (all-MiniLM-L6-v2). The first run downloads it.")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    embedding = DefaultEmbeddingFunction()
    saved = {}
    if MANIFEST_PATH.is_file():
        saved = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    reuse = (
        saved.get("chunker") == expected["chunker"]
        and saved.get("policy_sha256") == expected["policy_sha256"]
        and saved.get("eba_sha256") == expected["eba_sha256"]
    )
    collection = None
    if reuse:
        try:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding)
        except Exception:
            collection = None
        if collection is not None and collection.count() != saved.get("count"):
            collection = None

    if collection is None:
        log("policy index is missing or stale. Reading the EBA guideline and embedding the chunks.")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        guideline = chunk_guideline(EBA_PATH)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding,
            metadata={"hnsw:space": "cosine"},
        )
        ids, documents, metadatas = [], [], []
        for rule in rules:
            ids.append(rule["id"])
            documents.append(rule["text"])
            metadatas.append(
                {
                    "kind": "paraphrase",
                    "source": "Project paraphrase of EBA/GL/2020/06",
                    "page": 0,
                    "features": ";".join(rule["features"]),
                }
            )
        for chunk in guideline:
            ids.append(chunk["id"])
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "kind": "guideline",
                    "source": "EBA/GL/2020/06",
                    "page": int(chunk["page"]),
                    "features": "",
                }
            )
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        expected["count"] = collection.count()
        expected["guideline_chunks"] = len(guideline)
        MANIFEST_PATH.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        log(f"policy index ready ({expected['count']} chunks) at {CHROMA_PATH.relative_to(ROOT)}")
    else:
        log(f"policy index reused ({collection.count()} chunks)")

    return {
        "collection": collection,
        "feature_to_rules": feature_to_rules,
        "count": collection.count(),
    }


def retrieve(store, focus):
    collection = store["collection"]
    feature_to_rules = store["feature_to_rules"]
    wanted = []
    for factor in focus:
        for rule_id in mapped_ids(factor["feature"], factor["value_code"], feature_to_rules):
            if rule_id not in wanted:
                wanted.append(rule_id)

    chunks = []
    if wanted:
        found = collection.get(ids=wanted, include=["documents", "metadatas"])
        by_id = {
            item_id: (document, metadata)
            for item_id, document, metadata in zip(
                found["ids"], found["documents"], found["metadatas"]
            )
        }
        for rule_id in wanted:
            if rule_id not in by_id:
                continue
            document, metadata = by_id[rule_id]
            chunks.append(_chunk(rule_id, document, metadata))

    query = "; ".join(f"{factor['label']}: {factor['value_label']}" for factor in focus)
    query = f"Consumer creditworthiness assessment. {query}"
    semantic = collection.query(
        query_texts=[query],
        n_results=min(4, max(collection.count(), 1)),
        where={"kind": "guideline"},
    )
    seen = {chunk["id"] for chunk in chunks}
    for item_id, document, metadata in zip(
        semantic["ids"][0], semantic["documents"][0], semantic["metadatas"][0]
    ):
        if item_id in seen:
            continue
        chunks.append(_chunk(item_id, document, metadata))
        seen.add(item_id)
    return {"query": query, "chunks": chunks}


def _chunk(item_id, document, metadata):
    return {
        "id": item_id,
        "text": document,
        "source": metadata.get("source", ""),
        "page": int(metadata.get("page") or 0),
        "kind": metadata.get("kind", ""),
    }
