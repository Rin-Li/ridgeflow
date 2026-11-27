"""Fetch arXiv papers as local Markdown: the arXiv HTML rendering when it exists, else the PDF
through ``pdftotext``. Metadata (title, authors, date, abstract) comes from the arXiv API.

    python tools/fetch_papers.py                       # every id in tools/papers.txt
    python tools/fetch_papers.py 2410.16646 2402.03559 # just these
    python tools/fetch_papers.py --list                # what is already on disk

Files land in ``outputs/papers/<id>.md`` plus ``outputs/papers/INDEX.md``. Already-fetched
papers are skipped (``--force`` refetches). Requests are spaced 3 s apart, as arXiv asks.
Stdlib only; ``pdftotext`` (poppler-utils) is needed for the PDF fallback.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "tools" / "papers.txt"
OUT = ROOT / "outputs" / "papers"
AGENT = "paper-fetch/0.1 (research script; stdlib urllib)"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
_last = 0.0


def _get(url: str, binary: bool = False):
    global _last
    wait = 3.0 - (time.time() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
        return data if binary else data.decode("utf-8", "replace"), r.geturl()


def metadata(ids: list[str]) -> dict[str, dict]:
    xml, _ = _get("https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(ids))
    out = {}
    for e in ET.fromstring(xml).findall("a:entry", ATOM):
        raw = e.findtext("a:id", "", ATOM).rsplit("/abs/", 1)[-1]
        pid = re.sub(r"v\d+$", "", raw)
        out[pid] = {
            "title": " ".join(e.findtext("a:title", "", ATOM).split()),
            "authors": [a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)],
            "published": e.findtext("a:published", "", ATOM)[:10],
            "abstract": " ".join(e.findtext("a:summary", "", ATOM).split()),
        }
    return out


class _ArxivHTML(HTMLParser):
    """arXiv LaTeXML pages -> Markdown-ish text; math kept as its LaTeX ``alttext``."""

    SKIP = {"script", "style", "nav", "header", "footer", "button", "svg"}
    BLOCK = {"p", "div", "section", "li", "tr", "figcaption", "table", "br", "dd", "dt"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.math = 0
        self.in_article = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "article":
            self.in_article = True
        if tag in self.SKIP:
            self.skip += 1
            return
        if tag == "math":
            alt = a.get("alttext", "")
            display = a.get("display") == "block"
            self.parts.append(f"\n$$ {alt} $$\n" if display else f" ${alt}$ ")
            self.math += 1
            return
        if self.math:
            return
        m = re.fullmatch(r"h([1-6])", tag)
        if m:
            self.parts.append("\n\n" + "#" * int(m.group(1)) + " ")
        elif tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")
        if tag == "img" and a.get("alt"):
            self.parts.append(f"[figure: {a['alt']}]")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if tag == "math":
            self.math = max(0, self.math - 1)
            return
        if re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip or self.math:
            return
        self.parts.append(data)

    def text(self) -> str:
        s = "".join(self.parts)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" *\n *", "\n", s)
        return re.sub(r"\n{3,}", "\n\n", s).strip()


def html_text(pid: str) -> str | None:
    try:
        page, url = _get(f"https://arxiv.org/html/{pid}")
    except urllib.error.HTTPError:
        return None
    if "/html/" not in url or "ltx_document" not in page:
        return None  # redirected to the abstract page: no HTML rendering
    body = page[page.find("<article") :] if "<article" in page else page
    parser = _ArxivHTML()
    parser.feed(body)
    return parser.text()


def pdf_text(pid: str) -> str:
    data, _ = _get(f"https://arxiv.org/pdf/{pid}", binary=True)
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "paper.pdf"
        pdf.write_bytes(data)
        txt = Path(tmp) / "paper.txt"
        subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), str(txt)], check=True)
        return txt.read_text("utf-8", "replace")


def read_list() -> list[tuple[str, str, str]]:
    rows = []
    for line in LIST.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pid, group, *name = line.split()
            rows.append((pid, group, " ".join(name)))
    return rows


def write_index() -> None:
    lines = [
        "# Papers on disk\n",
        "| id | group | source | chars | title |",
        "|---|---|---|---|---|",
    ]
    for f in sorted(OUT.glob("*.md")):
        if f.name == "INDEX.md":
            continue
        head = f.read_text("utf-8").split("\n", 12)
        meta = {k: v for k, v in (h.split(": ", 1) for h in head if ": " in h and h[0] != "#")}
        title = head[0].lstrip("# ")
        lines.append(
            f"| {f.stem} | {meta.get('group', '')} | {meta.get('source', '')} | "
            f"{f.stat().st_size} | {title} |"
        )
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("ids", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.list:
        write_index()
        print((OUT / "INDEX.md").read_text())
        return

    known = {pid: (g, n) for pid, g, n in read_list()}
    ids = args.ids or list(known)
    todo = [i for i in ids if args.force or not (OUT / f"{i}.md").exists()]
    if not todo:
        print("nothing to fetch")
        write_index()
        return
    meta = {}
    for k in range(0, len(todo), 50):
        meta |= metadata(todo[k : k + 50])
    for pid in todo:
        m = meta.get(pid)
        if m is None:
            print(f"{pid}: not found on arXiv")
            continue
        group, name = known.get(pid, ("", ""))
        try:
            body, source = html_text(pid), "html"
            if not body or len(body) < 2000:
                body, source = pdf_text(pid), "pdf"
        except Exception as e:  # noqa: BLE001
            print(f"{pid}: failed ({e})")
            continue
        header = (
            f"# {m['title']}\n"
            f"arxiv: {pid}\n"
            f"group: {group}\n"
            f"short: {name}\n"
            f"authors: {', '.join(m['authors'][:8])}{' et al.' if len(m['authors']) > 8 else ''}\n"
            f"published: {m['published']}\n"
            f"source: {source}\n"
            f"url: https://arxiv.org/abs/{pid}\n\n"
            f"## Abstract\n\n{m['abstract']}\n\n---\n\n"
        )
        (OUT / f"{pid}.md").write_text(header + body, "utf-8")
        print(f"{pid}: {source:<4} {len(body):>7} chars  {m['title'][:70]}", flush=True)
    write_index()


if __name__ == "__main__":
    main()
