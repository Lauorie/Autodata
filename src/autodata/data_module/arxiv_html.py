"""Convert an arXiv HTML page (LaTeXML output) to the markdown the pipeline reads.

The paper draws CS papers from S2ORC. That corpus is not on hand, so the
smoke and gate runs use arXiv's own HTML renderings instead
(scripts/fetch_arxiv_papers.py). The body keeps what S2ORC full text keeps:
abstract, sections, captions, tables, and math as LaTeX source. Authors,
bibliography, and images go.

No markdown escaping: the reader is a model, not a renderer, and escaping
would double every backslash in the LaTeX.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

_DROP = (
    "script", "style", "img",
    "h1.ltx_title_document", ".ltx_authors", ".ltx_note_frontmatter",
    "section.ltx_bibliography", ".ltx_page_navbar", ".ltx_TOC",
)


class NoFullTextError(ValueError):
    """The page carries no LaTeXML article (arXiv had no HTML for the paper)."""


def _latex(math: Tag) -> str:
    return str(math.get("alttext", "")).strip()


def _flatten_equations(article: Tag, soup: BeautifulSoup) -> None:
    """Replace each display-equation table with one ``$$...$$`` paragraph."""
    for table in article.select("table.ltx_equation, table.ltx_equationgroup"):
        parts = [_latex(m) for m in table.find_all("math")]
        p = soup.new_tag("p")
        p.string = " ".join(f"$${s}$$" for s in parts if s)
        table.replace_with(p)


def _inline_math(article: Tag) -> None:
    for math in article.find_all("math"):
        fence = "$$" if math.get("display") == "block" else "$"
        math.replace_with(f"{fence}{_latex(math)}{fence}")


def html_to_markdown(html: str, title: str) -> str:
    """Return ``# title`` followed by the article body as markdown.

    Raises:
        NoFullTextError: when the page holds no ``article.ltx_document``.
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article", class_="ltx_document")
    if not isinstance(article, Tag):
        raise NoFullTextError("no LaTeXML article in page")
    for el in article.select(", ".join(_DROP)):
        el.decompose()
    _flatten_equations(article, soup)
    _inline_math(article)
    body = markdownify(
        str(article), heading_style="ATX", strip=["a"],
        escape_asterisks=False, escape_underscores=False, escape_misc=False,
    )
    lines = [ln.rstrip() for ln in body.splitlines()]
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return f"# {title.strip()}\n\n{text.strip()}\n"


__all__ = ["NoFullTextError", "html_to_markdown"]
