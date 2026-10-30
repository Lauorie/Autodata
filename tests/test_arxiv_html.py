"""arXiv HTML (LaTeXML) to the markdown the pipeline reads."""

import pytest

pytest.importorskip("bs4")
pytest.importorskip("markdownify")

from autodata.data_module.arxiv_html import NoFullTextError, html_to_markdown  # noqa: E402

_PAGE = """<html><head><style>.x{}</style><script>var a=1;</script></head><body>
<article class="ltx_document">
<h1 class="ltx_title ltx_title_document">A Title With <span>Markup</span></h1>
<div class="ltx_authors"><span class="ltx_personname">Jane Doe</span>
  <span class="ltx_note ltx_note_frontmatter">Corresponding: jane@uni.edu</span></div>
<div class="ltx_abstract"><h6 class="ltx_title">Abstract</h6>
  <p class="ltx_p">We study rank_r adapters.</p></div>
<section class="ltx_section"><h2 class="ltx_title"><span class="ltx_tag">1 </span>Method</h2>
<p class="ltx_p">The update is <math alttext="\\Delta W=BA" display="inline"><mi>Δ</mi></math>
  with rank <math alttext="r_{t}" display="inline"><mi>r</mi></math>, see
  <cite class="ltx_cite">[<a class="ltx_ref" href="#bib.bib1">1</a>]</cite>.</p>
<table class="ltx_equation ltx_eqn_table" id="S1.E1"><tbody><tr>
  <td class="ltx_eqn_cell"><math alttext="y=\\frac{\\alpha}{r}BAx," display="block"><mi>y</mi></math></td>
  <td class="ltx_eqn_cell ltx_eqn_eqno"><span class="ltx_tag">(1)</span></td></tr></tbody></table>
<figure class="ltx_figure"><img src="fig.png" alt="Refer to caption"/>
  <figcaption class="ltx_caption">Figure 1: Overview of the method.</figcaption></figure>
<table class="ltx_tabular"><tr><td>Model</td><td>Acc</td></tr><tr><td>Ours</td><td>91.2</td></tr></table>
</section>
<section class="ltx_bibliography"><h2 class="ltx_title">References</h2>
  <ul><li class="ltx_bibitem">[1] Some Prior Work. 2024.</li></ul></section>
</article></body></html>"""


@pytest.fixture(scope="module")
def md() -> str:
    return html_to_markdown(_PAGE, title="A Title With Markup")


def test_starts_with_the_given_title_as_h1(md: str) -> None:
    assert md.startswith("# A Title With Markup\n")
    assert md.count("\n# ") == 0                     # the page's own h1 is gone


def test_math_keeps_latex_source(md: str) -> None:
    assert "$\\Delta W=BA$" in md
    assert "$r_{t}$" in md                           # no markdown escaping inside math
    assert "$$y=\\frac{\\alpha}{r}BAx,$$" in md
    assert "Δ" not in md                             # MathML rendering dropped


def test_drops_authors_bibliography_and_images(md: str) -> None:
    for gone in ("Jane Doe", "jane@uni.edu", "Some Prior Work", "References",
                 "fig.png", "Refer to caption", "var a=1", ".x{}"):
        assert gone not in md, gone


def test_keeps_body_structure(md: str) -> None:
    assert "## 1 Method" in md
    assert "Abstract" in md and "rank_r adapters" in md
    assert "Figure 1: Overview of the method." in md
    assert "[1]" in md and "#bib" not in md          # citation marker, no link target
    assert "| Ours | 91.2 |" in md


def test_page_without_latexml_article_has_no_full_text() -> None:
    with pytest.raises(NoFullTextError):
        html_to_markdown("<html><body><p>No HTML version</p></body></html>", title="t")
