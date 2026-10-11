from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT_JS = ROOT / "chrome-extension" / "content.js"
STYLE_CSS = ROOT / "chrome-extension" / "style.css"


def test_render_sources_iterates_accumulated_source_history():
    content = CONTENT_JS.read_text(encoding="utf-8")
    start = content.index("function renderSources()")
    end = content.index("// 13. SEEK YOUTUBE VIDEO", start)
    renderer = content[start:end]

    assert "sourceHistory.forEach(" in renderer
    assert "(source, index) =>" in renderer
    assert "sources.forEach(" not in renderer
    assert "const sourceId = index + 1" in renderer


def test_source_list_is_collapsed_by_default_and_accessibly_toggleable():
    content = CONTENT_JS.read_text(encoding="utf-8")
    start = content.index("function renderSources()")
    end = content.index("// 13. SEEK YOUTUBE VIDEO", start)
    renderer = content[start:end]

    assert 'sourcesToggle.setAttribute("aria-expanded", "false")' in renderer
    assert 'list.hidden = true' in renderer
    assert 'sourcesToggle.setAttribute(\n                "aria-expanded"' in renderer
    assert 'list.hidden = !expanded' in renderer
    assert '"Video sources · " + sourceHistory.length' in renderer
    assert 'aria-controls", "chat-sources-list"' in renderer


def test_expanded_source_list_has_bounded_internal_scrolling():
    css = STYLE_CSS.read_text(encoding="utf-8")
    start = css.index(".sources-list {")
    end = css.index("}", start)
    source_rules = css[start:end]

    assert "max-height: 160px" in source_rules
    assert "overflow-y: auto" in source_rules
