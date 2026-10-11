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
    assert ".sources-list[hidden] {\n    display: none;" in css


def test_markdown_renderer_compacts_blank_lines_and_list_item_breaks():
    content = CONTENT_JS.read_text(encoding="utf-8")
    start = content.index("function renderMarkdown(text)")
    end = content.index("// REQUEST STATE RESET", start)
    renderer = content[start:end]

    assert r'.replace(/^[ \t]*\\[ \t]*$/gm, "")' in renderer
    assert r'.replace(/\n[ \t]*\n(?=[ \t]*[-*]\s+)/g, "\n")' in renderer
    assert r'html = html.replace(/<\/li>(?:<br\s*\/?>(?:\s*)?)+<li>/g, "</li><li>");' in renderer
    assert r'html = html.replace(/<ul>(?:<br\s*\/?>(?:\s*)?)+/g, "<ul>");' in renderer


def test_assistant_lists_use_compact_vertical_spacing():
    css = STYLE_CSS.read_text(encoding="utf-8")

    assert ".chat-message.assistant ul {" in css
    assert "margin: 0.35em 0 0.4em;" in css
    assert ".chat-message.assistant li {" in css
    assert "margin: 0 0 0.2em;" in css
