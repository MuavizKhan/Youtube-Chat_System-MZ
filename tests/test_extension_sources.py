from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT_JS = ROOT / "chrome-extension" / "content.js"


def test_render_sources_iterates_accumulated_source_history():
    content = CONTENT_JS.read_text(encoding="utf-8")
    start = content.index("function renderSources()")
    end = content.index("// 13. SEEK YOUTUBE VIDEO", start)
    renderer = content[start:end]

    # Render from the same collection populated by mergeSources, and provide
    # an index for the visible "Source N" label.
    assert "sourceHistory.forEach(" in renderer
    assert "(source, index) =>" in renderer
    assert "sources.forEach(" not in renderer
    assert "const sourceId =" in renderer
    assert "index + 1" in renderer
