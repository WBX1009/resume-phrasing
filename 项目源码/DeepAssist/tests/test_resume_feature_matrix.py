"""The resume matrix is a completion gate, not a decorative checklist."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "resume_feature_matrix.md"


def test_resume_feature_matrix_has_no_unchecked_rows_and_named_tests_exist():
    text = MATRIX.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0].replace(".", "", 1).isdigit():
            rows.append(cells)

    assert rows, "resume feature matrix must contain numbered feature rows"
    assert all(row[-1] == "✅" for row in rows)

    test_sources = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in (ROOT / "tests").glob("test_*.py")
        if path.name != Path(__file__).name
    )
    for name in sorted(set(re.findall(r"test_[A-Za-z0-9_]+", text))):
        if name.startswith("test_resume_feature_matrix") or (ROOT / "tests" / f"{name}.py").exists():
            continue
        assert re.search(rf"(?:def|async def)\s+{re.escape(name)}\s*\(", test_sources), name
