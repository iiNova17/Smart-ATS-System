"""The notebook generator must never replace the user's Markdown or extra cells."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def copy_generator_workspace(tmp_path):
    for relative in [
        "tools/generate_notebook.py",
        "backend/runtime.py",
        "backend/api.py",
        "ats/validation.py",
        "requirements-kaggle.txt",
        "notebooks/smart_ats_kaggle.ipynb",
    ]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    return tmp_path / "notebooks/smart_ats_kaggle.ipynb"


def test_regeneration_preserves_edited_markdown_metadata_and_inserted_cells(tmp_path):
    path = copy_generator_workspace(tmp_path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    notebook["metadata"]["user_setting"] = "keep me"
    notebook["cells"][0]["source"] = [
        "# My own introduction\n",
        "User wording, not the template.\n",
    ]
    notebook["cells"][0]["metadata"] = {"user_annotation": True}
    custom = {
        "id": "user-custom-code",
        "cell_type": "code",
        "metadata": {},
        "source": ["user_value = 123\n"],
        "outputs": [],
        "execution_count": None,
    }
    notebook["cells"].insert(
        3,
        {
            "id": "user-custom-note",
            "cell_type": "markdown",
            "metadata": {},
            "source": ["My extra notes.\n"],
        },
    )
    notebook["cells"].append(custom)
    managed = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    managed["source"] = ["old generated code\n"]
    path.write_text(json.dumps(notebook), encoding="utf-8")
    expected = deepcopy([cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"])
    subprocess.run(
        [sys.executable, str(tmp_path / "tools/generate_notebook.py")],
        check=True,
        capture_output=True,
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert [cell for cell in result["cells"] if cell["cell_type"] == "markdown"] == expected
    assert result["metadata"] == notebook["metadata"]
    assert custom in result["cells"]
    assert (
        next(cell for cell in result["cells"] if cell["id"] == managed["id"])["source"]
        != managed["source"]
    )


def test_unrecognized_code_layout_fails_without_overwriting_the_notebook(tmp_path):
    path = copy_generator_workspace(tmp_path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")["id"] = "changed-id"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    before = path.read_bytes()
    result = subprocess.run(
        [sys.executable, str(tmp_path / "tools/generate_notebook.py")], capture_output=True
    )
    assert result.returncode != 0
    assert path.read_bytes() == before
