"""Inject test_empty_name_not_allowed into tests/test_blueprints.py."""

import sys
from pathlib import Path

test_file = Path("repo/tests/test_blueprints.py")
if not test_file.exists():
    print(f"ERROR: {test_file} not found", file=sys.stderr)
    sys.exit(1)

content = test_file.read_text()

INJECTION = """

def test_empty_name_not_allowed(app, client):
    with pytest.raises(ValueError):
        flask.Blueprint("", __name__)

"""

MARKER = "def test_dotted_names_from_app(app, client):"

if "test_empty_name_not_allowed" in content:
    print("Test already present, skipping.")
else:
    if MARKER not in content:
        print(f"ERROR: marker not found in {test_file}", file=sys.stderr)
        sys.exit(1)
    content = content.replace(MARKER, INJECTION + MARKER)
    test_file.write_text(content)
    print("Injected test_empty_name_not_allowed.")
