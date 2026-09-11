import base64
import json

from trainspotting.commands.common import _write_json
from trainspotting.redact import MARKER, redact_credentials


def token(account=b"123456789012345678"):
    return base64.urlsafe_b64encode(account).decode().rstrip("=") + ".abcDEF." + "x" * 27


def test_masks_embedded_credential_and_preserves_surrounding_text():
    value = token()
    assert redact_credentials(f'client.run("{value}")') == f'client.run("{MARKER}")'
    assert redact_credentials(value + " " + value) == MARKER + " " + MARKER
    assert redact_credentials(MARKER) == MARKER


def test_dotted_text_without_a_numeric_account_id_is_unchanged():
    value = token(b"not-a-discord-user!")
    assert redact_credentials(value) == value
    assert redact_credentials("ordinary.training.example") == "ordinary.training.example"


def test_result_writer_redacts_nested_text_without_changing_metadata(tmp_path):
    path = tmp_path / "sample.json"
    _write_json(path, {"records": [{"row": 3, "prompt": token(), "label": "other"}], "sample": 1000})
    assert json.loads(path.read_text()) == {
        "records": [{"row": 3, "prompt": MARKER, "label": "other"}], "sample": 1000,
    }
