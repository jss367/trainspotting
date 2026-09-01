"""A rejected token must not disable a tool that only reads public data."""

import pytest

from trainspotting import hf


class Response:
    def __init__(self, status, payload=None):
        self.status_code = status
        self.headers = {}
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"raise_for_status reached with {self.status_code}")


@pytest.fixture(autouse=True)
def a_token_and_a_clean_flag(monkeypatch):
    monkeypatch.setattr(hf, "HEADERS", {"Authorization": "Bearer stale"})
    monkeypatch.setattr(hf, "_CREDENTIALS_REJECTED", False)


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_token_is_dropped_and_the_request_retried(status, monkeypatch):
    seen = []

    def fake_get(url, params=None, timeout=None, headers=None):
        seen.append(headers)
        return Response(200, {"ok": True}) if len(seen) > 1 else Response(status)

    monkeypatch.setattr(hf.requests, "get", fake_get)

    assert hf._get("info", dataset="d") == {"ok": True}
    assert seen[0] == {"Authorization": "Bearer stale"}
    assert seen[1] == {}, "the retry still carried the credentials the server refused"


def test_the_rest_of_the_run_does_not_pay_the_rejection_again(monkeypatch):
    """The flag is module-level on purpose. Retrying per request would spend a
    refused round trip on every call for the whole run."""
    seen = []

    def fake_get(url, params=None, timeout=None, headers=None):
        seen.append(headers)
        return Response(200, {"ok": True}) if len(seen) > 1 else Response(401)

    monkeypatch.setattr(hf.requests, "get", fake_get)
    hf._get("info", dataset="d")
    hf._get("info", dataset="d")

    assert seen == [{"Authorization": "Bearer stale"}, {}, {}]


def test_a_401_without_a_token_is_still_an_error(monkeypatch):
    """Nothing to drop, so nothing to retry — that 401 is about the dataset."""
    monkeypatch.setattr(hf, "HEADERS", {})
    monkeypatch.setattr(hf.requests, "get",
                        lambda *a, **k: Response(401))

    with pytest.raises(AssertionError, match="401"):
        hf._get("info", dataset="d")
