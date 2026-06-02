import pytest


def test_load_watchlist_returns_empty_when_no_file(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "WATCHLIST_FILE", tmp_path / "watchlist.json")
    result = app.load_watchlist()
    assert result == []


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "WATCHLIST_FILE", tmp_path / "watchlist.json")
    app.save_watchlist(["AAPL", "MSFT"])
    result = app.load_watchlist()
    assert "AAPL" in result
    assert "MSFT" in result
    assert len(result) == 2


def test_save_watchlist_deduplicates_and_uppercases(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "WATCHLIST_FILE", tmp_path / "watchlist.json")
    app.save_watchlist(["aapl", "AAPL", "msft"])
    result = app.load_watchlist()
    assert len(result) == 2
    assert "AAPL" in result
    assert "MSFT" in result


def test_load_watchlist_handles_corrupt_json(tmp_path, monkeypatch):
    import app
    wf = tmp_path / "watchlist.json"
    wf.write_text("not valid json")
    monkeypatch.setattr(app, "WATCHLIST_FILE", wf)
    result = app.load_watchlist()
    assert result == []


def test_save_watchlist_empty_list(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "WATCHLIST_FILE", tmp_path / "watchlist.json")
    app.save_watchlist([])
    result = app.load_watchlist()
    assert result == []
