def force_atty(monkeypatch):
    """
    To call on Pytest's monkeypatch fixture to force stdout to be a tty.
    """
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
