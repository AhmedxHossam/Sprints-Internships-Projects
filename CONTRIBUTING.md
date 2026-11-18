# Contributing

Open the relevant project's README and use an isolated virtual environment. Keep credentials, generated databases, traces, and model outputs with private data out of commits. Preserve the distinction between live integrations, synthetic fixtures, and architecture proposals.

Run the affected project's documented offline checks, then rebuild downloads with `python tools/build_downloads.py` and verify the repository with `python tools/check_repository.py`. Explain the behavior changed and the checks run in your pull request. Live API checks are optional and require your own account and quota.
