"""Project-level pytest configuration.

Imports should resolve through normal package installation (for example, `pip install -e .`).
"""

import os

os.environ.setdefault("BARRIKADA_SKIP_IMPORT_BUNDLE_CHECK", "1")
os.environ.setdefault("BARRIKADA_AUTO_DOWNLOAD_ARTIFACTS", "0")