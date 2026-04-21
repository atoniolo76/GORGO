"""Shim: implementation lives in ``build_hf_prefix_trie.py``.

Run::

    modal run data_processing/build_hf_prefix_trie.py::prefix_trie --preset lmsys
"""

from build_hf_prefix_trie import app, build_hf_disk_prefix_tries, prefix_trie

__all__ = ["app", "build_hf_disk_prefix_tries", "prefix_trie"]
