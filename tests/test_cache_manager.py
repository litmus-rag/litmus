import tempfile
from pathlib import Path

from litmus.cache.manager import CacheManager, clear_cache


def test_cache_manager_disabled_when_no_dir():
    mgr = CacheManager(None)
    mgr.put("prompt", "model", "response")
    assert mgr.get("prompt", "model") is None


def test_cache_manager_put_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CacheManager(tmp)
        mgr.put("prompt", "model", "response", temperature=0.5)
        assert mgr.get("prompt", "model", temperature=0.5) == "response"
        assert mgr.get("prompt", "model", temperature=0.9) is None


def test_cache_manager_persists_across_instances():
    with tempfile.TemporaryDirectory() as tmp:
        mgr1 = CacheManager(tmp)
        mgr1.put("p", "m", "r")
        mgr2 = CacheManager(tmp)
        assert mgr2.get("p", "m") == "r"


def test_cache_manager_clear_removes_entries():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CacheManager(tmp)
        mgr.put("p", "m", "r")
        mgr.clear()
        assert mgr.get("p", "m") is None
        assert not mgr.cache_path.exists()


def test_cache_manager_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CacheManager(tmp)
        mgr.set_checkpoint("batch1", [1, 2, 3])
        assert mgr.get_checkpoint("batch1") == [1, 2, 3]
        assert mgr.get_checkpoint("missing", "default") == "default"


def test_cache_manager_clear_checkpoint_specific_key():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CacheManager(tmp)
        mgr.set_checkpoint("a", 1)
        mgr.set_checkpoint("b", 2)
        mgr.clear_checkpoint("a")
        assert mgr.get_checkpoint("a") is None
        assert mgr.get_checkpoint("b") == 2


def test_module_level_clear_cache():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CacheManager(tmp)
        mgr.put("p", "m", "r")
        clear_cache(tmp)
        mgr2 = CacheManager(tmp)
        assert mgr2.get("p", "m") is None
