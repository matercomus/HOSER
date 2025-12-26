from __future__ import annotations

import sys
from types import ModuleType

from critics.lmtad_teacher import _import_lmtad_LMTAD_class


def test_load_lmtad_class_isolates_models_namespace(tmp_path):
    """Loads LM-TAD even if `models` is already cached.

    This simulates the real failure mode in HOSER where `models` is a namespace
    package, and LM-TAD's code expects to import `models.LMTAD` from its own
    repository.
    """

    lmtad_repo = tmp_path / "LMTAD"
    code_dir = lmtad_repo / "code"
    models_dir = code_dir / "models"
    models_dir.mkdir(parents=True)

    (models_dir / "__init__.py").write_text("\n")
    (models_dir / "LMTAD.py").write_text(
        "class LMTAD:\n    def __init__(self, cfg):\n        self.cfg = cfg\n"
    )

    # Make utils import models.LMTAD to exercise the collision path.
    (code_dir / "utils.py").write_text(
        "import models.LMTAD\n\n"
        "def log(*args, **kwargs):\n    return None\n"
    )

    original_sys_path = sys.path.copy()
    original_models = sys.modules.get("models")
    original_utils = sys.modules.get("utils")

    dummy_models = ModuleType("models")
    dummy_models.__dict__["_hoser_sentinel"] = True
    dummy_utils = ModuleType("utils")
    dummy_utils.__dict__["_hoser_sentinel"] = True

    sys.modules["models"] = dummy_models
    sys.modules["utils"] = dummy_utils

    try:
        LMTAD = _import_lmtad_LMTAD_class(str(lmtad_repo))
        assert isinstance(LMTAD, type)
        assert LMTAD.__name__ == "LMTAD"

        # Ensure we restored the caller's modules.
        assert sys.modules["models"] is dummy_models
        assert sys.modules["utils"] is dummy_utils
    finally:
        sys.path[:] = original_sys_path

        if original_models is None:
            sys.modules.pop("models", None)
        else:
            sys.modules["models"] = original_models

        if original_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = original_utils
