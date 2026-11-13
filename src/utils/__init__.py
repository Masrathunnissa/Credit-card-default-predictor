import importlib.util
from pathlib import Path

_module_path = Path(__file__).resolve().parent.parent / "utils.py"
_spec = importlib.util.spec_from_file_location("src._utils_file", _module_path)
_utils_file = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils_file)

save_model_version_info = _utils_file.save_model_version_info
