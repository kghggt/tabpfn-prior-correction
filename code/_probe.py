"""一次性探针：确认 tabpfn / sklearn / openml 的可用 API 形态。"""
import inspect
import sys

print("python:", sys.version.split()[0])

import tabpfn

print("tabpfn version:", getattr(tabpfn, "__version__", "unknown"))
print("tabpfn exports:", [x for x in dir(tabpfn) if not x.startswith("_")])

try:
    from tabpfn import TabPFNClassifier

    print("TabPFNClassifier.__init__:")
    print(inspect.signature(TabPFNClassifier.__init__))
except Exception as exc:  # noqa: BLE001
    print("TabPFNClassifier import failed:", repr(exc))

try:
    from tabpfn.constants import ModelVersion

    print("ModelVersion:", list(ModelVersion))
except Exception as exc:  # noqa: BLE001
    print("tabpfn.constants unavailable:", repr(exc))

try:
    import sklearn

    print("sklearn:", sklearn.__version__)
except Exception as exc:  # noqa: BLE001
    print("sklearn failed:", repr(exc))

try:
    import openml

    print("openml:", openml.__version__)
except Exception as exc:  # noqa: BLE001
    print("openml failed:", repr(exc))

for mod in ("xgboost", "lightgbm", "catboost", "imblearn", "matplotlib"):
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception:  # noqa: BLE001
        print(mod, "MISSING")
