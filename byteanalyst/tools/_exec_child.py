"""Child process for the ``run_python`` tool.

Runs in a *separate process* so that a runaway loop can actually be killed by the
parent's timeout, and so a crash cannot take the agent down. This is a
convenience sandbox for a local analysis tool, not a security boundary: it
restricts builtins and imports and caps output, and it does not expose the
network or the filesystem beyond the temporary job directory.

Protocol: ``python -m byteanalyst.tools._exec_child <job.json>``
Prints a single JSON object on stdout.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

ALLOWED_MODULES = {
    "pandas", "numpy", "math", "statistics", "datetime", "json", "re", "itertools",
    "functools", "collections", "decimal", "fractions", "random", "string", "textwrap",
    "scipy", "scipy.stats", "sklearn", "sklearn.preprocessing", "sklearn.cluster",
    "sklearn.linear_model", "sklearn.metrics", "duckdb",
}

SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "getattr", "hasattr", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "pow", "print", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "type", "zip", "True", "False", "None", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "ZeroDivisionError", "NotImplementedError", "len",
    "callable", "chr", "ord", "hash", "id", "object", "super", "property", "staticmethod",
    "classmethod", "bytes", "bytearray", "complex", "bin", "hex", "oct", "input",
}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if name in ALLOWED_MODULES or root in {m.split(".")[0] for m in ALLOWED_MODULES}:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(
        f"import of {name!r} is not allowed in run_python. "
        f"Allowed roots: {', '.join(sorted({m.split('.')[0] for m in ALLOWED_MODULES}))}"
    )


def _load_tables(job: dict) -> dict:
    import pandas as pd

    tables = {}
    for name, path in job["tables"].items():
        path = Path(path)
        if path.suffix == ".parquet":
            try:
                tables[name] = pd.read_parquet(path)
                continue
            except Exception:
                pass
        tables[name] = pd.read_pickle(path.with_suffix(".pkl"))
    return tables


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    started = time.perf_counter()
    payload: dict = {"ok": False, "stdout": "", "result": None, "result_type": None, "error": None}

    stdout_buffer = io.StringIO()
    try:
        import builtins as _builtins

        import numpy as np
        import pandas as pd

        tables = _load_tables(job)
        primary = job.get("dataset") or (next(iter(tables)) if tables else None)
        safe_builtins = {k: getattr(_builtins, k) for k in SAFE_BUILTINS if hasattr(_builtins, k)}
        safe_builtins["__import__"] = _restricted_import
        namespace: dict = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
            "tables": tables,
        }
        if primary:
            namespace["df"] = tables[primary]
            namespace["data"] = namespace["df"]
            namespace["table"] = namespace["df"]
        for name, frame in tables.items():
            namespace[name.replace("-", "_")] = frame

        with contextlib.redirect_stdout(stdout_buffer):
            exec(compile(job["code"], "<run_python>", "exec"), namespace)

        result = namespace.get("result", None)
        printed = stdout_buffer.getvalue()
        if isinstance(result, pd.DataFrame):
            # Keep meaningful labels: a groupby result's index is the answer, so it
            # is promoted to a real column rather than silently dropped.
            framed = result
            if not (isinstance(framed.index, pd.RangeIndex) and framed.index.name is None):
                framed = framed.reset_index()
            preview = framed.head(job.get("max_rows", 50))
            payload["result_type"] = "dataframe"
            payload["result"] = {
                "columns": [str(c) for c in preview.columns],
                "rows": json.loads(preview.to_json(orient="values", date_format="iso", default_handler=str)),
                "total_rows": int(len(result)),
            }
        elif isinstance(result, pd.Series):
            preview = result.head(job.get("max_rows", 50))
            index_label = str(result.index.name) if result.index.name else "index"
            value_label = str(result.name) if result.name is not None else "value"
            payload["result_type"] = "dataframe"
            payload["result"] = {
                "columns": [index_label, value_label],
                "rows": [[str(k), _clean(v)] for k, v in preview.items()],
                "total_rows": int(len(result)),
            }
        elif result is not None:
            payload["result_type"] = "text"
            payload["result"] = repr(result)[:4000]
        elif printed.strip():
            payload["result_type"] = "text"
            payload["result"] = printed[:4000]

        payload["stdout"] = printed[: job.get("max_output", 8000)]
        payload["ok"] = True
    except Exception as exc:
        payload["ok"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc(limit=5)
        payload["stdout"] = stdout_buffer.getvalue()[: job.get("max_output", 8000)]
    payload["duration_ms"] = int((time.perf_counter() - started) * 1000)

    sys.stdout.write(json.dumps(payload, default=str))
    return 0


def _clean(value):
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
