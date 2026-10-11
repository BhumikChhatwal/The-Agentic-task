"""Tool 3 — general computation: ``run_python``.

Distinct from SQL because it covers everything SQL cannot express in one
statement: p-values from a custom resampling loop, pivot/reshape chains, string
cleaning, multi-step feature engineering, or a bespoke metric. The code runs in a
separate process (so the timeout can really kill it) over pickle/parquet copies of
the loaded tables, with a restricted builtin set and an import allow-list.

The contract taught to the model: assign the thing you want to see to a variable
named ``result``. A DataFrame comes back as a table; anything else as text.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..formatting import DataTable
from .base import Tool, ToolContext, as_int, optional

DEFAULT_TIMEOUT = 20
MAX_TIMEOUT = 120


class PythonExecTool(Tool):
    name = "run_python"
    category = "computation"
    description = (
        "Execute Python (pandas/numpy/scipy) against the loaded tables and return whatever "
        "you assign to a variable named `result`. Available names: `df` (the selected "
        "dataset), `tables` (dict of every dataset) and every table by its own name, plus "
        "`pd`, `np`. Use this for transformations SQL cannot express, custom metrics, "
        "reshaping, string cleaning, or multi-step computation. Print nothing you do not "
        "need; assign to `result` instead. Imports are limited to a numeric-computing allow-list."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python source. Use `df`/`tables`; assign the answer to `result`. "
                    "Example: `result = df.groupby('region')['revenue'].agg(['sum','mean'])`"
                ),
            },
            "dataset": {
                "type": "string",
                "description": (
                    "Which dataset `df` points at. Optional: omit to use the first loaded table "
                    "(or when the snippet only uses `tables`)."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": f"Seconds before the process is killed (default {DEFAULT_TIMEOUT}).",
            },
        },
        "required": ["code"],
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        code = str(kwargs.get("code") or "")
        if not code.strip():
            raise ToolError("code must not be empty")
        dataset_name = optional(kwargs, "dataset")
        if dataset_name:
            # An explicitly named dataset must resolve, or the caller has a typo.
            selected = ctx.dataset(dataset_name).name
        else:
            # Otherwise bind `df` to the first table, but do not refuse to run:
            # plenty of snippets only need `tables` (or no data at all).
            primary = ctx.registry.primary
            selected = primary.name if primary else None
        timeout = as_int(optional(kwargs, "timeout", DEFAULT_TIMEOUT), "timeout")
        timeout = max(1, min(timeout, MAX_TIMEOUT))

        with tempfile.TemporaryDirectory(prefix="byteanalyst-exec-") as tmp:
            tmp_path = Path(tmp)
            table_paths: dict[str, str] = {}
            for dataset in ctx.registry.datasets():
                target = tmp_path / f"{dataset.sql_name}.parquet"
                try:
                    dataset.frame.to_parquet(target, index=False)
                except Exception:
                    target = tmp_path / f"{dataset.sql_name}.parquet"
                    dataset.frame.to_pickle(tmp_path / f"{dataset.sql_name}.pkl")
                table_paths[dataset.name] = str(target)

            job = {
                "code": code,
                "dataset": selected,
                "tables": table_paths,
                "max_rows": ctx.config.max_preview_rows,
                "max_output": ctx.config.max_tool_output_chars,
            }
            job_path = tmp_path / "job.json"
            job_path.write_text(json.dumps(job), encoding="utf-8")

            env = dict(os.environ)
            project_root = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = os.pathsep.join(
                [project_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
            )
            env["MPLBACKEND"] = "Agg"
            env["PYTHONIOENCODING"] = "utf-8"

            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "byteanalyst.tools._exec_child", str(job_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    cwd=str(tmp_path),
                    encoding="utf-8",
                )
            except subprocess.TimeoutExpired:
                return self.fail(
                    f"code did not finish within {timeout}s and was killed. "
                    "Simplify the computation, narrow the data first with run_sql, or raise `timeout`."
                )

        raw = (proc.stdout or "").strip()
        if not raw:
            stderr = (proc.stderr or "").strip()[-1500:]
            return self.fail(f"execution produced no output (exit {proc.returncode}). {stderr}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return self.fail(
                "could not parse execution output; the child process may have crashed. "
                f"stderr: {(proc.stderr or '')[-800:]}"
            )

        if not payload.get("ok"):
            error = payload.get("error") or "unknown error"
            tb = payload.get("traceback") or ""
            return self.fail(
                f"{error}\n{tb}"
                + (f"\nstdout before failure:\n{payload.get('stdout')}" if payload.get("stdout") else "")
            )

        table = None
        result_type = payload.get("result_type")
        result = payload.get("result")
        if result_type == "dataframe" and isinstance(result, dict):
            table = DataTable(
                result.get("columns", []),
                result.get("rows", []),
                total_rows=result.get("total_rows"),
                truncated=len(result.get("rows", [])) < result.get("total_rows", 0),
            )
            summary = f"Code ran. `result` is a table with {result.get('total_rows', 0):,} row(s)."
        elif result is not None:
            text = str(result)
            summary = "Code ran. `result` = " + (text[:600] + "..." if len(text) > 600 else text)
        else:
            summary = "Code ran with no `result` assigned and no output captured."

        detail: dict[str, Any] = {"duration_ms": payload.get("duration_ms")}
        if payload.get("stdout", "").strip():
            detail["stdout"] = payload["stdout"][:2000]
        return self.result(summary, table=table, detail=detail)
