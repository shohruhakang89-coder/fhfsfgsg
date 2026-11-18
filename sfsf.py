#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ULTIMATE CLOUD NOTEBOOK v4.0 — PRO AI TRAINING & LIVE STUDIO EDITION
================================================================================
- 100% Complete Jupyter IOPub Protocol: stream, display_data, update_display_data
  (with transient display_id for live HTML dashboards like OQIM-KOKORO StudioUI),
  execute_result, error, clear_output(wait=True).
- Zero-Freeze Stream & DOM Engine: Monotonic _rev versioning prevents frozen
  logs or skipped HTML/Plot/Audio updates.
- High-Speed In-Memory Output Cache + Throttled SQLite Persistence: Handles
  heavy base64 WAV + Spectrogram updates without choking WebSocket or DB.
- Full Audio Studio & Checkpoint Explorer: Scans /kaggle/working, workspace,
  and /temp (including /temp/oqim_kokoro_v7_work).
- Persistent Background Execution + Auto Cloudflare HTTPS Tunnel.
================================================================================
"""

import os
import sys
import time
import json
import uuid
import glob
import math
import queue
import shutil
import signal
import base64
import secrets
import sqlite3
import logging
import hashlib
import tarfile
import asyncio
import platform
import threading
import traceback
import subprocess
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple


# ==============================================================================
# 1. BOOTSTRAP: AUTO-INSTALL DEPENDENCIES & CLOUDFLARED
# ==============================================================================
def bootstrap_environment():
    required_pkgs = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn[standard]"),
        ("websockets", "websockets"),
        ("jupyter_client", "jupyter_client"),
        ("ipykernel", "ipykernel"),
        ("psutil", "psutil"),
        ("nbformat", "nbformat"),
        ("pynvml", "nvidia-ml-py"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("pydantic", "pydantic"),
        ("multipart", "python-multipart"),
    ]
    missing = []
    for mod_name, pip_name in required_pkgs:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[Bootstrap] Installing packages: {', '.join(missing)} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + missing
        )


bootstrap_environment()

import psutil
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell, new_output
import jupyter_client
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Request,
    HTTPException,
    Depends,
    UploadFile,
    File,
)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from pydantic import BaseModel


# ==============================================================================
# 2. ENVIRONMENT DETECTION & CONFIGURATION
# ==============================================================================
IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
BASE_ROOT = Path("/kaggle/working") if IS_KAGGLE else Path(os.getcwd()).resolve()
APP_DIR = BASE_ROOT / ".ultimate_cloud_notebook"
WORKSPACE_DIR = BASE_ROOT / "workspace"
CHECKPOINTS_DIR = WORKSPACE_DIR / "checkpoints"
AUDIO_OUT_DIR = WORKSPACE_DIR / "audio_outputs"
LOGS_DIR = APP_DIR / "logs"
DB_PATH = APP_DIR / "notebook_state.db"
TOKEN_FILE = APP_DIR / "auth_token.txt"
TUNNEL_URL_FILE = APP_DIR / "tunnel_url.txt"
AUTOSAVE_IPYNB = WORKSPACE_DIR / "autosave_notebook.ipynb"
TRAINING_CONFIG_FILE = WORKSPACE_DIR / "training_config.json"

for d in [APP_DIR, WORKSPACE_DIR, CHECKPOINTS_DIR, AUDIO_OUT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SERVER_HOST = os.environ.get("UCN_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("UCN_PORT", "8765"))
BOOT_TIME = time.time()


def get_or_create_token() -> str:
    env_tok = os.environ.get("UCN_TOKEN", "").strip()
    if env_tok:
        TOKEN_FILE.write_text(env_tok, encoding="utf-8")
        return env_tok
    if TOKEN_FILE.exists():
        existing = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if len(existing) >= 16:
            return existing
    new_tok = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(new_tok, encoding="utf-8")
    return new_tok


AUTH_TOKEN = get_or_create_token()


# ==============================================================================
# 3. LOGGING WITH SECRET SCRUBBING
# ==============================================================================
class TokenScrubbingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if AUTH_TOKEN and AUTH_TOKEN in msg:
            msg = msg.replace(AUTH_TOKEN, "[REDACTED_TOKEN]")
        return msg


logger = logging.getLogger("UltimateCloudNotebook")
logger.setLevel(logging.INFO)
logger.handlers.clear()

_fh = logging.FileHandler(LOGS_DIR / "system.log", encoding="utf-8")
_fh.setFormatter(TokenScrubbingFormatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)

_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(TokenScrubbingFormatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_sh)


# ==============================================================================
# 4. HELPER PYTHON MODULE (`ucloud`) FOR CELLS & AI TRAINING
# ==============================================================================
UCLOUD_HELPER_CODE = '''# -*- coding: utf-8 -*-
"""
ucloud — Built-in Helper Library for Ultimate Cloud Notebook
Provides audio formatting/playback, checkpoint management, and config persistence.
"""
import io
import os
import json
import glob
import wave
import struct
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
from IPython.display import Audio, display, HTML

WORKSPACE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = WORKSPACE_DIR / "checkpoints"
AUDIO_OUT_DIR = WORKSPACE_DIR / "audio_outputs"
TRAINING_CONFIG_FILE = WORKSPACE_DIR / "training_config.json"

for _d in (CHECKPOINTS_DIR, AUDIO_OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _to_numpy_audio(data: Any) -> np.ndarray:
    if hasattr(data, "detach") and hasattr(data, "cpu") and hasattr(data, "numpy"):
        arr = data.detach().cpu().numpy()
    elif hasattr(data, "numpy"):
        arr = data.numpy()
    else:
        arr = np.asarray(data)
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim == 2:
        if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 8:
            arr = arr.T
    elif arr.ndim > 2:
        arr = arr.reshape(1, -1)
    max_val = np.max(np.abs(arr)) if arr.size > 0 else 0.0
    if max_val > 1.0:
        arr = arr / max_val
    return np.clip(arr, -1.0, 1.0)


def save_wav(
    data: Any,
    sample_rate: int = 24000,
    filename: Optional[str] = None
) -> str:
    arr = _to_numpy_audio(data)
    channels, frames = arr.shape
    pcm16 = (arr.T * 32767.0).astype(np.int16)
    if not filename:
        import time
        filename = f"audio_{int(time.time() * 1000)}.wav"
    out_path = AUDIO_OUT_DIR / filename if not os.path.isabs(filename) else Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return str(out_path)


def play_audio(
    data: Any,
    sample_rate: int = 24000,
    label: str = "Generated Audio",
    save_copy: bool = True,
    filename: Optional[str] = None
):
    if isinstance(data, (str, Path)) and os.path.exists(str(data)):
        wav_path = str(data)
        with open(wav_path, "rb") as f:
            raw_bytes = f.read()
        ext = Path(wav_path).suffix.lower().lstrip(".") or "wav"
        mime = f"audio/{'mpeg' if ext == 'mp3' else ext}"
    else:
        arr = _to_numpy_audio(data)
        channels, _ = arr.shape
        pcm16 = (arr.T * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm16.tobytes())
        raw_bytes = buf.getvalue()
        mime = "audio/wav"
        if save_copy:
            save_wav(arr, sample_rate=sample_rate, filename=filename)

    import base64
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    html = f"""
    <div style="margin:6px 0;padding:10px 14px;background:#111827;border:1px solid #1f2937;border-radius:8px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
        <span style="font-size:12px;font-weight:700;color:#38bdf8;font-family:monospace;">🔊 {label} ({sample_rate} Hz)</span>
        <audio controls preload="metadata" style="height:34px;flex:1;min-width:240px;" src="data:{mime};base64,{b64}"></audio>
    </div>
    """
    display(HTML(html))


def save_training_config(config: Dict[str, Any]) -> str:
    TRAINING_CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(TRAINING_CONFIG_FILE)


def load_training_config(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if TRAINING_CONFIG_FILE.exists():
        try:
            return json.loads(TRAINING_CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default or {}


def save_checkpoint(state: Any, name: str = "checkpoint", step: Optional[int] = None, keep_last: int = 5) -> str:
    step_sfx = f"_step_{step:07d}" if step is not None else ""
    target = CHECKPOINTS_DIR / f"{name}{step_sfx}.pt"
    try:
        import torch
        torch.save(state, str(target))
    except Exception:
        import pickle
        with open(target, "wb") as f:
            pickle.dump(state, f)
    if keep_last and keep_last > 0:
        pattern = str(CHECKPOINTS_DIR / f"{name}_step_*.pt")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        while len(files) > keep_last:
            old = files.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass
    return str(target)


def load_latest_checkpoint(name: str = "checkpoint", map_location: str = "cpu") -> Tuple[Optional[Any], Optional[str]]:
    pattern = str(CHECKPOINTS_DIR / f"{name}*.pt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        return None, None
    latest = files[-1]
    try:
        import torch
        return torch.load(latest, map_location=map_location), latest
    except Exception:
        import pickle
        with open(latest, "rb") as f:
            return pickle.load(f), latest
'''
(WORKSPACE_DIR / "ucloud.py").write_text(UCLOUD_HELPER_CODE, encoding="utf-8")


# ==============================================================================
# 5. SQLITE PERSISTENT STATE ENGINE
# ==============================================================================
class PersistentStateDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cells (
                    id TEXT PRIMARY KEY,
                    cell_type TEXT NOT NULL DEFAULT 'code',
                    source TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'idle',
                    execution_count INTEGER,
                    outputs_json TEXT NOT NULL DEFAULT '[]',
                    started_at REAL,
                    finished_at REAL,
                    position INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.commit()
            cur = conn.execute("SELECT COUNT(*) AS cnt FROM cells")
            if cur.fetchone()["cnt"] == 0:
                self._seed_default_cells(conn)

    def _seed_default_cells(self, conn: sqlite3.Connection):
        now = time.time()
        welcome_md = (
            "### ⚡ Ultimate Cloud Notebook v4.0 — Pro AI & Audio Studio\n"
            "- **Persistent Execution**: Brauzerni yopsangiz ham kod va AI training orqa fonda to'xtovsiz ishlaydi.\n"
            "- **Full Display Support**: `OQIM-KOKORO StudioUI` (`display_id` / `update_display`), `tqdm`, `Matplotlib`, `Audio` va `Image` to'liq real-time ko'rsatiladi.\n"
            "- **Shortcuts**: `Shift + Enter` (Run & Next), `Ctrl + Enter` (Run in-place)."
        )
        sample_code = (
            "import torch, platform, ucloud\n"
            "import numpy as np\n\n"
            "print(f'Python: {platform.python_version()} | PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')\n"
            "if torch.cuda.is_available():\n"
            "    for i in range(torch.cuda.device_count()):\n"
            "        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')\n"
        )
        cells = [
            (uuid.uuid4().hex[:12], "markdown", welcome_md, "idle", None, "[]", None, None, 0, now),
            (uuid.uuid4().hex[:12], "code", sample_code, "idle", None, "[]", None, None, 1, now),
        ]
        conn.executemany(
            """
            INSERT INTO cells (id, cell_type, source, status, execution_count, outputs_json, started_at, finished_at, position, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            cells,
        )
        conn.commit()

    def _row_to_dict(self, r: sqlite3.Row) -> Dict[str, Any]:
        try:
            outputs = json.loads(r["outputs_json"] or "[]")
        except Exception:
            outputs = []
        return {
            "id": r["id"],
            "cell_type": r["cell_type"],
            "source": r["source"],
            "status": r["status"],
            "execution_count": r["execution_count"],
            "outputs": outputs,
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
            "position": r["position"],
            "updated_at": r["updated_at"],
        }

    def list_cells(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM cells ORDER BY position ASC, updated_at ASC").fetchall()
            return [self._row_to_dict(r) for r in rows]

    def get_cell(self, cell_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            r = conn.execute("SELECT * FROM cells WHERE id = ?", (cell_id,)).fetchone()
            return self._row_to_dict(r) if r else None

    def create_cell(
        self,
        cell_type: str = "code",
        source: str = "",
        after_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id, position FROM cells ORDER BY position ASC").fetchall()
            insert_pos = len(rows)
            if after_id:
                for idx, r in enumerate(rows):
                    if r["id"] == after_id:
                        insert_pos = idx + 1
                        break
            for idx in range(insert_pos, len(rows)):
                conn.execute("UPDATE cells SET position = ? WHERE id = ?", (idx + 1, rows[idx]["id"]))
            cid = uuid.uuid4().hex[:12]
            now = time.time()
            conn.execute(
                """
                INSERT INTO cells (id, cell_type, source, status, execution_count, outputs_json, started_at, finished_at, position, updated_at)
                VALUES (?, ?, ?, 'idle', NULL, '[]', NULL, NULL, ?, ?)
                """,
                (cid, cell_type, source, insert_pos, now),
            )
            conn.commit()
        self.export_to_ipynb(AUTOSAVE_IPYNB)
        return self.get_cell(cid)

    def update_cell(self, cell_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        allowed = {
            "cell_type",
            "source",
            "status",
            "execution_count",
            "outputs",
            "started_at",
            "finished_at",
            "position",
        }
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k == "outputs":
                sets.append("outputs_json = ?")
                vals.append(json.dumps(v, ensure_ascii=False))
            else:
                sets.append(f"{k} = ?")
                vals.append(v)
        if not sets:
            return self.get_cell(cell_id)
        sets.append("updated_at = ?")
        vals.append(time.time())
        vals.append(cell_id)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE cells SET {', '.join(sets)} WHERE id = ?", vals)
            conn.commit()
        return self.get_cell(cell_id)

    def delete_cell(self, cell_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM cells WHERE id = ?", (cell_id,))
            conn.commit()
            if cur.rowcount == 0:
                return False
            rows = conn.execute("SELECT id FROM cells ORDER BY position ASC").fetchall()
            for idx, r in enumerate(rows):
                conn.execute("UPDATE cells SET position = ? WHERE id = ?", (idx, r["id"]))
            if len(rows) == 0:
                self._seed_default_cells(conn)
            conn.commit()
        self.export_to_ipynb(AUTOSAVE_IPYNB)
        return True

    def move_cell(self, cell_id: str, direction: str) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id, position FROM cells ORDER BY position ASC").fetchall()
            ids = [r["id"] for r in rows]
            if cell_id not in ids:
                return self.list_cells()
            idx = ids.index(cell_id)
            swap_idx = idx - 1 if direction == "up" else idx + 1
            if 0 <= swap_idx < len(ids):
                ids[idx], ids[swap_idx] = ids[swap_idx], ids[idx]
                for pos, cid in enumerate(ids):
                    conn.execute("UPDATE cells SET position = ? WHERE id = ?", (pos, cid))
                conn.commit()
        self.export_to_ipynb(AUTOSAVE_IPYNB)
        return self.list_cells()

    def clear_all_outputs(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE cells SET outputs_json = '[]', updated_at = ?", (time.time(),))
            conn.commit()
        self.export_to_ipynb(AUTOSAVE_IPYNB)
        return self.list_cells()

    def export_to_ipynb(self, target_path: Path) -> Path:
        try:
            cells = self.list_cells()
            nb = new_notebook()
            nb_cells = []
            for c in cells:
                if c["cell_type"] == "markdown":
                    nb_cells.append(new_markdown_cell(source=c["source"]))
                else:
                    nc = new_code_cell(
                        source=c["source"],
                        execution_count=c["execution_count"],
                    )
                    nb_outs = []
                    for out in c.get("outputs", []):
                        otype = out.get("output_type")
                        try:
                            if otype == "stream":
                                nb_outs.append(
                                    new_output("stream", name=out.get("name", "stdout"), text=out.get("text", ""))
                                )
                            elif otype in ("display_data", "execute_result"):
                                kw = {"data": out.get("data", {}), "metadata": out.get("metadata", {})}
                                if otype == "execute_result":
                                    kw["execution_count"] = out.get("execution_count")
                                nb_outs.append(new_output(otype, **kw))
                            elif otype == "error":
                                nb_outs.append(
                                    new_output(
                                        "error",
                                        ename=out.get("ename", "Error"),
                                        evalue=out.get("evalue", ""),
                                        traceback=out.get("traceback", []),
                                    )
                                )
                        except Exception:
                            pass
                    nc.outputs = nb_outs
                    nb_cells.append(nc)
            nb.cells = nb_cells
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
        except Exception as e:
            logger.warning(f"Autosave .ipynb warning: {e}")
        return target_path

    def import_from_ipynb(self, content_str: str) -> List[Dict[str, Any]]:
        nb = nbformat.reads(content_str, as_version=4)
        now = time.time()
        new_rows = []
        for idx, c in enumerate(nb.cells):
            cid = uuid.uuid4().hex[:12]
            ctype = "markdown" if c.cell_type == "markdown" else "code"
            src = c.source if isinstance(c.source, str) else "".join(c.source)
            exec_cnt = c.get("execution_count") if ctype == "code" else None
            outs = []
            if ctype == "code":
                for r_idx, o in enumerate(c.get("outputs", []), start=1):
                    otype = o.get("output_type")
                    if otype == "stream":
                        txt = o.get("text", "")
                        if isinstance(txt, list):
                            txt = "".join(txt)
                        outs.append({"output_type": "stream", "name": o.get("name", "stdout"), "text": txt, "_rev": r_idx})
                    elif otype in ("display_data", "execute_result"):
                        outs.append(
                            {
                                "output_type": otype,
                                "data": dict(o.get("data", {})),
                                "metadata": dict(o.get("metadata", {})),
                                "execution_count": o.get("execution_count"),
                                "_rev": r_idx,
                            }
                        )
                    elif otype == "error":
                        outs.append(
                            {
                                "output_type": "error",
                                "ename": o.get("ename", "Error"),
                                "evalue": o.get("evalue", ""),
                                "traceback": list(o.get("traceback", [])),
                                "_rev": r_idx,
                            }
                        )
            new_rows.append((cid, ctype, src, "idle", exec_cnt, json.dumps(outs, ensure_ascii=False), None, None, idx, now))

        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM cells")
            if new_rows:
                conn.executemany(
                    """
                    INSERT INTO cells (id, cell_type, source, status, execution_count, outputs_json, started_at, finished_at, position, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    new_rows,
                )
            else:
                self._seed_default_cells(conn)
            conn.commit()
        self.export_to_ipynb(AUTOSAVE_IPYNB)
        return self.list_cells()


state_db = PersistentStateDB(DB_PATH)


# ==============================================================================
# 6. HARDWARE TELEMETRY MONITOR (DUAL GPU T4x2, CPU, RAM, DISK)
# ==============================================================================
class HardwareMonitor:
    def __init__(self):
        self.nvml_initialized = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_initialized = True
        except Exception:
            self.nvml_initialized = False

    def get_gpu_stats(self) -> List[Dict[str, Any]]:
        gpus = []
        if self.nvml_initialized:
            try:
                import pynvml
                count = pynvml.nvmlDeviceGetCount()
                for idx in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="replace")
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    except Exception:
                        util = 0
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        temp = 0
                    used_mb = round(mem.used / (1024**2), 1)
                    total_mb = round(mem.total / (1024**2), 1)
                    pct = round((mem.used / mem.total) * 100.0, 1) if mem.total else 0.0
                    gpus.append(
                        {
                            "index": idx,
                            "name": name,
                            "vram_used_mb": used_mb,
                            "vram_total_mb": total_mb,
                            "vram_percent": pct,
                            "util_percent": float(util),
                            "temp_c": int(temp),
                        }
                    )
                return gpus
            except Exception:
                pass

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=2.0,
                stderr=subprocess.DEVNULL,
            ).decode("utf-8", errors="replace")
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    used = float(parts[2])
                    total = float(parts[3])
                    pct = round((used / total) * 100.0, 1) if total > 0 else 0.0
                    gpus.append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "vram_used_mb": used,
                            "vram_total_mb": total,
                            "vram_percent": pct,
                            "util_percent": float(parts[4]),
                            "temp_c": int(float(parts[5])),
                        }
                    )
        except Exception:
            pass
        return gpus

    def collect(self, kernel_state: str) -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        du = psutil.disk_usage(str(BASE_ROOT))
        return {
            "mode": "Kaggle Mode" if IS_KAGGLE else "Standalone VPS Mode",
            "kernel_state": kernel_state,
            "uptime_sec": int(time.time() - BOOT_TIME),
            "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
            "cpu_Count": psutil.cpu_count(logical=True),
            "ram_used_gb": round(vm.used / (1024**3), 2),
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_percent": round(vm.percent, 1),
            "disk_used_gb": round(du.used / (1024**3), 2),
            "disk_free_gb": round(du.free / (1024**3), 2),
            "disk_total_gb": round(du.total / (1024**3), 2),
            "disk_percent": round(du.percent, 1),
            "gpus": self.get_gpu_stats(),
            "timestamp": time.time(),
        }


hw_monitor = HardwareMonitor()


# ==============================================================================
# 7. TERMINAL CARRIAGE-RETURN (\r) & TQDM STREAM NORMALIZER
# ==============================================================================
def normalize_terminal_stream(existing_text: str, new_chunk: str, max_chars: int = 300000) -> str:
    combined = (existing_text or "") + (new_chunk or "")
    combined = combined.replace("\r\n", "\n")
    lines = combined.split("\n")
    processed_lines = []
    for line in lines:
        if "\r" in line:
            segments = line.split("\r")
            current = ""
            for seg in segments:
                if not seg:
                    continue
                if len(seg) >= len(current):
                    current = seg
                else:
                    current = seg + current[len(seg) :]
            processed_lines.append(current)
        else:
            processed_lines.append(line)
    result = "\n".join(processed_lines)
    if len(result) > max_chars:
        result = "... [Stream truncated for browser performance] ...\n" + result[-max_chars:]
    return result


# ==============================================================================
# 8. REAL JUPYTER KERNEL ORCHESTRATOR (FULL IOPUB + DISPLAY_ID ENGINE)
# ==============================================================================
class KernelOrchestrator:
    def __init__(self, db: PersistentStateDB):
        self.db = db
        self.km: Optional[jupyter_client.KernelManager] = None
        self.kc = None
        self.kernel_state = "starting"
        self.exec_queue: List[str] = []
        self.active_cell_id: Optional[str] = None
        self.msg_to_cell: Dict[str, str] = {}
        self.execution_counter = 1
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self.ws_broadcast_callback = None

        # High-speed in-memory output cache + monotonic revision counter
        self._rev_counter = 1
        self.cell_outputs_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.pending_clear: Dict[str, bool] = {}
        self.display_id_index: Dict[str, List[Tuple[str, int]]] = {}
        self._dirty_cells: set = set()
        self._last_db_flush: Dict[str, float] = {}
        self._last_ws_push: Dict[str, float] = {}

        self._start_kernel_process()
        self._iopub_thread = threading.Thread(target=self._iopub_listener_loop, daemon=True)
        self._iopub_thread.start()
        self._queue_thread = threading.Thread(target=self._execution_worker_loop, daemon=True)
        self._queue_thread.start()

    def _next_rev(self) -> int:
        with self._lock:
            self._rev_counter += 1
            return self._rev_counter

    def _start_kernel_process(self):
        with self._lock:
            self.kernel_state = "starting"
            try:
                if self.kc:
                    try:
                        self.kc.stop_channels()
                    except Exception:
                        pass
                if self.km and self.km.is_alive():
                    try:
                        self.km.shutdown_kernel(now=True)
                    except Exception:
                        pass

                env = os.environ.copy()
                py_path = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{WORKSPACE_DIR}:{BASE_ROOT}:{py_path}" if py_path else f"{WORKSPACE_DIR}:{BASE_ROOT}"
                env["PYTHONUNBUFFERED"] = "1"

                self.km = jupyter_client.KernelManager(kernel_name="python3")
                self.km.start_kernel(cwd=str(WORKSPACE_DIR), env=env)
                self.kc = self.km.client()
                self.kc.start_channels()
                self.kc.wait_for_ready(timeout=30)

                init_code = (
                    "import os, sys\n"
                    f"if {repr(str(WORKSPACE_DIR))} not in sys.path: sys.path.insert(0, {repr(str(WORKSPACE_DIR))})\n"
                    f"if {repr(str(BASE_ROOT))} not in sys.path: sys.path.insert(0, {repr(str(BASE_ROOT))})\n"
                    "try:\n"
                    "    import matplotlib\n"
                    "    matplotlib.use('module://matplotlib_inline.backend_inline')\n"
                    "except Exception:\n"
                    "    pass\n"
                    "import ucloud\n"
                )
                self.kc.execute(init_code, silent=True, store_history=False)
                self.kernel_state = "idle"
                logger.info("IPython Kernel started and initialized successfully.")
            except Exception as e:
                self.kernel_state = "error"
                logger.error(f"Failed to start IPython Kernel: {e}\n{traceback.format_exc()}")

    def _emit_ws(self, payload: Dict[str, Any]):
        if self.ws_broadcast_callback:
            try:
                self.ws_broadcast_callback(payload)
            except Exception:
                pass

    def _sync_cell_state(self, cell_id: str, force: bool = False):
        now = time.time()
        with self._lock:
            outputs = self.cell_outputs_cache.get(cell_id)
            if outputs is None:
                c = self.db.get_cell(cell_id)
                if c:
                    self._emit_ws({"type": "cell_updated", "cell": c})
                return
            outputs_snapshot = [dict(o) for o in outputs]

        last_ws = self._last_ws_push.get(cell_id, 0.0)
        last_db = self._last_db_flush.get(cell_id, 0.0)

        if force or (now - last_db >= 1.2):
            self._last_db_flush[cell_id] = now
            self._dirty_cells.discard(cell_id)
            cell = self.db.update_cell(cell_id, outputs=outputs_snapshot)
            if cell:
                self._last_ws_push[cell_id] = now
                self._emit_ws({"type": "cell_updated", "cell": cell})
        elif now - last_ws >= 0.20:
            self._last_ws_push[cell_id] = now
            self._dirty_cells.add(cell_id)
            cell = self.db.get_cell(cell_id)
            if cell:
                cell["outputs"] = outputs_snapshot
                self._emit_ws({"type": "cell_updated", "cell": cell})
        else:
            self._dirty_cells.add(cell_id)

    def _flush_dirty_cells(self):
        for cid in list(self._dirty_cells):
            self._sync_cell_state(cid, force=True)

    def restart_kernel(self):
        logger.info("Restarting IPython kernel...")
        with self._lock:
            queued_ids = list(self.exec_queue)
            self.exec_queue.clear()
            active_id = self.active_cell_id
            self.active_cell_id = None
            self.msg_to_cell.clear()
            self.cell_outputs_cache.clear()
            self.pending_clear.clear()
            self.display_id_index.clear()
            self._dirty_cells.clear()

        if active_id:
            c = self.db.update_cell(active_id, status="error", finished_at=time.time())
            if c:
                self._emit_ws({"type": "cell_updated", "cell": c})
        for qid in queued_ids:
            c = self.db.update_cell(qid, status="idle")
            if c:
                self._emit_ws({"type": "cell_updated", "cell": c})

        self._start_kernel_process()
        self._emit_ws({"type": "kernel_state", "state": self.kernel_state})

    def interrupt_kernel(self):
        logger.info("Interrupting IPython kernel (SIGINT)...")
        with self._lock:
            queued_ids = list(self.exec_queue)
            self.exec_queue.clear()
        for qid in queued_ids:
            c = self.db.update_cell(qid, status="idle")
            if c:
                self._emit_ws({"type": "cell_updated", "cell": c})
        if self.km and self.km.is_alive():
            try:
                self.km.interrupt_kernel()
            except Exception as e:
                logger.warning(f"Interrupt error: {e}")

    def enqueue_cell(self, cell_id: str) -> Optional[Dict[str, Any]]:
        cell = self.db.get_cell(cell_id)
        if not cell:
            return None
        if cell["cell_type"] == "markdown":
            return cell
        with self._lock:
            if cell_id not in self.exec_queue and cell_id != self.active_cell_id:
                self.exec_queue.append(cell_id)
                cell = self.db.update_cell(cell_id, status="queued")
                self._emit_ws({"type": "cell_updated", "cell": cell})
        return cell

    def enqueue_all(self) -> List[Dict[str, Any]]:
        cells = self.db.list_cells()
        for c in cells:
            if c["cell_type"] == "code" and c["source"].strip():
                self.enqueue_cell(c["id"])
        return self.db.list_cells()

    def _execution_worker_loop(self):
        while not self._stop_event.is_set():
            if time.time() - getattr(self, "_last_periodic_flush", 0.0) >= 0.4:
                self._last_periodic_flush = time.time()
                if self._dirty_cells:
                    for cid in list(self._dirty_cells):
                        if time.time() - self._last_ws_push.get(cid, 0.0) >= 0.25:
                            self._sync_cell_state(cid, force=False)

            next_cid = None
            with self._lock:
                if self.active_cell_id is None and self.exec_queue:
                    next_cid = self.exec_queue.pop(0)
                    self.active_cell_id = next_cid

            if not next_cid:
                time.sleep(0.05)
                continue

            cell = self.db.get_cell(next_cid)
            if not cell:
                with self._lock:
                    self.active_cell_id = None
                continue

            if not self.km or not self.km.is_alive():
                logger.warning("Kernel died unexpectedly; auto-restarting...")
                self._start_kernel_process()

            exec_cnt = self.execution_counter
            self.execution_counter += 1
            now = time.time()

            with self._lock:
                self.cell_outputs_cache[next_cid] = []
                self.pending_clear[next_cid] = False
                for did in list(self.display_id_index.keys()):
                    self.display_id_index[did] = [
                        (c_id, idx) for (c_id, idx) in self.display_id_index[did] if c_id != next_cid
                    ]

            updated = self.db.update_cell(
                next_cid,
                status="running",
                execution_count=exec_cnt,
                outputs=[],
                started_at=now,
                finished_at=None,
            )
            self.kernel_state = "busy"
            self._emit_ws({"type": "kernel_state", "state": "busy"})
            self._emit_ws({"type": "cell_updated", "cell": updated})

            try:
                msg_id = self.kc.execute(
                    cell["source"],
                    silent=False,
                    store_history=True,
                    allow_stdin=False,
                    stop_on_error=True,
                )
                with self._lock:
                    self.msg_to_cell[msg_id] = next_cid

                while self.active_cell_id == next_cid and not self._stop_event.is_set():
                    if not self.km.is_alive():
                        self._append_output(
                            next_cid,
                            {
                                "output_type": "error",
                                "ename": "KernelDiedError",
                                "evalue": "Kernel process terminated unexpectedly (check OOM / memory).",
                                "traceback": ["Kernel process terminated unexpectedly."],
                                "_rev": self._next_rev(),
                            },
                        )
                        self._sync_cell_state(next_cid, force=True)
                        c = self.db.update_cell(next_cid, status="error", finished_at=time.time())
                        self._emit_ws({"type": "cell_updated", "cell": c})
                        with self._lock:
                            self.active_cell_id = None
                        self._start_kernel_process()
                        break

                    if self._dirty_cells:
                        for d_cid in list(self._dirty_cells):
                            if time.time() - self._last_ws_push.get(d_cid, 0.0) >= 0.25:
                                self._sync_cell_state(d_cid, force=False)

                    # Drain shell channel replies so ZMQ sockets never fill up
                    try:
                        self.kc.get_shell_msg(timeout=0.05)
                    except queue.Empty:
                        pass
                    except Exception:
                        time.sleep(0.04)

                self._flush_dirty_cells()
                self.db.export_to_ipynb(AUTOSAVE_IPYNB)

            except Exception as e:
                logger.error(f"Cell execution error: {e}")
                self._append_output(
                    next_cid,
                    {
                        "output_type": "error",
                        "ename": type(e).__name__,
                        "evalue": str(e),
                        "traceback": traceback.format_exc().splitlines(),
                        "_rev": self._next_rev(),
                    },
                )
                self._sync_cell_state(next_cid, force=True)
                c = self.db.update_cell(next_cid, status="error", finished_at=time.time())
                self._emit_ws({"type": "cell_updated", "cell": c})
                with self._lock:
                    self.active_cell_id = None

    def _clear_cell_outputs_if_pending(self, cell_id: str):
        if self.pending_clear.get(cell_id):
            self.pending_clear[cell_id] = False
            self.cell_outputs_cache[cell_id] = []
            for did in list(self.display_id_index.keys()):
                self.display_id_index[did] = [
                    (c_id, idx) for (c_id, idx) in self.display_id_index[did] if c_id != cell_id
                ]

    def _append_output(self, cell_id: str, new_out: Dict[str, Any], display_id: Optional[str] = None):
        with self._lock:
            if cell_id not in self.cell_outputs_cache:
                cell = self.db.get_cell(cell_id)
                self.cell_outputs_cache[cell_id] = list(cell.get("outputs", [])) if cell else []

            self._clear_cell_outputs_if_pending(cell_id)
            outputs = self.cell_outputs_cache[cell_id]
            new_out["_rev"] = self._next_rev()

            if display_id:
                new_out["display_id"] = display_id
                # If this cell already has an output with the same display_id, update in place
                for idx, existing in enumerate(outputs):
                    if existing.get("display_id") == display_id:
                        outputs[idx] = new_out
                        return
                outputs.append(new_out)
                out_idx = len(outputs) - 1
                self.display_id_index.setdefault(display_id, []).append((cell_id, out_idx))
                return

            if new_out["output_type"] == "stream":
                sname = new_out.get("name", "stdout")
                if outputs and outputs[-1].get("output_type") == "stream" and outputs[-1].get("name") == sname:
                    outputs[-1]["text"] = normalize_terminal_stream(
                        outputs[-1].get("text", ""),
                        new_out.get("text", ""),
                    )
                    outputs[-1]["_rev"] = new_out["_rev"]
                else:
                    new_out["text"] = normalize_terminal_stream("", new_out.get("text", ""))
                    outputs.append(new_out)
            else:
                outputs.append(new_out)

    def _update_display_by_id(self, fallback_cell_id: Optional[str], display_id: str, data: Dict[str, Any], metadata: Dict[str, Any]):
        rev = self._next_rev()
        updated_cells = set()

        with self._lock:
            # Scan active cache first for matching display_id
            for cid, outs in self.cell_outputs_cache.items():
                for idx, out in enumerate(outs):
                    if out.get("display_id") == display_id:
                        outs[idx]["data"] = data
                        outs[idx]["metadata"] = metadata
                        outs[idx]["_rev"] = rev
                        updated_cells.add(cid)

            # If not found in existing outputs, append to fallback_cell_id
            if not updated_cells and fallback_cell_id:
                self._append_output(
                    fallback_cell_id,
                    {
                        "output_type": "display_data",
                        "data": data,
                        "metadata": metadata,
                        "display_id": display_id,
                        "_rev": rev,
                    },
                    display_id=display_id,
                )
                updated_cells.add(fallback_cell_id)

        for cid in updated_cells:
            self._sync_cell_state(cid, force=False)

    def _iopub_listener_loop(self):
        while not self._stop_event.is_set():
            if not self.kc:
                time.sleep(0.1)
                continue
            try:
                msg = self.kc.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                continue
            except Exception:
                time.sleep(0.1)
                continue

            msg_type = msg.get("header", {}).get("msg_type", "")
            parent_id = msg.get("parent_header", {}).get("msg_id", "")
            content = msg.get("content", {})

            with self._lock:
                cell_id = self.msg_to_cell.get(parent_id) or self.active_cell_id

            if msg_type == "status":
                exec_state = content.get("execution_state", "")
                if exec_state in ("busy", "idle"):
                    self.kernel_state = exec_state
                    self._emit_ws({"type": "kernel_state", "state": exec_state})
                    if exec_state == "idle" and cell_id and parent_id in self.msg_to_cell:
                        self._sync_cell_state(cell_id, force=True)
                        cell = self.db.get_cell(cell_id)
                        if cell and cell["status"] == "running":
                            has_err = any(o.get("output_type") == "error" for o in cell.get("outputs", []))
                            final_status = "error" if has_err else "success"
                            updated = self.db.update_cell(cell_id, status=final_status, finished_at=time.time())
                            self._emit_ws({"type": "cell_updated", "cell": updated})
                        with self._lock:
                            self.msg_to_cell.pop(parent_id, None)
                            if self.active_cell_id == cell_id:
                                self.active_cell_id = None
                continue

            if not cell_id:
                continue

            if msg_type == "clear_output":
                wait_flag = bool(content.get("wait", False))
                with self._lock:
                    if wait_flag:
                        self.pending_clear[cell_id] = True
                    else:
                        self.pending_clear[cell_id] = False
                        self.cell_outputs_cache[cell_id] = []
                if not wait_flag:
                    self._sync_cell_state(cell_id, force=False)

            elif msg_type == "stream":
                self._append_output(
                    cell_id,
                    {
                        "output_type": "stream",
                        "name": content.get("name", "stdout"),
                        "text": content.get("text", ""),
                    },
                )
                self._sync_cell_state(cell_id, force=False)

            elif msg_type in ("display_data", "execute_result"):
                transient = content.get("transient") or {}
                display_id = transient.get("display_id")
                self._append_output(
                    cell_id,
                    {
                        "output_type": msg_type,
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                        "execution_count": content.get("execution_count"),
                    },
                    display_id=display_id,
                )
                self._sync_cell_state(cell_id, force=False)

            elif msg_type == "update_display_data":
                transient = content.get("transient") or {}
                display_id = transient.get("display_id")
                data = content.get("data", {})
                metadata = content.get("metadata", {})
                if display_id:
                    self._update_display_by_id(cell_id, display_id, data, metadata)
                else:
                    self._append_output(
                        cell_id,
                        {
                            "output_type": "display_data",
                            "data": data,
                            "metadata": metadata,
                        },
                    )
                    self._sync_cell_state(cell_id, force=False)

            elif msg_type == "error":
                self._append_output(
                    cell_id,
                    {
                        "output_type": "error",
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", ""),
                        "traceback": content.get("traceback", []),
                    },
                )
                self._sync_cell_state(cell_id, force=True)


kernel_orchestrator = KernelOrchestrator(state_db)


# ==============================================================================
# 9. FASTAPI APPLICATION, AUTHENTICATION & WEBSOCKET HUB
# ==============================================================================
app = FastAPI(title="Ultimate Cloud Notebook v4.0", docs_url=None, redoc_url=None)


class WebSocketHub:
    def __init__(self):
        self.clients: List[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)

    async def broadcast_async(self, message: Dict[str, Any]):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def broadcast_threadsafe(self, message: Dict[str, Any]):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast_async(message), self.loop)


ws_hub = WebSocketHub()
kernel_orchestrator.ws_broadcast_callback = ws_hub.broadcast_threadsafe


@app.on_event("startup")
async def on_startup():
    ws_hub.loop = asyncio.get_running_loop()
    asyncio.create_task(telemetry_push_loop())


async def telemetry_push_loop():
    while True:
        await asyncio.sleep(2.0)
        if ws_hub.clients:
            stats = hw_monitor.collect(kernel_orchestrator.kernel_state)
            await ws_hub.broadcast_async({"type": "telemetry", "telemetry": stats})


def verify_token_str(candidate: Optional[str]) -> bool:
    if not candidate:
        return False
    return secrets.compare_digest(candidate.strip(), AUTH_TOKEN)


async def require_auth(request: Request):
    q_tok = request.query_params.get("token")
    h_tok = request.headers.get("X-Auth-Token") or request.headers.get("Authorization", "").replace("Bearer ", "")
    c_tok = request.cookies.get("ucn_token")
    if verify_token_str(q_tok) or verify_token_str(h_tok) or verify_token_str(c_tok):
        return True
    raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing token.")


class CellCreateReq(BaseModel):
    cell_type: str = "code"
    source: str = ""
    after_id: Optional[str] = None


class CellUpdateReq(BaseModel):
    source: Optional[str] = None
    cell_type: Optional[str] = None


class CellMoveReq(BaseModel):
    direction: str


class ConfigSaveReq(BaseModel):
    config: Dict[str, Any]


@app.get("/api/state", dependencies=[Depends(require_auth)])
async def api_get_state():
    return {
        "cells": state_db.list_cells(),
        "telemetry": hw_monitor.collect(kernel_orchestrator.kernel_state),
        "workspace_dir": str(WORKSPACE_DIR),
        "is_kaggle": IS_KAGGLE,
    }


@app.post("/api/cells", dependencies=[Depends(require_auth)])
async def api_create_cell(req: CellCreateReq):
    cell = state_db.create_cell(cell_type=req.cell_type, source=req.source, after_id=req.after_id)
    await ws_hub.broadcast_async({"type": "cells_list", "cells": state_db.list_cells()})
    return cell


@app.put("/api/cells/{cid}", dependencies=[Depends(require_auth)])
async def api_update_cell(cid: str, req: CellUpdateReq):
    updates = {}
    if req.source is not None:
        updates["source"] = req.source
    if req.cell_type is not None:
        updates["cell_type"] = req.cell_type
    cell = state_db.update_cell(cid, **updates)
    if not cell:
        raise HTTPException(status_code=404, detail="Cell not found")
    state_db.export_to_ipynb(AUTOSAVE_IPYNB)
    await ws_hub.broadcast_async({"type": "cell_updated", "cell": cell})
    return cell


@app.delete("/api/cells/{cid}", dependencies=[Depends(require_auth)])
async def api_delete_cell(cid: str):
    state_db.delete_cell(cid)
    cells = state_db.list_cells()
    await ws_hub.broadcast_async({"type": "cells_list", "cells": cells})
    return {"cells": cells}


@app.post("/api/cells/{cid}/move", dependencies=[Depends(require_auth)])
async def api_move_cell(cid: str, req: CellMoveReq):
    cells = state_db.move_cell(cid, req.direction)
    await ws_hub.broadcast_async({"type": "cells_list", "cells": cells})
    return {"cells": cells}


@app.post("/api/cells/{cid}/clear", dependencies=[Depends(require_auth)])
async def api_clear_cell_output(cid: str):
    with kernel_orchestrator._lock:
        kernel_orchestrator.cell_outputs_cache[cid] = []
    cell = state_db.update_cell(cid, outputs=[], status="idle")
    await ws_hub.broadcast_async({"type": "cell_updated", "cell": cell})
    return cell


@app.post("/api/cells/{cid}/run", dependencies=[Depends(require_auth)])
async def api_run_cell(cid: str, req: Optional[CellUpdateReq] = None):
    if req and req.source is not None:
        state_db.update_cell(cid, source=req.source)
    cell = kernel_orchestrator.enqueue_cell(cid)
    if not cell:
        raise HTTPException(status_code=404, detail="Cell not found")
    return cell


@app.post("/api/kernel/run-all", dependencies=[Depends(require_auth)])
async def api_run_all():
    cells = kernel_orchestrator.enqueue_all()
    await ws_hub.broadcast_async({"type": "cells_list", "cells": cells})
    return {"cells": cells}


@app.post("/api/kernel/interrupt", dependencies=[Depends(require_auth)])
async def api_interrupt_kernel():
    kernel_orchestrator.interrupt_kernel()
    return {"status": "interrupted"}


@app.post("/api/kernel/restart", dependencies=[Depends(require_auth)])
async def api_restart_kernel():
    threading.Thread(target=kernel_orchestrator.restart_kernel, daemon=True).start()
    return {"status": "restarting"}


@app.post("/api/cells/clear-all", dependencies=[Depends(require_auth)])
async def api_clear_all_outputs():
    with kernel_orchestrator._lock:
        kernel_orchestrator.cell_outputs_cache.clear()
    cells = state_db.clear_all_outputs()
    await ws_hub.broadcast_async({"type": "cells_list", "cells": cells})
    return {"cells": cells}


@app.get("/api/notebook/export", dependencies=[Depends(require_auth)])
async def api_export_ipynb():
    path = state_db.export_to_ipynb(AUTOSAVE_IPYNB)
    return FileResponse(
        path=str(path),
        filename=f"cloud_notebook_{int(time.time())}.ipynb",
        media_type="application/x-ipynb+json",
    )


@app.post("/api/notebook/import", dependencies=[Depends(require_auth)])
async def api_import_ipynb(file: UploadFile = File(...)):
    raw = await file.read()
    cells = state_db.import_from_ipynb(raw.decode("utf-8", errors="replace"))
    await ws_hub.broadcast_async({"type": "cells_list", "cells": cells})
    return {"cells": cells}


def _get_allowed_roots() -> List[Path]:
    roots = [WORKSPACE_DIR.resolve(), BASE_ROOT.resolve()]
    for extra in ("/temp", "/tmp"):
        p = Path(extra)
        if p.exists():
            try:
                roots.append(p.resolve())
            except Exception:
                pass
    return roots


@app.get("/api/files", dependencies=[Depends(require_auth)])
async def api_list_workspace_files():
    audio_files = []
    checkpoints = []
    other_files = []

    scan_dirs = [WORKSPACE_DIR, BASE_ROOT]
    if Path("/temp").exists():
        scan_dirs.append(Path("/temp"))

    seen_paths = set()
    for sdir in scan_dirs:
        for root, dirs, files in os.walk(str(sdir)):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ("__pycache__", "node_modules", "venv", ".venv", "hf_cache", "raw_ds")
            ]
            for fn in files:
                if fn.startswith("."):
                    continue
                full = Path(root) / fn
                try:
                    resolved_str = str(full.resolve())
                except Exception:
                    continue
                if resolved_str in seen_paths:
                    continue
                seen_paths.add(resolved_str)
                try:
                    st = full.stat()
                except OSError:
                    continue

                rel_label = resolved_str
                for prefix in (str(BASE_ROOT), "/temp"):
                    if resolved_str.startswith(prefix):
                        rel_label = resolved_str[len(prefix) :].lstrip("/") or fn
                        if prefix == "/temp":
                            rel_label = f"temp/{rel_label}"
                        break

                ext = full.suffix.lower()
                item = {
                    "name": fn,
                    "rel_path": resolved_str,
                    "display_path": rel_label,
                    "size_mb": round(st.st_size / (1024**2), 2),
                    "mtime": st.st_mtime,
                }
                if ext in (".wav", ".mp3", ".ogg", ".flac", ".m4a"):
                    audio_files.append(item)
                elif ext in (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx"):
                    checkpoints.append(item)
                else:
                    other_files.append(item)

    audio_files.sort(key=lambda x: x["mtime"], reverse=True)
    checkpoints.sort(key=lambda x: x["mtime"], reverse=True)
    other_files.sort(key=lambda x: x["mtime"], reverse=True)
    return {
        "audio_files": audio_files[:60],
        "checkpoints": checkpoints[:60],
        "other_files": other_files[:100],
    }


@app.get("/api/media", dependencies=[Depends(require_auth)])
async def api_stream_media(path: str):
    raw_path = Path(path)
    target = (raw_path if raw_path.is_absolute() else (BASE_ROOT / path)).resolve()
    allowed_roots = _get_allowed_roots()
    if not any(str(target).startswith(str(r)) for r in allowed_roots):
        raise HTTPException(status_code=403, detail="Access denied")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(target), filename=target.name)


@app.get("/api/training-config", dependencies=[Depends(require_auth)])
async def api_get_training_config():
    if TRAINING_CONFIG_FILE.exists():
        try:
            return json.loads(TRAINING_CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"model_name": "oqim-kokoro-24khz", "batch_size": 8, "learning_rate": 0.0001, "sample_rate": 24000}


@app.post("/api/training-config", dependencies=[Depends(require_auth)])
async def api_save_training_config(req: ConfigSaveReq):
    TRAINING_CONFIG_FILE.write_text(json.dumps(req.config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "saved", "config": req.config}


@app.get("/api/logs", dependencies=[Depends(require_auth)])
async def api_get_system_logs():
    log_file = LOGS_DIR / "system.log"
    if not log_file.exists():
        return {"logs": ""}
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-250:]
    return {"logs": "\n".join(lines)}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    q_tok = ws.query_params.get("token")
    c_tok = ws.cookies.get("ucn_token")
    if not (verify_token_str(q_tok) or verify_token_str(c_tok)):
        await ws.close(code=1008)
        return

    await ws_hub.connect(ws)
    try:
        await ws.send_json(
            {
                "type": "init",
                "cells": state_db.list_cells(),
                "telemetry": hw_monitor.collect(kernel_orchestrator.kernel_state),
            }
        )
        while True:
            msg = await ws.receive_json()
            action = msg.get("action")
            if action == "ping":
                await ws.send_json({"type": "pong"})
            elif action == "save_source":
                cid = msg.get("cell_id")
                src = msg.get("source")
                if cid and src is not None:
                    state_db.update_cell(cid, source=src)
    except WebSocketDisconnect:
        ws_hub.disconnect(ws)
    except Exception:
        ws_hub.disconnect(ws)


# ==============================================================================
# 10. MODERN CLOUD NOTEBOOK FRONTEND (INCREMENTAL DOM & DISPLAY_ID ENGINE)
# ==============================================================================
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ultimate Cloud Notebook v4.0 — Pro AI & Audio Studio</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/tokyo-night-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.1/marked.min.js"></script>
<style>
  :root { color-scheme: dark; }
  body { background-color: #090d16; color: #e2e8f0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; overflow: hidden; }
  ::-webkit-scrollbar { width: 7px; height: 7px; }
  ::-webkit-scrollbar-track { background: #0f172a; }
  ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #475569; }
  .code-editor-wrap { position: relative; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13.5px; line-height: 1.55; }
  .code-textarea {
    width: 100%; background: #0b1120; color: #f1f5f9; caret-color: #38bdf8;
    border: none; outline: none; resize: none; padding: 12px 14px 12px 48px;
    font-family: inherit; font-size: inherit; line-height: inherit;
    tab-size: 4; white-space: pre; overflow-x: auto; display: block;
  }
  .line-numbers {
    position: absolute; top: 0; left: 0; bottom: 0; width: 38px;
    padding: 12px 6px 12px 0; text-align: right; color: #475569;
    background: #070b14; border-right: 1px solid #1e293b;
    user-select: none; pointer-events: none; font-family: inherit; font-size: 12px; line-height: 1.55;
  }
  .cell-card { transition: border-color 0.15s ease, box-shadow 0.15s ease; }
  .cell-card.running { border-color: #38bdf8 !important; box-shadow: 0 0 15px rgba(56, 189, 248, 0.15); }
  .cell-card.error { border-color: #f43f5e !important; }
  .cell-card.success { border-color: #10b981 !important; }
  .output-area { max-height: 780px; overflow-y: auto; background: #060911; border-top: 1px solid #1e293b; }
  .ansi-err { color: #fb7185; }
  .markdown-body h1, .markdown-body h2, .markdown-body h3 { font-weight: 700; color: #f8fafc; margin: 0.4em 0; }
  .markdown-body h3 { font-size: 1.1rem; }
  .markdown-body ul { list-style: disc; padding-left: 1.4rem; margin: 0.4em 0; }
  .markdown-body code { background: #1e293b; padding: 2px 5px; border-radius: 4px; color: #38bdf8; font-size: 0.9em; }
  @keyframes pulse-fast { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
  .animate-pulse-fast { animation: pulse-fast 1.1s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
</style>
</head>
<body class="h-screen flex flex-col select-none">

<!-- TOP NAVIGATION & HARDWARE TELEMETRY BAR -->
<header class="bg-slate-900/95 border-b border-slate-800 px-4 py-2.5 flex items-center justify-between gap-3 z-20 flex-wrap">
  <div class="flex items-center gap-3">
    <div class="flex items-center gap-2">
      <span class="w-3 h-3 rounded-full bg-sky-400 shadow-[0_0_10px_#38bdf8]"></span>
      <span class="font-bold text-sm tracking-wide text-white">ULTIMATE CLOUD NOTEBOOK</span>
      <span id="env-badge" class="text-[11px] font-mono px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/30">v4.0 PRO</span>
    </div>
    <div class="h-4 w-[1px] bg-slate-800 hidden sm:block"></div>
    <div class="flex items-center gap-2 text-xs font-mono">
      <span id="ws-dot" class="w-2 h-2 rounded-full bg-emerald-400"></span>
      <span id="ws-status" class="text-slate-300">Connected</span>
      <span class="text-slate-600">|</span>
      <span class="text-slate-400">Kernel:</span>
      <span id="kernel-badge" class="px-2 py-0.5 rounded font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">idle</span>
    </div>
  </div>

  <!-- REAL-TIME HARDWARE METRICS (CPU, RAM, DISK, DUAL GPU) -->
  <div class="flex items-center gap-3 text-xs font-mono flex-wrap">
    <div class="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded flex items-center gap-2">
      <span class="text-slate-400">CPU</span>
      <span id="hw-cpu" class="text-sky-400 font-semibold">0%</span>
    </div>
    <div class="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded flex items-center gap-2">
      <span class="text-slate-400">RAM</span>
      <span id="hw-ram" class="text-indigo-400 font-semibold">0 / 0 GB</span>
    </div>
    <div class="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded flex items-center gap-2">
      <span class="text-slate-400">DISK</span>
      <span id="hw-disk" class="text-amber-400 font-semibold">0 GB free</span>
    </div>
    <div id="gpu-container" class="flex items-center gap-2"></div>
  </div>

  <!-- ACTION TOOLBAR -->
  <div class="flex items-center gap-1.5">
    <button onclick="addCell('code')" class="px-2.5 py-1.5 rounded bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold flex items-center gap-1 transition">
      <span>+ Code</span>
    </button>
    <button onclick="addCell('markdown')" class="px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition">
      <span>+ Markdown</span>
    </button>
    <button onclick="runAllCells()" class="px-2.5 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition">
      ▶ Run All
    </button>
    <button onclick="interruptKernel()" class="px-2.5 py-1.5 rounded bg-amber-600/90 hover:bg-amber-500 text-white text-xs font-semibold transition" title="Interrupt Kernel (SIGINT)">
      ⏹ Stop
    </button>
    <button onclick="restartKernel()" class="px-2.5 py-1.5 rounded bg-rose-600/90 hover:bg-rose-500 text-white text-xs font-semibold transition" title="Restart IPython Kernel">
      ↻ Restart
    </button>
    <button onclick="clearAllOutputs()" class="px-2 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition" title="Clear All Outputs">
      🧹 Clear
    </button>
    <button onclick="exportNotebook()" class="px-2 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition" title="Download .ipynb">
      ⬇ .ipynb
    </button>
    <label class="px-2 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs cursor-pointer transition" title="Import .ipynb">
      ⬆ Import
      <input type="file" id="ipynb-upload" accept=".ipynb" class="hidden" onchange="importNotebook(event)">
    </label>
    <button onclick="toggleSidebar()" class="px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-sky-400 text-xs font-semibold transition">
      🎛 Studio Panel
    </button>
  </div>
</header>

<!-- MAIN WORKSPACE AREA -->
<div class="flex-1 flex overflow-hidden">

  <!-- NOTEBOOK CELLS SCROLL CONTAINER -->
  <main id="notebook-scroll" class="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-5 select-text">
    <div id="cells-container" class="max-w-5xl mx-auto space-y-5 pb-24"></div>
  </main>

  <!-- RIGHT STUDIO SIDEBAR: AUDIO PLAYER, CHECKPOINTS & LOGS -->
  <aside id="studio-sidebar" class="w-96 bg-slate-900/90 border-l border-slate-800 flex flex-col overflow-hidden">
    <div class="flex border-b border-slate-800 text-xs font-semibold">
      <button onclick="switchTab('audio')" id="tab-btn-audio" class="flex-1 py-2.5 text-sky-400 border-b-2 border-sky-400 bg-slate-950/40">🔊 Audio & Files</button>
      <button onclick="switchTab('config')" id="tab-btn-config" class="flex-1 py-2.5 text-slate-400 hover:text-slate-200">⚙ AI Config</button>
      <button onclick="switchTab('logs')" id="tab-btn-logs" class="flex-1 py-2.5 text-slate-400 hover:text-slate-200">📜 Logs</button>
    </div>

    <!-- TAB 1: AUDIO STUDIO & CHECKPOINTS -->
    <div id="tab-audio" class="flex-1 overflow-y-auto p-3.5 space-y-4 text-xs">
      <div class="flex items-center justify-between">
        <span class="font-bold text-slate-200 uppercase tracking-wider">Workspace Audio Studio</span>
        <button onclick="refreshFiles()" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-sky-400 font-mono">↻ Refresh</button>
      </div>
      <div id="audio-files-list" class="space-y-2"></div>

      <div class="pt-3 border-t border-slate-800">
        <span class="font-bold text-slate-200 uppercase tracking-wider block mb-2">Model Checkpoints (.pt / .ckpt)</span>
        <div id="checkpoint-files-list" class="space-y-1.5"></div>
      </div>

      <div class="pt-3 border-t border-slate-800">
        <span class="font-bold text-slate-200 uppercase tracking-wider block mb-2">Other Workspace & /temp Files</span>
        <div id="other-files-list" class="space-y-1"></div>
      </div>
    </div>

    <!-- TAB 2: PERSISTENT TRAINING CONFIG -->
    <div id="tab-config" class="hidden flex-1 overflow-y-auto p-3.5 space-y-3 text-xs">
      <div class="text-slate-400 leading-relaxed">
        Saved to <code class="text-sky-400">workspace/training_config.json</code>. Access in cells via <code class="text-emerald-400">ucloud.load_training_config()</code>.
      </div>
      <textarea id="config-editor" rows="16" class="w-full bg-slate-950 border border-slate-800 rounded p-2.5 font-mono text-xs text-slate-100 focus:border-sky-500 outline-none"></textarea>
      <button onclick="saveTrainingConfig()" class="w-full py-2 rounded bg-sky-600 hover:bg-sky-500 text-white font-semibold transition">💾 Save Training Config</button>
    </div>

    <!-- TAB 3: SYSTEM LOGS -->
    <div id="tab-logs" class="hidden flex-1 overflow-y-auto p-3.5 flex flex-col gap-2 text-xs">
      <div class="flex justify-between items-center">
        <span class="font-bold text-slate-300">Backend & Kernel Log</span>
        <button onclick="loadLogs()" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-sky-400">↻ Reload</button>
      </div>
      <pre id="system-logs-box" class="flex-1 bg-slate-950 border border-slate-800 rounded p-2.5 font-mono text-[11px] text-slate-300 overflow-auto whitespace-pre-wrap select-text"></pre>
    </div>
  </aside>
</div>

<script>
const urlParams = new URLSearchParams(window.location.search);
const AUTH_TOKEN = urlParams.get('token') || '';
let cellsState = [];
let cellMap = {};
let outputSignatures = {};
let activeTimers = {};
let ws = null;
let saveDebounceTimers = {};

function apiHeaders(extra = {}) {
  return Object.assign({ 'X-Auth-Token': AUTH_TOKEN }, extra);
}

async function apiFetch(path, opts = {}) {
  opts.headers = apiHeaders(opts.headers || {});
  const sep = path.includes('?') ? '&' : '?';
  const res = await fetch(`${path}${sep}token=${encodeURIComponent(AUTH_TOKEN)}`, opts);
  if (!res.ok) throw new Error(`API Error ${res.status}`);
  return res.json();
}

function connectWebSocket() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${proto}//${window.location.host}/ws?token=${encodeURIComponent(AUTH_TOKEN)}`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    document.getElementById('ws-dot').className = 'w-2 h-2 rounded-full bg-emerald-400';
    document.getElementById('ws-status').textContent = 'Live Sync';
  };

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'init') {
        updateTelemetryUI(msg.telemetry);
        reconcileCellsList(msg.cells);
      } else if (msg.type === 'cells_list') {
        reconcileCellsList(msg.cells);
      } else if (msg.type === 'cell_updated' && msg.cell) {
        patchSingleCell(msg.cell);
      } else if (msg.type === 'telemetry') {
        updateTelemetryUI(msg.telemetry);
      } else if (msg.type === 'kernel_state') {
        updateKernelBadge(msg.state);
      }
    } catch (e) { console.error(e); }
  };

  ws.onclose = () => {
    document.getElementById('ws-dot').className = 'w-2 h-2 rounded-full bg-rose-500 animate-ping';
    document.getElementById('ws-status').textContent = 'Reconnecting...';
    setTimeout(connectWebSocket, 2000);
  };
}

function updateKernelBadge(state) {
  const badge = document.getElementById('kernel-badge');
  badge.textContent = state;
  if (state === 'busy') {
    badge.className = 'px-2 py-0.5 rounded font-semibold bg-amber-500/15 text-amber-400 border border-amber-500/40 animate-pulse-fast';
  } else if (state === 'idle') {
    badge.className = 'px-2 py-0.5 rounded font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30';
  } else {
    badge.className = 'px-2 py-0.5 rounded font-semibold bg-rose-500/15 text-rose-400 border border-rose-500/40';
  }
}

function updateTelemetryUI(t) {
  if (!t) return;
  document.getElementById('env-badge').textContent = t.mode;
  updateKernelBadge(t.kernel_state);
  document.getElementById('hw-cpu').textContent = `${t.cpu_percent}%`;
  document.getElementById('hw-ram').textContent = `${t.ram_used_gb} / ${t.ram_total_gb} GB (${t.ram_percent}%)`;
  document.getElementById('hw-disk').textContent = `${t.disk_free_gb} GB free`;

  const gpuBox = document.getElementById('gpu-container');
  if (t.gpus && t.gpus.length > 0) {
    gpuBox.innerHTML = t.gpus.map(g => `
      <div class="bg-slate-950 border border-emerald-500/30 px-2.5 py-1 rounded flex items-center gap-2">
        <span class="text-emerald-400 font-bold">GPU${g.index}</span>
        <span class="text-slate-200">${g.vram_used_mb}/${g.vram_total_mb}MB</span>
        <span class="text-sky-400">(${g.util_percent}%)</span>
        <span class="text-amber-300">${g.temp_c}°C</span>
      </div>
    `).join('');
  } else {
    gpuBox.innerHTML = `<div class="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded text-slate-500">CPU Only</div>`;
  }
}

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function computeOutputSignature(out) {
  if (!out) return 'null';
  const rev = out._rev || 0;
  if (out.output_type === 'stream') {
    const txt = out.text || '';
    const tail = txt.slice(-64);
    return `stream:${out.name}:${txt.length}:${rev}:${tail}`;
  }
  if (out.output_type === 'error') {
    return `err:${out.ename}:${out.evalue}:${(out.traceback || []).length}:${rev}`;
  }
  const data = out.data || {};
  const keys = Object.keys(data).sort();
  const lens = keys.map(k => {
    const val = String(data[k] || '');
    return `${k}:${val.length}:${val.slice(-48)}`;
  }).join('|');
  return `${out.output_type}:${out.display_id || ''}:${rev}:${lens}`;
}

function createOutputItemDOM(out) {
  const wrapper = document.createElement('div');
  wrapper.className = 'output-item px-4 py-2 border-b border-slate-900/80 last:border-none font-mono text-xs';

  const otype = out.output_type;
  if (otype === 'stream') {
    const pre = document.createElement('pre');
    pre.className = `whitespace-pre-wrap break-words leading-relaxed ${out.name === 'stderr' ? 'text-rose-300/90' : 'text-slate-200'}`;
    pre.textContent = out.text || '';
    wrapper.appendChild(pre);
  } else if (otype === 'error') {
    const pre = document.createElement('pre');
    pre.className = 'whitespace-pre-wrap break-words text-rose-400 bg-rose-950/20 p-2.5 rounded border border-rose-500/30';
    const cleanTb = (out.traceback || []).join('\n').replace(/\x1b\[[0-9;]*m/g, '');
    pre.textContent = cleanTb || `${out.ename}: ${out.evalue}`;
    wrapper.appendChild(pre);
  } else if (otype === 'display_data' || otype === 'execute_result') {
    const d = out.data || {};
    if (d['text/html']) {
      const htmlBox = document.createElement('div');
      htmlBox.className = 'py-1 overflow-x-auto text-slate-100 font-sans';
      htmlBox.innerHTML = Array.isArray(d['text/html']) ? d['text/html'].join('') : d['text/html'];
      // Execute inline scripts if any
      htmlBox.querySelectorAll('script').forEach(oldScript => {
        const newScript = document.createElement('script');
        Array.from(oldScript.attributes).forEach(attr => newScript.setAttribute(attr.name, attr.value));
        newScript.textContent = oldScript.textContent;
        oldScript.parentNode.replaceChild(newScript, oldScript);
      });
      wrapper.appendChild(htmlBox);
    }
    if (d['image/png']) {
      const img = document.createElement('img');
      img.className = 'max-w-full h-auto rounded my-1.5 border border-slate-800';
      img.src = 'data:image/png;base64,' + String(d['image/png']).trim();
      wrapper.appendChild(img);
    }
    if (d['image/jpeg']) {
      const img = document.createElement('img');
      img.className = 'max-w-full h-auto rounded my-1.5 border border-slate-800';
      img.src = 'data:image/jpeg;base64,' + String(d['image/jpeg']).trim();
      wrapper.appendChild(img);
    }
    if (d['image/svg+xml']) {
      const svgBox = document.createElement('div');
      svgBox.className = 'my-1.5 overflow-x-auto';
      svgBox.innerHTML = Array.isArray(d['image/svg+xml']) ? d['image/svg+xml'].join('') : d['image/svg+xml'];
      wrapper.appendChild(svgBox);
    }
    if (d['audio/wav'] || d['audio/mpeg'] || d['audio/ogg']) {
      const mime = d['audio/wav'] ? 'audio/wav' : (d['audio/mpeg'] ? 'audio/mpeg' : 'audio/ogg');
      const b64 = String(d[mime]).trim();
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.className = 'my-1.5 h-9 w-full max-w-md';
      audio.src = `data:${mime};base64,${b64}`;
      wrapper.appendChild(audio);
    }
    if (d['text/markdown']) {
      const mdDiv = document.createElement('div');
      mdDiv.className = 'markdown-body font-sans py-1';
      const mdStr = Array.isArray(d['text/markdown']) ? d['text/markdown'].join('') : d['text/markdown'];
      mdDiv.innerHTML = marked.parse(mdStr);
      wrapper.appendChild(mdDiv);
    }
    if (d['application/json']) {
      const pre = document.createElement('pre');
      pre.className = 'text-sky-300 bg-slate-950 p-2 rounded';
      pre.textContent = JSON.stringify(d['application/json'], null, 2);
      wrapper.appendChild(pre);
    }
    if (!d['text/html'] && !d['image/png'] && !d['image/jpeg'] && !d['image/svg+xml'] && !d['text/markdown'] && d['text/plain']) {
      const pre = document.createElement('pre');
      pre.className = 'whitespace-pre-wrap break-words text-emerald-300';
      const txt = Array.isArray(d['text/plain']) ? d['text/plain'].join('') : d['text/plain'];
      pre.textContent = txt;
      wrapper.appendChild(pre);
    }
  }
  return wrapper;
}

function syncCellOutputsDOM(cell) {
  const outContainer = document.getElementById(`outputs-${cell.id}`);
  if (!outContainer) return;

  const outputs = cell.outputs || [];
  if (outputs.length === 0) {
    outContainer.innerHTML = '';
    outContainer.classList.add('hidden');
    outputSignatures[cell.id] = [];
    return;
  }

  outContainer.classList.remove('hidden');
  const prevSigs = outputSignatures[cell.id] || [];
  const nextSigs = outputs.map(computeOutputSignature);

  const wasAtBottom = (outContainer.scrollHeight - outContainer.scrollTop - outContainer.clientHeight) < 60;

  while (outContainer.children.length > outputs.length) {
    outContainer.removeChild(outContainer.lastChild);
  }

  for (let i = 0; i < outputs.length; i++) {
    const out = outputs[i];
    const sig = nextSigs[i];
    const existingNode = outContainer.children[i];

    if (!existingNode) {
      outContainer.appendChild(createOutputItemDOM(out));
    } else if (prevSigs[i] !== sig) {
      if (out.output_type === 'stream' && existingNode.firstChild && existingNode.firstChild.tagName === 'PRE') {
        existingNode.firstChild.textContent = out.text || '';
      } else {
        // Avoid interrupting an actively playing <audio> element if user is listening to it
        const playingAudio = existingNode.querySelector('audio');
        if (playingAudio && !playingAudio.paused && playingAudio.currentTime > 0) {
          continue;
        }
        const replacement = createOutputItemDOM(out);
        outContainer.replaceChild(replacement, existingNode);
      }
    }
  }

  outputSignatures[cell.id] = nextSigs;
  if (wasAtBottom && cell.status === 'running') {
    outContainer.scrollTop = outContainer.scrollHeight;
  }
}

function updateLineNumbers(cid) {
  const ta = document.getElementById(`editor-${cid}`);
  const ln = document.getElementById(`lines-${cid}`);
  if (!ta || !ln) return;
  ta.style.height = 'auto';
  ta.style.height = Math.max(68, ta.scrollHeight) + 'px';
  const count = (ta.value.match(/\n/g) || []).length + 1;
  let nums = '';
  for (let i = 1; i <= count; i++) nums += i + '\n';
  ln.textContent = nums;
}

function statusBadgeHTML(cell) {
  const st = cell.status || 'idle';
  if (st === 'running') {
    return `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-sky-500/20 text-sky-400 border border-sky-500/40 flex items-center gap-1.5">
      <span class="w-1.5 h-1.5 rounded-full bg-sky-400 animate-ping"></span>Running
    </span>`;
  }
  if (st === 'queued') {
    return `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-amber-500/15 text-amber-300 border border-amber-500/30">⏳ Queued</span>`;
  }
  if (st === 'success') {
    return `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">✔ Success</span>`;
  }
  if (st === 'error') {
    return `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-rose-500/15 text-rose-400 border border-rose-500/30">✖ Error</span>`;
  }
  return `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-slate-800 text-slate-400">Idle</span>`;
}

function formatDuration(cell) {
  if (!cell.started_at) return '';
  const end = cell.finished_at || (cell.status === 'running' ? Date.now() / 1000 : cell.started_at);
  const diff = Math.max(0, end - cell.started_at);
  return diff < 60 ? `${diff.toFixed(1)}s` : `${Math.floor(diff / 60)}m ${(diff % 60).toFixed(0)}s`;
}

function buildCellDOM(cell) {
  const card = document.createElement('div');
  card.id = `cell-${cell.id}`;
  card.className = `cell-card rounded-xl bg-slate-900/90 border border-slate-800 overflow-hidden shadow-lg ${cell.status}`;

  const isMd = cell.cell_type === 'markdown';
  const execLabel = isMd ? 'MD' : `[${cell.execution_count || ' '}]`;

  card.innerHTML = `
    <div class="px-3.5 py-2 bg-slate-950/70 border-b border-slate-800/80 flex items-center justify-between gap-2 flex-wrap">
      <div class="flex items-center gap-2.5">
        <span id="exec-${cell.id}" class="font-mono text-xs font-bold text-sky-400 min-w-[36px]">${execLabel}</span>
        <div id="status-${cell.id}">${statusBadgeHTML(cell)}</div>
        <span id="timer-${cell.id}" class="font-mono text-[11px] text-slate-400">${formatDuration(cell)}</span>
      </div>
      <div class="flex items-center gap-1">
        ${!isMd ? `
          <button onclick="runCell('${cell.id}')" class="px-2.5 py-1 rounded bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold transition" title="Run Cell (Shift+Enter)">▶ Run</button>
        ` : `
          <button onclick="toggleMarkdownEdit('${cell.id}')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-sky-400 text-xs font-medium transition">✎ Edit / Preview</button>
        `}
        <button onclick="moveCell('${cell.id}', 'up')" class="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 text-xs" title="Move Up">↑</button>
        <button onclick="moveCell('${cell.id}', 'down')" class="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 text-xs" title="Move Down">↓</button>
        <button onclick="copyCell('${cell.id}')" class="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 text-xs" title="Copy Code">📋</button>
        <button onclick="clearCellOutput('${cell.id}')" class="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 text-xs" title="Clear Output">🧹</button>
        <button onclick="addCell('${cell.cell_type}', '${cell.id}')" class="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 text-xs" title="Insert Cell Below">＋</button>
        <button onclick="deleteCell('${cell.id}')" class="p-1 rounded hover:bg-rose-950 text-slate-400 hover:text-rose-400 text-xs" title="Delete Cell">🗑</button>
      </div>
    </div>

    <div class="code-editor-wrap ${isMd ? 'hidden' : ''}" id="editor-wrap-${cell.id}">
      <div id="lines-${cell.id}" class="line-numbers">1</div>
      <textarea id="editor-${cell.id}" class="code-textarea" spellcheck="false" placeholder="Write Python code here..."></textarea>
    </div>

    ${isMd ? `<div id="md-preview-${cell.id}" ondblclick="toggleMarkdownEdit('${cell.id}')" class="markdown-body px-5 py-4 text-sm text-slate-200 cursor-pointer"></div>` : ''}

    <div id="outputs-${cell.id}" class="output-area hidden"></div>
  `;

  setTimeout(() => {
    const ta = document.getElementById(`editor-${cell.id}`);
    if (ta) {
      ta.value = cell.source || '';
      updateLineNumbers(cell.id);
      ta.addEventListener('input', () => {
        updateLineNumbers(cell.id);
        scheduleSourceSave(cell.id, ta.value);
      });
      ta.addEventListener('keydown', (e) => handleEditorKeys(e, cell.id));
    }
    if (isMd) {
      const mdPrev = document.getElementById(`md-preview-${cell.id}`);
      if (mdPrev) mdPrev.innerHTML = marked.parse(cell.source || '*Double-click to edit Markdown*');
    }
    syncCellOutputsDOM(cell);
    manageCellTimer(cell);
  }, 0);

  return card;
}

function patchSingleCell(cell) {
  cellMap[cell.id] = cell;
  const idx = cellsState.findIndex(c => c.id === cell.id);
  if (idx >= 0) cellsState[idx] = cell;

  const card = document.getElementById(`cell-${cell.id}`);
  if (!card) {
    reconcileCellsList(cellsState);
    return;
  }

  card.className = `cell-card rounded-xl bg-slate-900/90 border border-slate-800 overflow-hidden shadow-lg ${cell.status}`;
  const execEl = document.getElementById(`exec-${cell.id}`);
  if (execEl) execEl.textContent = cell.cell_type === 'markdown' ? 'MD' : `[${cell.execution_count || ' '}]`;

  const stEl = document.getElementById(`status-${cell.id}`);
  if (stEl) stEl.innerHTML = statusBadgeHTML(cell);

  const tmEl = document.getElementById(`timer-${cell.id}`);
  if (tmEl) tmEl.textContent = formatDuration(cell);

  const ta = document.getElementById(`editor-${cell.id}`);
  if (ta && document.activeElement !== ta && ta.value !== cell.source) {
    ta.value = cell.source || '';
    updateLineNumbers(cell.id);
  }

  syncCellOutputsDOM(cell);
  manageCellTimer(cell);
}

function reconcileCellsList(newCells) {
  cellsState = newCells || [];
  const container = document.getElementById('cells-container');
  const newIds = new Set(cellsState.map(c => c.id));

  Array.from(container.children).forEach(child => {
    const cid = child.id.replace('cell-', '');
    if (!newIds.has(cid)) {
      container.removeChild(child);
      delete cellMap[cid];
      delete outputSignatures[cid];
    }
  });

  cellsState.forEach((cell, index) => {
    cellMap[cell.id] = cell;
    let existing = document.getElementById(`cell-${cell.id}`);
    if (!existing) {
      existing = buildCellDOM(cell);
      if (index < container.children.length) {
        container.insertBefore(existing, container.children[index]);
      } else {
        container.appendChild(existing);
      }
    } else {
      if (container.children[index] !== existing) {
        container.insertBefore(existing, container.children[index]);
      }
      patchSingleCell(cell);
    }
  });
}

function manageCellTimer(cell) {
  if (activeTimers[cell.id]) {
    clearInterval(activeTimers[cell.id]);
    delete activeTimers[cell.id];
  }
  if (cell.status === 'running' && cell.started_at) {
    activeTimers[cell.id] = setInterval(() => {
      const tmEl = document.getElementById(`timer-${cell.id}`);
      if (tmEl) tmEl.textContent = formatDuration(cellMap[cell.id] || cell);
    }, 200);
  }
}

function handleEditorKeys(e, cid) {
  if (e.key === 'Tab') {
    e.preventDefault();
    const ta = e.target;
    const start = ta.selectionStart;
    const end = ta.selectionEnd;
    ta.value = ta.value.substring(0, start) + '    ' + ta.value.substring(end);
    ta.selectionStart = ta.selectionEnd = start + 4;
    updateLineNumbers(cid);
    scheduleSourceSave(cid, ta.value);
  } else if (e.key === 'Enter' && e.shiftKey) {
    e.preventDefault();
    runCell(cid, true);
  } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    runCell(cid, false);
  }
}

function scheduleSourceSave(cid, sourceText) {
  if (cellMap[cid]) cellMap[cid].source = sourceText;
  if (saveDebounceTimers[cid]) clearTimeout(saveDebounceTimers[cid]);
  saveDebounceTimers[cid] = setTimeout(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'save_source', cell_id: cid, source: sourceText }));
    } else {
      apiFetch(`/api/cells/${cid}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: sourceText })
      }).catch(() => {});
    }
  }, 350);
}

async function runCell(cid, moveNext = false) {
  const ta = document.getElementById(`editor-${cid}`);
  const src = ta ? ta.value : (cellMap[cid]?.source || '');
  await apiFetch(`/api/cells/${cid}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: src })
  });
  if (moveNext) {
    const idx = cellsState.findIndex(c => c.id === cid);
    if (idx >= 0 && idx + 1 < cellsState.length) {
      const nextId = cellsState[idx + 1].id;
      const nextTa = document.getElementById(`editor-${nextId}`);
      if (nextTa) nextTa.focus();
    } else {
      await addCell('code', cid);
    }
  }
}

async function addCell(cellType = 'code', afterId = null) {
  const created = await apiFetch('/api/cells', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cell_type: cellType, source: '', after_id: afterId })
  });
  setTimeout(() => {
    const ta = document.getElementById(`editor-${created.id}`);
    if (ta) ta.focus();
  }, 100);
}

async function deleteCell(cid) {
  await apiFetch(`/api/cells/${cid}`, { method: 'DELETE' });
}

async function moveCell(cid, direction) {
  await apiFetch(`/api/cells/${cid}/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ direction })
  });
}

async function clearCellOutput(cid) {
  await apiFetch(`/api/cells/${cid}/clear`, { method: 'POST' });
}

async function runAllCells() {
  await apiFetch('/api/kernel/run-all', { method: 'POST' });
}

async function interruptKernel() {
  await apiFetch('/api/kernel/interrupt', { method: 'POST' });
}

async function restartKernel() {
  if (!confirm('Restart IPython kernel? All in-memory variables will reset.')) return;
  await apiFetch('/api/kernel/restart', { method: 'POST' });
}

async function clearAllOutputs() {
  await apiFetch('/api/cells/clear-all', { method: 'POST' });
}

function copyCell(cid) {
  const ta = document.getElementById(`editor-${cid}`);
  if (ta) navigator.clipboard.writeText(ta.value);
}

function toggleMarkdownEdit(cid) {
  const wrap = document.getElementById(`editor-wrap-${cid}`);
  const prev = document.getElementById(`md-preview-${cid}`);
  const ta = document.getElementById(`editor-${cid}`);
  if (!wrap || !prev) return;
  const isHidden = wrap.classList.contains('hidden');
  if (isHidden) {
    wrap.classList.remove('hidden');
    prev.classList.add('hidden');
    updateLineNumbers(cid);
    ta.focus();
  } else {
    wrap.classList.add('hidden');
    prev.classList.remove('hidden');
    prev.innerHTML = marked.parse(ta.value || '*Double-click to edit Markdown*');
    scheduleSourceSave(cid, ta.value);
  }
}

function exportNotebook() {
  window.open(`/api/notebook/export?token=${encodeURIComponent(AUTH_TOKEN)}`, '_blank');
}

async function importNotebook(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  await apiFetch('/api/notebook/import', { method: 'POST', body: fd });
  ev.target.value = '';
}

function toggleSidebar() {
  document.getElementById('studio-sidebar').classList.toggle('hidden');
}

function switchTab(tab) {
  ['audio', 'config', 'logs'].forEach(t => {
    document.getElementById(`tab-${t}`).classList.toggle('hidden', t !== tab);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (t === tab) {
      btn.className = 'flex-1 py-2.5 text-sky-400 border-b-2 border-sky-400 bg-slate-950/40';
    } else {
      btn.className = 'flex-1 py-2.5 text-slate-400 hover:text-slate-200';
    }
  });
  if (tab === 'audio') refreshFiles();
  if (tab === 'config') loadTrainingConfig();
  if (tab === 'logs') loadLogs();
}

async function refreshFiles() {
  try {
    const data = await apiFetch('/api/files');
    const audioBox = document.getElementById('audio-files-list');
    if (!data.audio_files.length) {
      audioBox.innerHTML = `<div class="text-slate-500 italic">No audio files yet.</div>`;
    } else {
      audioBox.innerHTML = data.audio_files.map(f => {
        const mediaUrl = `/api/media?path=${encodeURIComponent(f.rel_path)}&token=${encodeURIComponent(AUTH_TOKEN)}`;
        return `
          <div class="p-2.5 rounded bg-slate-950 border border-slate-800 space-y-1.5">
            <div class="flex items-center justify-between gap-2">
              <span class="font-mono text-sky-300 truncate" title="${escapeHtml(f.display_path)}">${escapeHtml(f.name)}</span>
              <a href="${mediaUrl}" download="${escapeHtml(f.name)}" class="text-slate-400 hover:text-white">⬇ ${f.size_mb}MB</a>
            </div>
            <audio controls preload="none" class="w-full h-8" src="${mediaUrl}"></audio>
          </div>
        `;
      }).join('');
    }

    const ckptBox = document.getElementById('checkpoint-files-list');
    if (!data.checkpoints.length) {
      ckptBox.innerHTML = `<div class="text-slate-500 italic">No checkpoints saved yet.</div>`;
    } else {
      ckptBox.innerHTML = data.checkpoints.map(f => {
        const mediaUrl = `/api/media?path=${encodeURIComponent(f.rel_path)}&token=${encodeURIComponent(AUTH_TOKEN)}`;
        return `
          <div class="px-2.5 py-1.5 rounded bg-slate-950 border border-slate-800 flex items-center justify-between">
            <span class="font-mono text-emerald-400 truncate" title="${escapeHtml(f.display_path)}">${escapeHtml(f.name)}</span>
            <a href="${mediaUrl}" download="${escapeHtml(f.name)}" class="text-slate-400 hover:text-white font-mono">⬇ ${f.size_mb}MB</a>
          </div>
        `;
      }).join('');
    }

    const otherBox = document.getElementById('other-files-list');
    otherBox.innerHTML = (data.other_files || []).slice(0, 30).map(f => {
      const mediaUrl = `/api/media?path=${encodeURIComponent(f.rel_path)}&token=${encodeURIComponent(AUTH_TOKEN)}`;
      return `
        <div class="px-2 py-1 rounded hover:bg-slate-950 flex items-center justify-between text-[11px]">
          <span class="font-mono text-slate-300 truncate" title="${escapeHtml(f.display_path)}">${escapeHtml(f.display_path)}</span>
          <a href="${mediaUrl}" download="${escapeHtml(f.name)}" class="text-slate-500 hover:text-sky-400">${f.size_mb}MB</a>
        </div>
      `;
    }).join('');
  } catch (e) {}
}

async function loadTrainingConfig() {
  try {
    const cfg = await apiFetch('/api/training-config');
    document.getElementById('config-editor').value = JSON.stringify(cfg, null, 2);
  } catch (e) {}
}

async function saveTrainingConfig() {
  try {
    const parsed = JSON.parse(document.getElementById('config-editor').value);
    await apiFetch('/api/training-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: parsed })
    });
    alert('Training configuration saved!');
  } catch (e) {
    alert('Invalid JSON format: ' + e.message);
  }
}

async function loadLogs() {
  try {
    const res = await apiFetch('/api/logs');
    const box = document.getElementById('system-logs-box');
    box.textContent = res.logs || '';
    box.scrollTop = box.scrollHeight;
  } catch (e) {}
}

window.addEventListener('DOMContentLoaded', () => {
  connectWebSocket();
  refreshFiles();
  setInterval(refreshFiles, 12000);
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    q_tok = request.query_params.get("token")
    c_tok = request.cookies.get("ucn_token")
    if not (verify_token_str(q_tok) or verify_token_str(c_tok)):
        return HTMLResponse(
            content="""
            <html><body style="background:#090d16;color:#f8fafc;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
            <div style="background:#111827;padding:28px;border-radius:12px;border:1px solid #1f2937;max-width:420px;width:100%;">
                <h3 style="margin-top:0;color:#38bdf8;">🔒 Ultimate Cloud Notebook Auth</h3>
                <p style="font-size:13px;color:#94a3b8;">Enter your access token or open the full Cloudflare link printed in the console.</p>
                <form method="get" action="/">
                    <input type="password" name="token" placeholder="Paste Token..." style="width:100%;padding:10px;border-radius:6px;border:1px solid #374151;background:#030712;color:white;margin-bottom:12px;">
                    <button type="submit" style="width:100%;padding:10px;border-radius:6px;background:#0284c7;color:white;border:none;font-weight:bold;cursor:pointer;">Unlock Notebook</button>
                </form>
            </div></body></html>
            """,
            status_code=401,
        )
    resp = HTMLResponse(content=FRONTEND_HTML)
    resp.set_cookie(key="ucn_token", value=AUTH_TOKEN, httponly=True, samesite="lax")
    return resp


# ==============================================================================
# 11. AUTOMATIC CLOUDFLARE TUNNEL (TRYCLOUDFLARE.COM) MANAGER
# ==============================================================================
def ensure_cloudflared_binary() -> str:
    bin_path = APP_DIR / "cloudflared"
    if shutil.which("cloudflared"):
        return shutil.which("cloudflared")
    if bin_path.exists() and os.access(str(bin_path), os.X_OK):
        return str(bin_path)

    machine = platform.machine().lower()
    arch = "arm64" if ("aarch64" in machine or "arm64" in machine) else "amd64"
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
    logger.info(f"Downloading cloudflared binary from {url} ...")
    urllib.request.urlretrieve(url, str(bin_path))
    os.chmod(str(bin_path), 0o755)
    return str(bin_path)


def start_cloudflare_tunnel(local_port: int, timeout: int = 35) -> Optional[str]:
    try:
        cf_bin = ensure_cloudflared_binary()
    except Exception as e:
        logger.error(f"Could not install cloudflared: {e}")
        return None

    import re
    cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{local_port}", "--no-autoupdate"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url_pattern = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
    found_url = None
    start_t = time.time()

    def _reader():
        nonlocal found_url
        for line in iter(proc.stdout.readline, ""):
            m = url_pattern.search(line)
            if m and not found_url:
                found_url = m.group(0)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    while time.time() - start_t < timeout:
        if found_url:
            TUNNEL_URL_FILE.write_text(found_url, encoding="utf-8")
            return found_url
        time.sleep(0.25)
    return None


# ==============================================================================
# 12. NON-BLOCKING KAGGLE / COLAB / CLI LAUNCHER
# ==============================================================================
def _is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def launch_in_notebook_or_cli():
    import uvicorn

    in_interactive_nb = "ipykernel" in sys.modules

    if not _is_port_in_use(SERVER_PORT):
        config = uvicorn.Config(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        srv_thread = threading.Thread(target=server.run, daemon=True)
        srv_thread.start()
        time.sleep(1.5)
        public_base = start_cloudflare_tunnel(SERVER_PORT)
    else:
        logger.info(f"Server already running on port {SERVER_PORT}, reusing existing session.")
        public_base = TUNNEL_URL_FILE.read_text(encoding="utf-8").strip() if TUNNEL_URL_FILE.exists() else None
        if not public_base:
            public_base = start_cloudflare_tunnel(SERVER_PORT)

    local_url = f"http://127.0.0.1:{SERVER_PORT}/?token={AUTH_TOKEN}"
    public_url = f"{public_base}/?token={AUTH_TOKEN}" if public_base else None

    banner = [
        "=" * 78,
        "🚀 ULTIMATE CLOUD NOTEBOOK v4.0 IS LIVE AND RUNNING!",
        "=" * 78,
        f"Mode            : {'Kaggle Mode (/kaggle/working)' if IS_KAGGLE else 'Standalone VPS Mode'}",
        f"Workspace Path  : {WORKSPACE_DIR}",
        f"Autosave .ipynb : {AUTOSAVE_IPYNB}",
        f"Local URL       : {local_url}",
        f"Cloudflare URL  : {public_url or 'Tunnel unavailable (use Local URL)'}",
        "=" * 78,
    ]
    print("\n".join(banner), flush=True)

    if in_interactive_nb:
        try:
            from IPython.display import display, HTML
            target_link = public_url or local_url
            display(
                HTML(
                    f"""
                    <div style="margin:14px 0;padding:18px 22px;background:#0f172a;border:2px solid #0284c7;border-radius:12px;color:#f8fafc;font-family:sans-serif;">
                        <div style="font-size:16px;font-weight:800;color:#38bdf8;margin-bottom:8px;">
                            ⚡ ULTIMATE CLOUD NOTEBOOK v4.0 — PUBLIC CONTROL PANEL
                        </div>
                        <div style="font-size:13px;color:#cbd5e1;margin-bottom:14px;">
                            Brauzerni yopsangiz ham runtime va AI mashg'ulotlar to'xtamaydi. Quyidagi havola orqali istalgan vaqtda qayta kiring:
                        </div>
                        <a href="{target_link}" target="_blank"
                           style="display:inline-block;padding:10px 20px;background:#0284c7;color:#ffffff;font-weight:700;text-decoration:none;border-radius:8px;font-size:14px;">
                            🌐 OPEN CLOUD NOTEBOOK STUDIO ↗
                        </a>
                        <div style="margin-top:12px;font-family:monospace;font-size:12px;color:#94a3b8;word-break:break-all;">
                            URL: {target_link}
                        </div>
                    </div>
                    """
                )
            )
        except Exception:
            pass

    # Keep cell alive so Kaggle/Colab session stays active while background threads run
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[Ultimate Cloud Notebook] Launcher interrupted. Background server remains active in session.")


if __name__ == "__main__":
    launch_in_notebook_or_cli()
