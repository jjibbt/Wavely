"""Installed resources and per-user writable data for WAVELY."""

import os
import shutil
import sys
from pathlib import Path


APP_ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[1]
)
if APP_ROOT.name.lower() == "runtime":
    APP_ROOT = APP_ROOT.parent

DATA_ROOT = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Wavely"
CONFIG_DIR = DATA_ROOT / "config"
FACES_DIR = DATA_ROOT / "faces"
LOG_DIR = DATA_ROOT / "logs"


def prepare_user_data():
    """Copy existing data once, leaving the original installation untouched."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "wavely_dashboard.json", "wavely_window.json", "wavely_console_window.json",
        "wavely_app_settings.json", "home_assistant_actions.json",
        "face_model.yml", "face_labels.json",
    ):
        source = APP_ROOT / "config" / name
        target = CONFIG_DIR / name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
    if not FACES_DIR.exists():
        source = APP_ROOT / "assets" / "faces"
        if source.is_dir():
            shutil.copytree(source, FACES_DIR)
        else:
            FACES_DIR.mkdir(parents=True)
    source_log = APP_ROOT / "logs" / "wavely_activity.log"
    target_log = LOG_DIR / source_log.name
    if source_log.is_file() and not target_log.exists():
        shutil.copy2(source_log, target_log)
