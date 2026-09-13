"""Portable WAVELY settings files, excluding biometric and log data."""

import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from wavely_actions import GESTURES


APP_KEYS = (
    "camera_index", "camera_name", "camera_device_id",
    "audio_input_device_id", "audio_input_name",
    "audio_output_device_id", "audio_output_name",
    "resolution", "mirror", "gesture_sensitivity", "start_with_windows",
)
RESOLUTIONS = ("640x480", "1280x720", "1920x1080")


def validate_bundle(bundle):
    if not isinstance(bundle, dict) or bundle.get("format") != "wavely-settings" or bundle.get("version") != 1:
        raise ValueError("This is not a supported WAVELY settings file.")
    app = bundle.get("app")
    home = bundle.get("home_assistant")
    if not isinstance(app, dict) or not isinstance(home, dict):
        raise ValueError("The settings file is incomplete.")
    camera = app.get("camera_index")
    sensitivity = app.get("gesture_sensitivity")
    if type(camera) is not int or not 0 <= camera <= 9:
        raise ValueError("The camera number must be between 0 and 9.")
    if app.get("resolution") not in RESOLUTIONS:
        raise ValueError("The camera resolution is unsupported.")
    if type(app.get("mirror")) is not bool or type(app.get("start_with_windows")) is not bool:
        raise ValueError("A Windows or camera preference is invalid.")
    if type(sensitivity) not in (int, float) or not 0.4 <= sensitivity <= 0.9:
        raise ValueError("The gesture sensitivity is invalid.")
    for key in ("camera_name", "camera_device_id", "audio_input_name", "audio_output_device_id", "audio_output_name"):
        if key in app and not isinstance(app[key], str):
            raise ValueError("A saved device name or identifier is invalid.")
    if "audio_input_device_id" in app and type(app["audio_input_device_id"]) not in (int, str):
        raise ValueError("The audio input identifier is invalid.")

    url = home.get("webhook_url", "")
    if not isinstance(url, str):
        raise ValueError("The Home Assistant address is invalid.")
    if url:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("The Home Assistant address must be a full HTTP or HTTPS URL.")
    actions = home.get("actions")
    if not isinstance(actions, list):
        raise ValueError("The action list is invalid.")
    seen = set()
    for item in actions:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not re.fullmatch(r"[a-z0-9_]+", item["id"]):
            raise ValueError("An action ID is invalid.")
        gesture = item.get("gesture")
        if gesture is not None:
            if gesture not in GESTURES or gesture in seen:
                raise ValueError("An action has an unknown or duplicate gesture.")
            seen.add(gesture)
        if not isinstance(item.get("label", ""), str):
            raise ValueError("An action label is invalid.")
        entity = item.get("entity_id", "")
        if not isinstance(entity, str) or (entity and not re.fullmatch(r"[a-z_]+\.[a-z0-9_]+", entity)):
            raise ValueError("An entity ID is invalid.")
    return bundle


def make_bundle(app_config, home_config):
    bundle = {
        "format": "wavely-settings",
        "version": 1,
        "app": {
            key: app_config.get(key, "" if key != "camera_index" and key != "audio_input_device_id" else (0 if key == "camera_index" else -1))
            for key in APP_KEYS
        },
        "home_assistant": {
            "webhook_url": home_config.get("webhook_url", ""),
            "actions": home_config.get("actions", []),
        },
    }
    return validate_bundle(bundle)


def load_bundle(path):
    try:
        bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read the settings file: {error}") from error
    return validate_bundle(bundle)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(data, file, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_bundle(bundle, config_dir, data_root):
    validate_bundle(bundle)
    config_dir = Path(config_dir)
    backup_dir = Path(data_root) / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir.mkdir(parents=True)
    targets = {
        "wavely_app_settings.json": {**bundle["app"], "setup_complete": False},
        "home_assistant_actions.json": bundle["home_assistant"],
    }
    previous = {}
    for name in targets:
        target = config_dir / name
        previous[name] = target.read_bytes() if target.exists() else None
        if target.exists():
            shutil.copy2(target, backup_dir / name)
    try:
        for name, data in targets.items():
            write_json(config_dir / name, data)
    except OSError:
        for name, content in previous.items():
            target = config_dir / name
            if content is None:
                target.unlink(missing_ok=True)
            else:
                temporary = target.with_suffix(".restore")
                temporary.write_bytes(content)
                os.replace(temporary, target)
        raise
    return backup_dir
