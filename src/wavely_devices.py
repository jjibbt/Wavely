"""Windows device discovery used by the WAVELY settings and speech paths."""

import ctypes
import json
import subprocess


class _WaveCaps(ctypes.Structure):
    _fields_ = [
        ("manufacturer_id", ctypes.c_ushort),
        ("product_id", ctypes.c_ushort),
        ("driver_version", ctypes.c_ulong),
        ("product_name", ctypes.c_wchar * 32),
        ("formats", ctypes.c_ulong),
        ("channels", ctypes.c_ushort),
        ("reserved", ctypes.c_ushort),
    ]


def enumerate_windows_cameras():
    """Return present Windows camera names and stable PnP instance IDs."""
    command = (
        "$items = @(Get-PnpDevice -PresentOnly | "
        "Where-Object { $_.Class -in @('Camera', 'Image') -and $_.Status -eq 'OK' } | "
        "Select-Object FriendlyName, InstanceId); "
        "$items | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        data = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        data = [data]
    return [
        {"name": item.get("FriendlyName", "").strip(), "device_id": item.get("InstanceId", "").strip()}
        for item in data
        if isinstance(item, dict) and item.get("FriendlyName")
    ]


def _enumerate_wave_devices(function_name):
    function = getattr(ctypes.windll.winmm, function_name)
    count = function()
    devices = []
    caps_function = getattr(ctypes.windll.winmm, "waveInGetDevCapsW" if function_name == "waveInGetNumDevs" else "waveOutGetDevCapsW")
    for device_id in range(count):
        caps = _WaveCaps()
        if caps_function(device_id, ctypes.byref(caps), ctypes.sizeof(caps)) == 0:
            devices.append({"device_id": device_id, "name": caps.product_name.strip()})
    return devices


def _enumerate_audio_endpoints():
    command = (
        "Get-PnpDevice -PresentOnly -Class AudioEndpoint | "
        "Where-Object Status -eq 'OK' | Select-Object FriendlyName, InstanceId | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        data = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        data = [data]
    return [
        {"device_id": item.get("InstanceId", "").strip(), "name": item.get("FriendlyName", "").strip(),
         "direction": "input" if "{0.0.1." in item.get("InstanceId", "") else "output"}
        for item in data
        if isinstance(item, dict) and item.get("FriendlyName") and item.get("InstanceId")
    ]


def enumerate_audio_inputs():
    endpoints = [item for item in _enumerate_audio_endpoints() if item["direction"] == "input"]
    if endpoints:
        return endpoints
    try:
        return _enumerate_wave_devices("waveInGetNumDevs")
    except (AttributeError, OSError):
        return []


def enumerate_audio_outputs():
    """Return SAPI output tokens, which pyttsx3 can select directly."""
    try:
        import comtypes.client

        category = comtypes.client.CreateObject("SAPI.SpObjectTokenCategory")
        category.SetId("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\AudioOutput")
        tokens = category.EnumerateTokens()
        outputs = [{"device_id": token.Id, "name": token.GetDescription()} for token in tokens]
        if outputs:
            return outputs
    except Exception:
        pass
    endpoints = [item for item in _enumerate_audio_endpoints() if item["direction"] == "output"]
    if endpoints:
        return endpoints
    try:
        return [{"device_id": f"waveout:{item['device_id']}", "name": item["name"]} for item in _enumerate_wave_devices("waveOutGetNumDevs")]
    except (AttributeError, OSError):
        return []


def configure_speech_output(engine, output_id):
    """Apply a saved SAPI output token when the installed speech driver supports it."""
    if not output_id:
        return False
    try:
        import comtypes.client

        category = comtypes.client.CreateObject("SAPI.SpObjectTokenCategory")
        category.SetId("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\AudioOutput")
        for token in category.EnumerateTokens():
            if token.Id == output_id:
                engine._driver._tts.AudioOutput = token
                return True
    except Exception:
        pass
    return False