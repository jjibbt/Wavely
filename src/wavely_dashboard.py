"""Single-window control panel for WAVELY Vision on Windows."""

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from urllib.parse import urlparse
from wavely_actions import GESTURES, payload_for_gesture
from wavely_devices import enumerate_audio_inputs, enumerate_audio_outputs, enumerate_windows_cameras
from wavely_paths import APP_ROOT as ROOT, CONFIG_DIR, DATA_ROOT, FACES_DIR, LOG_DIR, prepare_user_data
from wavely_settings import apply_bundle, load_bundle, make_bundle, write_json


prepare_user_data()
SETTINGS_FILE = CONFIG_DIR / "wavely_dashboard.json"
CAMERA_ASPECT_FILE = CONFIG_DIR / "active_camera_aspect.json"
APP_CONFIG_FILE = CONFIG_DIR / "wavely_app_settings.json"
ACTION_CONFIG_FILE = CONFIG_DIR / "home_assistant_actions.json"
ACTIVITY_FILE = LOG_DIR / "wavely_activity.log"
VISION_SCRIPT = ROOT / "src" / "wavely_vision.py"
VISION_PROGRAM = ROOT / "runtime" / "WavelyVision.exe"
ENROLMENT_SCRIPT = ROOT / "src" / "video_enrol.py"
TRAINING_SCRIPT = ROOT / "src" / "train_all_faces.py"
# Use the windowless Python runner: progress is captured in WAVELY Logs rather
# than opening a separate command prompt during enrolment or model training.
ENROLMENT_PROGRAM = ROOT / "runtime" / "WavelyEnrol.exe"
TRAINING_PROGRAM = ROOT / "runtime" / "WavelyTrain.exe"
WINDOW_TITLE = "WAVELY Vision"
ENROLMENT_WINDOW_TITLE = "WAVELY Video Face Enrolment"

GWL_STYLE = -16
WS_CHILD = 0x40000000
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_POPUP = 0x80000000
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020

user32 = ctypes.windll.user32


class Tooltip:
    """Small dark hover label for the compact toolbar icon buttons."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _event=None):
        if self.tip:
            return
        self.tip = tk.Toplevel(self.widget)
        self.tip.overrideredirect(True)
        self.tip.configure(bg="#171d25")
        x = self.widget.winfo_rootx() + 4
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip.geometry(f"+{x}+{y}")
        tk.Label(
            self.tip, text=self.text, bg="#171d25", fg="#f2f6fa",
            font=("Segoe UI", 9), padx=8, pady=4,
        ).pack()

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class WavelyDashboard:
    def __init__(self):
        self.process = None
        self.process_mode = None
        self.vision_handle = 0
        self.settings = self.load_settings()
        self.app_config = self.load_json(APP_CONFIG_FILE, {
            "camera_index": 0, "camera_name": "", "camera_device_id": "",
            "audio_input_device_id": -1, "audio_input_name": "",
            "audio_output_device_id": "", "audio_output_name": "",
            "resolution": "1280x720", "mirror": True,
            "gesture_sensitivity": 0.60, "start_with_windows": False,
        })
        self.layout_ready = False
        self.command_history = self.settings.get("command_history", [])
        if not isinstance(self.command_history, list):
            self.command_history = []
        self.history_index = len(self.command_history)
        self.pending_deletion = None
        self.active_enrolment_person = None
        self.enrolment_start_image_count = 0
        self.enrolment_stopped_early = False
        self.stream_aspect = 16 / 9
        self.aspect_file_mtime = None
        self.theme_defaults = {}
        saved_theme = self.settings.get("theme")
        self.theme_is_dark = saved_theme == "dark" if saved_theme in {"dark", "light"} else True

        self.root = tk.Tk()
        self.root.title("WAVELY Vision Control")
        self.root.configure(bg="#10151c")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        self.configure_vision_button_styles(style)
        style.configure(
            "Wavely.TCombobox",
            fieldbackground="#171d25",
            background="#356da8",
            foreground="#f2f6fa",
            arrowcolor="#f2f6fa",
            bordercolor="#356da8",
            lightcolor="#356da8",
            darkcolor="#356da8",
        )
        style.map(
            "Wavely.TCombobox",
            fieldbackground=[("readonly", "#171d25")],
            foreground=[("readonly", "#f2f6fa")],
            selectbackground=[("readonly", "#171d25")],
            selectforeground=[("readonly", "#f2f6fa")],
        )
        self.root.minsize(1000, 620)
        default_geometry = "1520x850+120+80"
        saved_geometry = self.settings.get("geometry", default_geometry)
        if not isinstance(saved_geometry, str) or not re.fullmatch(r"\d+x\d+[+-]\d+[+-]\d+", saved_geometry):
            saved_geometry = default_geometry
        try:
            self.root.geometry(saved_geometry)
        except tk.TclError:
            self.root.geometry(default_geometry)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Configure>", self.save_layout_later)
        self.root.bind("<Control-comma>", lambda _event: self.open_settings())
        self.root.bind("<F5>", lambda _event: self.open_settings())
        self.root.bind("<Control-l>", lambda _event: self.open_activity_log())
        self.root.after(700, self.enable_layout_saving)

        if self.settings.get("maximized"):
            self.root.after(100, lambda: self.root.state("zoomed"))

        toolbar = tk.Frame(self.root, bg="#10151c", padx=12, pady=10)
        self.toolbar = toolbar
        toolbar.pack(fill="x")

        self.toggle_button = ttk.Button(
            toolbar,
            text="Start WAVELY Vision",
            command=self.toggle_vision,
            style="Vision.Start.TButton",
        )
        self.toggle_button.pack(side="left", fill="y")

        enrol_group = tk.Frame(toolbar, bg="#10151c")
        self.enrol_group = enrol_group
        enrol_group.pack(side="left", padx=(8, 0))

        self.enrol_button = tk.Menubutton(
            enrol_group,
            text="Video Enrolment  ▾",
            bg="#356da8",
            fg="white",
            activebackground="#285681",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=7,
            font=("Segoe UI", 11, "bold"),
        )
        self.enrol_button.pack(fill="x")
        self.enrol_menu = tk.Menu(
            self.enrol_button, tearoff=False, bg="#171d25", fg="#f2f6fa",
            activebackground="#356da8", activeforeground="white", relief="flat",
            font=("Segoe UI", 10),
        )
        self.enrol_menu.add_command(label="All poses", command=lambda: self.start_enrolment(pose_index=0))
        self.enrol_menu.add_separator()
        for pose_index, pose_label in enumerate(self.enrolment_pose_labels(), start=1):
            self.enrol_menu.add_command(
                label=pose_label,
                command=lambda index=pose_index: self.start_enrolment(pose_index=index),
            )
        self.enrol_button.configure(menu=self.enrol_menu)
        self.enrol_button.bind("<Button-1>", self.enrol_button_pressed, add="+")

        people = self.available_people()
        self.person_var = tk.StringVar(value="Select person...")
        self.person_picker = ttk.Combobox(
            enrol_group,
            textvariable=self.person_var,
            values=["Select person..."] + people,
            state="readonly",
            width=20,
            style="Wavely.TCombobox",
        )
        self.person_picker.pack(fill="x", pady=(12, 0), ipady=2)
        self.tools_button = tk.Menubutton(toolbar, text="Tools  ▾", bg="#273343", fg="white", activebackground="#356da8", activeforeground="white", relief="flat", padx=14, pady=22, font=("Segoe UI", 10, "bold"))
        self.tools_button.pack(side="left", padx=(10, 0), fill="y")
        self.tools_menu = tk.Menu(self.tools_button, tearoff=False, bg="#171d25", fg="#f2f6fa", activebackground="#356da8", activeforeground="white", font=("Segoe UI", 10))
        self.tools_menu.add_command(label="Settings and camera", command=self.open_settings)
        self.tools_menu.add_command(label="Export settings...", command=self.export_settings)
        self.tools_menu.add_command(label="Import settings...", command=self.import_settings)
        self.tools_menu.add_command(label="People manager", command=self.open_people_manager)
        self.tools_menu.add_command(label="Gesture test mode", command=self.start_test_mode)
        self.tools_menu.add_command(label="Home Assistant actions", command=self.open_home_actions)
        self.tools_menu.add_command(label="Health check", command=self.open_health_check)
        self.tools_menu.add_command(label="Activity log", command=self.open_activity_log)
        self.tools_button.configure(menu=self.tools_menu)
        self.theme_button = tk.Button(
            toolbar,
            text="☀" if self.theme_is_dark else "☾",
            command=self.toggle_theme,
            bg="#273343",
            fg="white",
            activebackground="#356da8",
            activeforeground="white",
            relief="flat",
            padx=10,
            pady=7,
            font=("Segoe UI Symbol", 12, "bold"),
        )
        self.theme_button.pack(side="left", padx=(6, 0), fill="y")
        self.theme_tooltip = Tooltip(self.theme_button, "Switch to light mode" if self.theme_is_dark else "Switch to dark mode")

        self.status_box = tk.Frame(toolbar, bg="#10151c")
        self.status_box.pack(side="right", padx=(12, 0), fill="y")
        self.status_text = tk.Frame(self.status_box, bg="#10151c")
        self.status_text.pack(side="left")
        self.status_line = tk.Frame(self.status_text, bg="#10151c")
        self.status_line.pack(anchor="e")
        self.status_dot = tk.Canvas(self.status_line, width=14, height=14, bg="#10151c", highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 7), pady=(2, 0))
        self.status_circle = self.status_dot.create_oval(2, 2, 12, 12, fill="#cc4b4c", outline="")
        self.status = tk.Label(self.status_line, text="Vision is off", bg="#10151c", fg="#cdd6e2", font=("Segoe UI", 10, "bold"), anchor="e")
        self.status.pack(side="left")
        self.readiness = tk.Label(self.status_text, text="Checking readiness...", bg="#10151c", fg="#aeb8c6", font=("Segoe UI", 8), anchor="e")
        self.readiness.pack(anchor="e")
        self.root.after(10, self.enable_hover_feedback)

        body = tk.PanedWindow(
            self.root,
            orient="horizontal",
            sashwidth=7,
            bg="#10151c",
            bd=0,
            showhandle=False,
        )
        self.body = body
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        camera_panel = tk.Frame(body, bg="black", highlightthickness=1, highlightbackground="#2d3a4a")
        self.camera_host = tk.Frame(camera_panel, bg="black")
        self.camera_host.pack(fill="both", expand=True)
        self.camera_message = tk.Label(
            self.camera_host,
            text="WAVELY Vision is off\n\nUse Start WAVELY Vision to activate the camera.",
            justify="center",
            bg="black",
            fg="#cdd6e2",
            font=("Segoe UI", 13),
        )
        self.camera_message.place(relx=0.5, rely=0.5, anchor="center")
        self.camera_host.bind("<Configure>", self.resize_embedded_vision)
        body.add(camera_panel, minsize=650, stretch="always")

        self.logs_panel = tk.Frame(body, bg="#171d25", highlightthickness=1, highlightbackground="#2d3a4a")
        self.logs_header = tk.Frame(self.logs_panel, bg="#171d25")
        self.logs_header.pack(fill="x")
        self.logs_title = tk.Label(self.logs_header, text="WAVELY LOGS", bg="#171d25", fg="#cdd6e2", anchor="w", padx=10, pady=9, font=("Segoe UI", 10, "bold")); self.logs_title.pack(side="left")
        self.logs_copy_button = tk.Button(self.logs_header, text="Copy", command=self.copy_logs, bg="#273343", fg="white", activebackground="#356da8", activeforeground="white", relief="flat", padx=8, pady=3, font=("Segoe UI", 8))
        self.logs_copy_button.pack(side="right", padx=(0, 5), pady=5)
        self.logs_clear_button = tk.Button(self.logs_header, text="Clear", command=self.clear_logs, bg="#273343", fg="white", activebackground="#356da8", activeforeground="white", relief="flat", padx=8, pady=3, font=("Segoe UI", 8))
        self.logs_clear_button.pack(side="right", padx=(0, 5), pady=5)
        self.logs = scrolledtext.ScrolledText(
            self.logs_panel,
            bg="#0b0e12",
            fg="#d7e1ec",
            insertbackground="white",
            relief="flat",
            wrap="word",
            font=("Cascadia Mono", 9),
            state="disabled",
        )
        self.logs.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.command_bar = tk.Frame(self.logs_panel, bg="#171d25", padx=8, pady=0)
        self.command_bar.pack(fill="x", pady=(0, 8))
        self.command_entry = tk.Entry(
            self.command_bar,
            bg="#0b0e12",
            fg="#f2f6fa",
            insertbackground="white",
            relief="flat",
            font=("Cascadia Mono", 9),
        )
        self.command_entry.pack(side="left", fill="x", expand=True, ipady=7)
        self.command_entry.bind("<Return>", self.run_command)
        self.command_entry.bind("<Up>", self.previous_command)
        self.command_entry.bind("<Down>", self.next_command)
        tk.Button(
            self.command_bar,
            text="Run",
            command=self.run_command,
            bg="#356da8",
            fg="white",
            activebackground="#285681",
            activeforeground="white",
            relief="flat",
            padx=14,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(7, 0))
        body.add(self.logs_panel, minsize=340)

        self.root.after(300, self.watch_process)
        self.root.after(250, self.poll_camera_aspect)
        # Run after the full interface exists; applying a theme during toolbar
        # construction leaves later panels with the old colour palette.
        self.root.after(250, lambda: self.apply_windows_theme(self.theme_is_dark))
        self.root.after(500, self.refresh_readiness)
        if not self.app_config.get("setup_complete") and not any(FACES_DIR.iterdir()):
            self.root.after(900, lambda: self.open_settings(first_run=True))

    @staticmethod
    def load_settings():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def load_json(path, fallback):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return fallback

    def save_app_config(self):
        APP_CONFIG_FILE.write_text(json.dumps(self.app_config, indent=2), encoding="utf-8")

    def write_activity(self, message):
        ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        with ACTIVITY_FILE.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message.strip()}\n")

    @staticmethod
    def available_people():
        faces_folder = FACES_DIR
        people = sorted(folder.name for folder in faces_folder.iterdir() if folder.is_dir()) if faces_folder.exists() else []
        return people

    def refresh_people(self):
        current = self.person_var.get()
        people = self.available_people()
        self.person_picker.configure(values=["Select person..."] + people)
        self.person_var.set(current if current in people else "Select person...")

    @staticmethod
    def enrolment_pose_labels():
        return [
            "Front facing neutral",
            "Turn head left",
            "Turn head right",
            "Look slightly up",
            "Look slightly down",
            "Smile",
            "Talk / change expression",
            "Move a little closer",
            "Move a little farther away",
            "Move through different angles",
        ]

    @staticmethod
    def count_enrolment_images(person):
        folder = FACES_DIR / person
        return len(list(folder.glob("video_*.jpg"))) if folder.exists() else 0

    def enrol_button_pressed(self, _event=None):
        if self.process_mode == "enrolment" and self.process and self.process.poll() is None:
            self.stop_enrolment()
            return "break"

    def centre_dialog(self, dialog):
        """Size the dialog from its controls, then centre it over Wavely."""
        self.apply_dialog_theme(dialog)
        dialog.update_idletasks()
        width = dialog.winfo_reqwidth()
        height = dialog.winfo_reqheight()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2
        dialog.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
        dialog.deiconify()
        dialog.update_idletasks()
        dialog.lift()

    def open_add_person_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Add new person")
        dialog.configure(bg="#10151c")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        tk.Label(
            dialog, text="Add new person", bg="#10151c", fg="#f2f6fa",
            font=("Segoe UI", 13, "bold"), padx=28, pady=14,
        ).pack()
        tk.Label(
            dialog, text="Enter their name. You can enrol them later from the main screen.",
            bg="#10151c", fg="#b8c3d1", font=("Segoe UI", 10), padx=28,
        ).pack()
        name_var = tk.StringVar()
        name_entry = tk.Entry(
            dialog, textvariable=name_var, width=29, bg="#171d25", fg="#f2f6fa",
            insertbackground="white", relief="flat", highlightthickness=2,
            highlightbackground="#356da8", highlightcolor="#57a6f5",
        )
        name_entry.pack(padx=28, pady=12, ipady=5)
        buttons = tk.Frame(dialog, bg="#10151c", padx=28, pady=16)
        buttons.pack(fill="x")
        tk.Button(buttons, text="Cancel", command=dialog.destroy, bg="#356da8", fg="white", relief="flat", padx=16, pady=7).pack(side="right")
        tk.Button(
            buttons, text="Confirm", bg="#19a974", fg="white", relief="flat", padx=16, pady=7,
            command=lambda: self.confirm_add_person(name_var.get(), dialog),
        ).pack(side="right", padx=(0, 8))
        name_entry.bind("<Return>", lambda _event: self.confirm_add_person(name_var.get(), dialog))
        self.centre_dialog(dialog)
        name_entry.focus_set()

    def confirm_add_person(self, person, dialog):
        person = " ".join(person.split()).strip()
        if not person:
            messagebox.showwarning("Person required", "Enter a name before confirming.", parent=dialog)
            return
        safe_name = re.sub(r"[^A-Za-z0-9 _-]", "", person).strip()
        if not safe_name:
            messagebox.showwarning("Person required", "Enter a usable name before confirming.", parent=dialog)
            return
        dialog.destroy()
        (FACES_DIR / safe_name).mkdir(parents=True, exist_ok=True)
        self.refresh_people()
        self.person_var.set(safe_name)
        self.write_log(f'Added "{safe_name}". Choose an enrolment pose from Video Enrolment.\n')

    def open_delete_person_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Delete person")
        dialog.configure(bg="#10151c")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        tk.Label(
            dialog,
            text="Delete person",
            bg="#10151c", fg="#f2f6fa", font=("Segoe UI", 13, "bold"), padx=28, pady=14,
        ).pack()
        tk.Label(
            dialog,
            text="Choose the enrolment you want to remove.",
            bg="#10151c", fg="#b8c3d1", font=("Segoe UI", 10), padx=28,
        ).pack()
        people = self.available_people()
        listbox = tk.Listbox(
            dialog, height=min(7, max(3, len(people))), width=30, bg="#171d25", fg="#f2f6fa",
            selectbackground="#356da8", selectforeground="white", relief="flat",
            highlightthickness=2, highlightbackground="#356da8", activestyle="none",
        )
        for person in people:
            listbox.insert("end", person)
        listbox.pack(padx=28, pady=12, fill="x")
        buttons = tk.Frame(dialog, bg="#10151c", padx=28, pady=16)
        buttons.pack(fill="x")
        tk.Button(buttons, text="Cancel", command=dialog.destroy, bg="#356da8", fg="white", relief="flat", padx=16, pady=7).pack(side="right")
        tk.Button(
            buttons, text="Confirm", bg="#a93d3e", fg="white", relief="flat", padx=16, pady=7,
            command=lambda: self.show_delete_confirmation(dialog, listbox),
        ).pack(side="right", padx=(0, 8))
        self.centre_dialog(dialog)

    def show_delete_confirmation(self, dialog, listbox):
        selected = listbox.curselection()
        if not selected:
            messagebox.showwarning("Person required", "Choose a person to delete.", parent=dialog)
            return
        person = listbox.get(selected[0]).split("   —", 1)[0].strip()
        for child in dialog.winfo_children():
            child.destroy()
        tk.Label(
            dialog, text=f'Delete "{person}"?', bg="#10151c", fg="#f2f6fa",
            font=("Segoe UI", 13, "bold"), padx=28, pady=14,
        ).pack()
        tk.Label(
            dialog, text='Type confirm to permanently remove their enrolment.',
            bg="#10151c", fg="#b8c3d1", font=("Segoe UI", 10), padx=28,
        ).pack()
        confirm_var = tk.StringVar()
        confirm_entry = tk.Entry(
            dialog, textvariable=confirm_var, width=29, bg="#171d25", fg="#f2f6fa",
            insertbackground="white", relief="flat", highlightthickness=2,
            highlightbackground="#356da8", highlightcolor="#57a6f5",
        )
        confirm_entry.pack(padx=28, pady=12, ipady=5)
        buttons = tk.Frame(dialog, bg="#10151c", padx=28, pady=16)
        buttons.pack(fill="x")
        tk.Button(buttons, text="Cancel", command=dialog.destroy, bg="#356da8", fg="white", relief="flat", padx=16, pady=7).pack(side="right")
        tk.Button(
            buttons, text="Delete", bg="#a93d3e", fg="white", relief="flat", padx=16, pady=7,
            command=lambda: self.delete_from_dialog(person, confirm_var.get(), dialog),
        ).pack(side="right", padx=(0, 8))
        confirm_entry.bind("<Return>", lambda _event: self.delete_from_dialog(person, confirm_var.get(), dialog))
        self.centre_dialog(dialog)
        confirm_entry.focus_set()

    def delete_from_dialog(self, person, confirmation, dialog):
        if confirmation.casefold().strip() != "confirm":
            messagebox.showwarning("Confirmation required", 'Type confirm to delete this person.', parent=dialog)
            return
        dialog.destroy()
        self.pending_deletion = person
        self.complete_deletion("confirm")

    def save_layout_later(self, _event=None):
        if not self.layout_ready:
            return
        if hasattr(self, "save_job"):
            self.root.after_cancel(self.save_job)
        self.save_job = self.root.after(400, self.save_layout)

    def enable_layout_saving(self):
        self.layout_ready = True

    def save_layout(self):
        maximized = self.root.state() == "zoomed"
        saved_geometry = self.settings.get("geometry", "1520x850+120+80")
        if not maximized:
            saved_geometry = (
                f"{self.root.winfo_width()}x{self.root.winfo_height()}"
                f"+{self.root.winfo_x()}+{self.root.winfo_y()}"
            )

        self.settings = {
            "geometry": saved_geometry,
            "maximized": maximized,
            "command_history": self.command_history[-50:],
            "theme": "dark" if self.theme_is_dark else "light",
        }
        SETTINGS_FILE.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")

    def write_log(self, line):
        if not self.root.winfo_exists():
            return
        self.logs.configure(state="normal")
        self.logs.insert("end", line)
        self.logs.see("end")
        self.logs.configure(state="disabled")
        clean = " ".join(line.split())
        if clean and ("TRIGGERED:" in clean or "ERROR" in clean or "enrolment" in clean.casefold()):
            self.write_activity(clean)

    def copy_logs(self):
        """Copy the visible dashboard log without exposing private webhook data."""
        content = self.logs.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.write_log("Visible logs copied to the clipboard.\n")

    def clear_logs(self):
        """Clear only the dashboard view; the timestamped activity file is kept."""
        self.logs.configure(state="normal")
        self.logs.delete("1.0", "end")
        self.logs.configure(state="disabled")
        self.write_log("Dashboard log cleared. Activity history is still available from Tools.\n")

    def refresh_readiness(self):
        """Show a compact, actionable summary of the local WAVELY setup."""
        camera_name = self.app_config.get("camera_name") or "No camera selected"
        people = self.available_people()
        model_ready = (CONFIG_DIR / "face_model.yml").is_file() and (CONFIG_DIR / "face_labels.json").is_file()
        home = self.load_json(ACTION_CONFIG_FILE, {"webhook_url": "", "actions": []})
        webhook_ready = bool(str(home.get("webhook_url", "")).strip())
        if not self.app_config.get("setup_complete"):
            summary = "Setup required"
        elif not camera_name or camera_name == "No camera selected":
            summary = "Choose a camera"
        elif people and model_ready and webhook_ready:
            summary = f"Ready · {camera_name} · {len(people)} enrolled"
        elif not people or not model_ready:
            summary = f"Camera ready · {camera_name} · Face model not trained"
        elif not webhook_ready:
            summary = f"Camera ready · {camera_name} · Home Assistant not configured"
        else:
            summary = f"Camera ready · {camera_name}"
        if hasattr(self, "readiness") and self.readiness.winfo_exists():
            self.readiness.configure(text=summary)
            self.root.after(2500, self.refresh_readiness)

    def open_health_check(self):
        dialog = self.dark_dialog("WAVELY health check")
        tk.Label(dialog, text="WAVELY health check", bg="#10151c", fg="#f2f6fa", font=("Segoe UI", 14, "bold"), padx=24, pady=14).pack(anchor="w")
        tk.Label(dialog, text="A quick local check of the things WAVELY needs before you start Vision.", bg="#10151c", fg="#aeb8c6", wraplength=560, justify="left", padx=24).pack(anchor="w")
        checks = tk.Frame(dialog, bg="#10151c", padx=24, pady=14)
        checks.pack(fill="both", expand=True)
        home = self.load_json(ACTION_CONFIG_FILE, {"webhook_url": "", "actions": []})
        camera = self.app_config.get("camera_name") or ""
        model = (CONFIG_DIR / "face_model.yml").is_file() and (CONFIG_DIR / "face_labels.json").is_file()
        rows = [
            (bool(self.app_config.get("setup_complete")), "required", "Initial setup is complete", "Open Settings and choose a camera"),
            (bool(camera), "required", f"Camera selected: {camera or 'none'}", "Open Settings and refresh devices"),
            (bool(self.available_people()), "optional", f"Enrolled people: {len(self.available_people())}", "Use Video Enrolment to add a person"),
            (model, "optional", "Face model is available", "Train the face model from People manager"),
            (bool(str(home.get("webhook_url", "")).strip()), "optional", "Home Assistant is configured", "Configure it from Tools if you use Home Assistant"),
            (VISION_PROGRAM.exists() or Path(sys.executable).exists(), "required", "Vision runtime is available", "Reinstall WAVELY if the runtime is missing"),
        ]
        for passed, importance, label, action in rows:
            colour = "#19a974" if passed else ("#f2c14e" if importance == "required" else "#8f9baa")
            suffix = "" if passed else (" · optional" if importance == "optional" else " · needs attention")
            tk.Label(checks, text=("✓" if passed else "!") + "  " + label + suffix, bg="#10151c", fg=colour, font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x", pady=3)
            if not passed:
                tk.Label(checks, text="     " + action, bg="#10151c", fg="#aeb8c6", font=("Segoe UI", 9), anchor="w").pack(fill="x")
        buttons = tk.Frame(dialog, bg="#10151c", padx=24, pady=16)
        buttons.pack(fill="x")
        tk.Button(buttons, text="Settings", command=lambda: (dialog.destroy(), self.open_settings()), bg="#356da8", fg="white", relief="flat", padx=14, pady=7).pack(side="left")
        tk.Button(buttons, text="Close", command=dialog.destroy, bg="#273343", fg="white", relief="flat", padx=14, pady=7).pack(side="right")
        self.centre_dialog(dialog)

    def set_status(self, text, colour="#aeb8c6"):
        self.status.configure(text=text, fg="#f2f6fa" if self.theme_is_dark else "#17212b")
        self.status_dot.itemconfigure(self.status_circle, fill=colour)

    def toggle_theme(self):
        """Switch the dashboard theme immediately and remember the choice."""
        self.theme_is_dark = not self.theme_is_dark
        self.apply_windows_theme(self.theme_is_dark)
        for window in self.root.winfo_children():
            if isinstance(window, tk.Toplevel) and window.winfo_exists():
                self.apply_dialog_theme(window, self.theme_is_dark)
        self.save_layout()

    @staticmethod
    def configure_vision_button_styles(style):
        for name, color, hover in (
            ("Vision.Start.TButton", "#19a974", "#14865d"),
            ("Vision.Stop.TButton", "#cc4b4c", "#a93d3e"),
        ):
            style.configure(
                name, background=color, foreground="white",
                font=("Segoe UI", 11, "bold"), padding=(18, 22),
                borderwidth=0, relief="flat",
            )
            style.map(
                name,
                background=[("disabled", color), ("pressed", hover), ("active", hover)],
                foreground=[("disabled", "white"), ("active", "white")],
            )

    def apply_windows_theme(self, dark=None):
        """Apply a predictable Windows light or dark palette to the live interface."""
        dark = self.theme_is_dark if dark is None else dark
        if dark:
            app_bg, panel_bg, text_fg = "#10151c", "#171d25", "#f2f6fa"
            log_bg, log_fg, border = "#0b0e12", "#d7e1ec", "#2d3a4a"
            combo_field, combo_arrow = "#171d25", "#f2f6fa"
        else:
            app_bg, panel_bg, text_fg = "#f3f5f7", "#eef1f4", "#17212b"
            log_bg, log_fg, border = "#ffffff", "#17212b", "#b6c2d0"
            combo_field, combo_arrow = "#ffffff", "#17212b"

        # Set every persistent part of the main window explicitly.  This avoids
        # Tk's colour aliases retaining a colour from a previous theme.
        self.root.configure(bg=app_bg)
        self.toolbar.configure(bg=app_bg)
        self.enrol_group.configure(bg=app_bg)
        self.body.configure(bg=app_bg)
        self.status_box.configure(bg=app_bg)
        self.status_text.configure(bg=app_bg)
        self.status_line.configure(bg=app_bg)
        self.status_dot.configure(bg=app_bg)
        self.status.configure(bg=app_bg, fg=text_fg)
        self.readiness.configure(bg=app_bg, fg="#aeb8c6" if dark else "#526272")
        self.logs_panel.configure(bg=panel_bg, highlightbackground=border)
        self.logs_header.configure(bg=panel_bg)
        self.logs_title.configure(bg=panel_bg, fg=text_fg)
        self.logs_copy_button.configure(bg="#273343", fg="white")
        self.logs_clear_button.configure(bg="#273343", fg="white")
        self.command_bar.configure(bg=panel_bg)
        self.logs.configure(bg=log_bg, fg=log_fg, insertbackground=log_fg,
                            selectbackground="#356da8", selectforeground="white")
        command_bg = "#0b0e12" if dark else "#f4f6f8"
        self.command_entry.configure(bg=command_bg, fg=log_fg, insertbackground=log_fg,
                                     highlightthickness=1, highlightbackground=border,
                                     highlightcolor="#356da8")
        try:
            self.logs.vbar.update_idletasks()

            scrollbar_hwnd = self.logs.vbar.winfo_id()

            if dark:
                scrollbar_theme = "DarkMode_Explorer"
            else:
                scrollbar_theme = "Explorer"

            ctypes.windll.uxtheme.SetWindowTheme(
                scrollbar_hwnd,
                scrollbar_theme,
                None
            )

            ctypes.windll.user32.RedrawWindow(
                scrollbar_hwnd,
                None,
                None,
                0x0085
            )

        except (AttributeError, OSError, tk.TclError):
            pass

        style = ttk.Style(self.root)
        # clam honours colour settings on all supported Windows versions, unlike
        # the Windows system combobox style which keeps a stale field colour.
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure_vision_button_styles(style)
        self.theme_button.configure(
            text="☀" if dark else "☾",
            bg="#273343" if dark else "#526272",
            activebackground="#356da8",
            fg="white",
        )
        self.theme_tooltip.text = "Switch to light mode" if dark else "Switch to dark mode"
        style.configure("Wavely.TCombobox", fieldbackground=combo_field,
                        background="#356da8", foreground=combo_arrow,
                        arrowcolor=combo_arrow, bordercolor="#356da8",
                        lightcolor="#356da8", darkcolor="#356da8")
        style.map("Wavely.TCombobox",
                  fieldbackground=[("readonly", combo_field)],
                  foreground=[("readonly", combo_arrow)],
                  selectbackground=[("readonly", combo_field)],
                  selectforeground=[("readonly", combo_arrow)])
        self.person_picker.configure(style="Wavely.TCombobox")

        # Menus and their pop-outs are not ttk controls, so colour them directly.
        for menu in (self.enrol_menu, self.tools_menu):
            menu.configure(bg=panel_bg, fg=text_fg, activebackground="#356da8",
                           activeforeground="white")
        self.apply_windows_backdrop(self.root, dark)
        self.apply_windows_titlebar(self.root, dark)

    def apply_dialog_theme(self, dialog, dark=None):
        """Theme a transient window built with the dashboard's standard controls."""
        dark = self.theme_is_dark if dark is None else dark
        if dark:
            app_bg, panel_bg, text_fg, input_bg = "#10151c", "#171d25", "#f2f6fa", "#0b0e12"
        else:
            app_bg, panel_bg, text_fg, input_bg = "#f3f5f7", "#ffffff", "#17212b", "#ffffff"
        dark_colours = {"#10151c", "#171d25", "#0b0e12", "#273343", "#2d3a4a"}
        light_text = {"#f2f6fa", "#cdd6e2", "#aeb8c6", "#b8c3d1", "#d7e1ec"}
        dialog.configure(bg=app_bg)
        def visit(widget):
            try:
                if isinstance(widget, (tk.Frame, tk.LabelFrame)):
                    widget.configure(bg=app_bg)
                elif isinstance(widget, (tk.Entry, tk.Text, tk.Spinbox)):
                    widget.configure(background=input_bg, foreground=text_fg, insertbackground=text_fg)
                elif isinstance(widget, tk.Listbox):
                    widget.configure(background=input_bg, foreground=text_fg,
                                     selectbackground="#356da8", selectforeground="white")
                elif isinstance(widget, tk.Scrollbar):
                    widget.configure(bg="#273343" if dark else "#b6c2d0",
                                     troughcolor="#0b0e12" if dark else "#ffffff",
                                     activebackground="#356da8")
                elif isinstance(widget, (tk.Button, tk.Menubutton)):
                    # Preserve the coloured action buttons.  A neutral dark button
                    # becomes visible slate in light mode instead of white-on-white.
                    current_bg = widget.cget("bg")
                    if current_bg == "#273343":
                        widget.configure(bg="#526272" if not dark else "#273343", fg="white",
                                         activebackground="#356da8", activeforeground="white")
                elif isinstance(widget, tk.Checkbutton):
                    widget.configure(bg=app_bg, fg=text_fg, selectcolor=input_bg,
                                     activebackground=app_bg, activeforeground=text_fg)
                elif isinstance(widget, tk.Scale):
                    widget.configure(bg=app_bg, fg=text_fg,
                                     troughcolor="#273343" if dark else "#dce1e6",
                                     activebackground="#356da8")
                elif isinstance(widget, tk.Label):
                    widget.configure(bg=app_bg, fg=text_fg)
                else:
                    bg = widget.cget("bg")
                    if not dark and bg in dark_colours:
                        widget.configure(bg=app_bg)
                    fg = widget.cget("fg")
                    if not dark and fg in light_text:
                        widget.configure(fg=text_fg)
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                visit(child)
        visit(dialog)
        self.apply_windows_backdrop(dialog, dark)

    @staticmethod
    def apply_windows_backdrop(window, dark):
        """Request the Windows 11 Mica backdrop; older Windows safely ignore it."""
        try:
            window.update_idletasks()
            value = ctypes.c_int(2)  # DWMSBT_MAINWINDOW (Mica)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(window.winfo_id(), 38, ctypes.byref(value), ctypes.sizeof(value))
            mode = ctypes.c_int(1 if dark else 0)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(window.winfo_id(), 20, ctypes.byref(mode), ctypes.sizeof(mode))
        except (AttributeError, OSError):
            pass

    @staticmethod
    def apply_windows_titlebar(window, dark):
        """Request a native light or dark Windows title bar when supported."""
        try:
            window.update_idletasks()
            hwnd = window.winfo_id()
            value = ctypes.c_int(1 if dark else 0)
            dwmapi = ctypes.windll.dwmapi
            # Windows 10 20H1+ uses attribute 20. Older builds used 19.
            result = dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
            if result != 0:
                dwmapi.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(value), ctypes.sizeof(value))
        except (AttributeError, OSError, tk.TclError):
            pass

    def enable_hover_feedback(self):
        def visit(widget):
            if isinstance(widget, (tk.Button, tk.Menubutton)):
                normal = widget.cget("bg")
                hover = "#4b86c5" if normal not in {"#19a974", "#a93d3e", "#cc4b4c"} else ({"#19a974": "#33c58c", "#a93d3e": "#cf5657", "#cc4b4c": "#df6464"}[normal])
                widget.bind("<Enter>", lambda _event, w=widget, c=hover: w.configure(bg=c, relief="raised"), add="+")
                widget.bind("<Leave>", lambda _event, w=widget, c=normal: w.configure(bg=c, relief="flat"), add="+")
            for child in widget.winfo_children():
                visit(child)
        visit(self.root)

    def dark_dialog(self, title):
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(title)
        dialog.configure(bg="#10151c" if self.theme_is_dark else "#f3f5f7")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.apply_windows_backdrop(dialog, self.theme_is_dark)
        return dialog

    def open_settings(self, first_run=False):
        dialog = self.dark_dialog("WAVELY first-run setup" if first_run else "WAVELY settings")
        dialog.resizable(True, False)
        tk.Label(dialog, text="Set up your camera" if first_run else "Settings and camera", bg="#10151c", fg="#f2f6fa", font=("Segoe UI", 14, "bold"), padx=24, pady=14).pack()
        description = "Connect a webcam, detect it below, then preview it in Gesture Test Mode. You can change these settings later from Tools." if first_run else "These settings apply the next time Vision or Video Enrolment starts."
        tk.Label(dialog, text=description, bg="#10151c", fg="#aeb8c6", wraplength=500, justify="left").pack(anchor="w", padx=28)
        form = tk.Frame(dialog, bg="#10151c", padx=28, pady=8)
        form.pack(fill="both", expand=True)
        camera_devices = {}
        camera_var = tk.StringVar(value=self.app_config.get("camera_name", ""))
        resolution_var = tk.StringVar(value=self.app_config.get("resolution", "1280x720"))
        mirror_var = tk.BooleanVar(value=bool(self.app_config.get("mirror", True)))
        startup_var = tk.BooleanVar(value=bool(self.app_config.get("start_with_windows", False)))
        sensitivity_var = tk.DoubleVar(value=float(self.app_config.get("gesture_sensitivity", 0.60)))
        audio_input_devices = {item["name"]: item for item in enumerate_audio_inputs()}
        audio_output_devices = {item["name"]: item for item in enumerate_audio_outputs()}

        def section(title, description, control):
            tk.Label(form, text=title, bg="#10151c", fg="#f2f6fa", font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x", pady=(2, 0))
            tk.Label(form, text=description, bg="#10151c", fg="#aeb8c6", wraplength=490, justify="left", anchor="w").pack(fill="x", pady=(0, 2))
            control.pack(anchor="w", pady=(0, 4))

        camera = ttk.Combobox(form, textvariable=camera_var, values=[], state="readonly", style="Wavely.TCombobox", width=47)
        section("Camera", "Choose the Windows camera to use. Wavely keeps its OpenCV index internally.", camera)
        detect_status = tk.StringVar(value="Detecting connected cameras...")
        detect_button = tk.Button(form, text="Refresh devices", bg="#356da8", fg="white", relief="flat", padx=12, pady=6)
        detect_button.configure(command=lambda: self.detect_cameras(camera_var, camera, camera_devices, detect_status, detect_button))
        detect_button.pack(anchor="w")
        tk.Label(form, textvariable=detect_status, bg="#10151c", fg="#aeb8c6", wraplength=490, justify="left").pack(anchor="w", pady=(3, 10))
        audio_input_var = tk.StringVar(value=self.app_config.get("audio_input_name", ""))
        audio_input = ttk.Combobox(form, textvariable=audio_input_var, values=list(audio_input_devices), state="readonly", style="Wavely.TCombobox", width=47)
        section("Audio input", "Reserved for future microphone features. Wavely currently does not use microphone input.", audio_input)
        audio_output_var = tk.StringVar(value=self.app_config.get("audio_output_name", ""))
        audio_output = ttk.Combobox(form, textvariable=audio_output_var, values=list(audio_output_devices), state="readonly", style="Wavely.TCombobox", width=47)
        section("Audio output", "Speech uses this Windows SAPI output when the selected device is available.", audio_output)
        audio_refresh_button = tk.Button(form, text="Refresh audio devices", bg="#356da8", fg="white", relief="flat", padx=12, pady=6)
        audio_refresh_button.configure(command=lambda: self.refresh_audio_devices(audio_input, audio_input_devices, audio_output, audio_output_devices, detect_status))
        audio_refresh_button.pack(anchor="w", pady=(0, 8))
        resolution = ttk.Combobox(form, textvariable=resolution_var, values=["640x480", "1280x720", "1920x1080"], state="readonly", style="Wavely.TCombobox", width=17)
        section("Resolution", "Higher detail can improve detection, but uses more processing power.", resolution)
        mirror = tk.Checkbutton(form, text="Mirror camera image", variable=mirror_var, bg="#10151c", fg="#f2f6fa", selectcolor="#171d25", activebackground="#10151c", activeforeground="white")
        section("Camera mirror", "Makes the live image move like a normal mirror.", mirror)
        startup = tk.Checkbutton(form, text="Start WAVELY with Windows", variable=startup_var, bg="#10151c", fg="#f2f6fa", selectcolor="#171d25", activebackground="#10151c", activeforeground="white")
        section("Start with Windows", "Opens WAVELY automatically after you sign in.", startup)
        sensitivity = tk.Scale(form, variable=sensitivity_var, from_=0.40, to=0.90, resolution=0.05, orient="horizontal", length=260, bg="#10151c", fg="#f2f6fa", highlightthickness=0, troughcolor="#273343", activebackground="#356da8")
        section("Gesture sensitivity", "Lower reacts sooner. Higher needs a steadier held gesture.", sensitivity)

        actions = tk.Frame(dialog, bg="#10151c", padx=28, pady=8); actions.pack(fill="x")
        tk.Button(actions, text="Preview / test camera", command=lambda: self.save_settings_and_test(dialog, camera_var, camera_devices, audio_input_var, audio_input_devices, audio_output_var, audio_output_devices, resolution_var, mirror_var, sensitivity_var, startup_var), bg="#356da8", fg="white", relief="flat", padx=12, pady=7).pack(side="left")
        tk.Button(actions, text="Save", command=lambda: self.save_settings(dialog, camera_var, camera_devices, audio_input_var, audio_input_devices, audio_output_var, audio_output_devices, resolution_var, mirror_var, sensitivity_var, startup_var), bg="#19a974", fg="white", relief="flat", padx=18, pady=7).pack(side="right")
        tk.Button(actions, text="Cancel", command=dialog.destroy, bg="#273343", fg="white", relief="flat", padx=18, pady=7).pack(side="right", padx=(0, 8))
        self.centre_dialog(dialog)
        self.root.after(0, lambda: self.detect_cameras(camera_var, camera, camera_devices, detect_status, detect_button))

    def save_settings(self, dialog, camera, camera_devices, audio_input, audio_input_devices, audio_output, audio_output_devices, resolution, mirror, sensitivity, startup):
        selected_camera = camera_devices.get(camera.get())
        if selected_camera is None:
            messagebox.showwarning("Camera required", "Choose a detected camera, then use Refresh devices if needed.", parent=dialog)
            return False
        selected_input = audio_input_devices.get(audio_input.get(), {})
        selected_output = audio_output_devices.get(audio_output.get(), {})
        self.app_config.update({
            "camera_index": selected_camera["index"],
            "camera_name": selected_camera["name"],
            "camera_device_id": selected_camera.get("device_id", ""),
            "audio_input_device_id": selected_input.get("device_id", -1),
            "audio_input_name": audio_input.get(),
            "audio_output_device_id": selected_output.get("device_id", ""),
            "audio_output_name": audio_output.get(),
            "resolution": resolution.get(), "mirror": mirror.get(),
            "gesture_sensitivity": sensitivity.get(), "start_with_windows": startup.get(),
            "setup_complete": True,
        })
        self.save_app_config(); self.set_windows_startup(startup.get()); dialog.destroy()
        self.write_log("Settings saved. They apply next time Vision starts.\n")
        return True

    def save_settings_and_test(self, dialog, *values):
        if self.save_settings(dialog, *values):
            self.start_test_mode()

    @staticmethod
    def refresh_audio_devices(audio_input, audio_input_devices, audio_output, audio_output_devices, status_var):
        audio_input_devices.clear()
        audio_input_devices.update({item["name"]: item for item in enumerate_audio_inputs()})
        audio_output_devices.clear()
        audio_output_devices.update({item["name"]: item for item in enumerate_audio_outputs()})
        audio_input.configure(values=list(audio_input_devices))
        audio_output.configure(values=list(audio_output_devices))
        status_var.set("Audio devices refreshed.")

    def export_settings(self):
        destination = filedialog.asksaveasfilename(
            parent=self.root, title="Export WAVELY settings", defaultextension=".json",
            initialfile="WAVELY-settings.json", filetypes=[("WAVELY settings", "*.json")],
        )
        if not destination:
            return
        try:
            home = self.load_json(ACTION_CONFIG_FILE, {"webhook_url": "", "actions": []})
            write_json(destination, make_bundle(self.app_config, home))
        except (KeyError, OSError, ValueError) as error:
            messagebox.showerror("Export failed", str(error), parent=self.root)
            return
        messagebox.showinfo("Settings exported", "Your settings were saved. The file includes your Home Assistant webhook address if configured; keep it private. Face images and logs are not included.", parent=self.root)

    def import_settings(self):
        source = filedialog.askopenfilename(
            parent=self.root, title="Import WAVELY settings", filetypes=[("WAVELY settings", "*.json")],
        )
        if not source:
            return
        try:
            bundle = load_bundle(source)
        except ValueError as error:
            messagebox.showerror("Import failed", str(error), parent=self.root)
            return
        if not messagebox.askyesno("Import settings", f"Replace camera preferences and {len(bundle['home_assistant']['actions'])} Home Assistant actions? Your current settings will be backed up first.", parent=self.root):
            return
        try:
            backup_dir = apply_bundle(bundle, CONFIG_DIR, DATA_ROOT)
            self.app_config = self.load_json(APP_CONFIG_FILE, {})
            self.set_windows_startup(self.app_config.get("start_with_windows", False))
        except OSError as error:
            messagebox.showerror("Import failed", f"{error}\nYour previous settings are in the WAVELY data backups folder.", parent=self.root)
            return
        self.write_log(f"Settings imported. Previous settings saved in {backup_dir}. Restart Vision to apply them.\n")
        messagebox.showinfo("Settings imported", "Review the camera for this PC. Restart Vision to apply the imported settings.", parent=self.root)
        self.open_settings()

    def detect_cameras(self, camera_var, camera_picker, camera_devices, status_var, button):
        if self.process and self.process.poll() is None:
            status_var.set("Stop Vision or enrolment before detecting cameras.")
            return
        if not ENROLMENT_PROGRAM.exists():
            status_var.set("Camera detection is unavailable: the enrolment runtime is missing.")
            return
        button.configure(state="disabled")
        status_var.set("Checking connected cameras...")

        def worker():
            try:
                result = subprocess.run([str(ENROLMENT_PROGRAM), "--list-cameras"], cwd=ROOT, capture_output=True, text=True, timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
                data = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else {}
                indices = data.get("cameras", [])
                windows_cameras = enumerate_windows_cameras()
                cameras = {}
                for position, index in enumerate(indices):
                    device = windows_cameras[position] if position < len(windows_cameras) else {}
                    name = device.get("name") or "Detected camera"
                    display_name = name if name not in cameras else f"{name} ({position + 1})"
                    cameras[display_name] = {"index": index, "name": name, "device_id": device.get("device_id", "")}
                message = f"Found {len(cameras)} camera device(s)." if cameras else "No working camera found. Check its connection and Windows camera permissions."
            except (OSError, ValueError, IndexError, json.JSONDecodeError, subprocess.TimeoutExpired):
                cameras = {}
                message = "Camera check failed. Check the webcam connection and try again."

            def finish():
                button.configure(state="normal")
                status_var.set(message)
                camera_devices.clear()
                camera_devices.update(cameras)
                camera_picker.configure(values=list(cameras))
                selected_name = self.app_config.get("camera_name", "")
                selected_id = self.app_config.get("camera_device_id", "")
                selected = next((name for name, item in cameras.items() if item["device_id"] == selected_id), None)
                if selected is None and selected_name in cameras:
                    selected = selected_name
                if selected is None and cameras:
                    selected = next(iter(cameras))
                    if selected_id or selected_name:
                        status_var.set(f"Previous camera was not found. Using {selected}.")
                camera_var.set(selected or "No camera detected")

            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def set_windows_startup(self, enabled):
        startup = Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs/Startup/WAVELY Vision.cmd"
        if enabled:
            startup.parent.mkdir(parents=True, exist_ok=True); startup.write_text(f'@start "" "{ROOT / "Wavely.exe"}"\r\n', encoding="utf-8")
        else:
            startup.unlink(missing_ok=True)

    def open_people_manager(self):
        dialog = self.dark_dialog("People manager")
        tk.Label(dialog, text="People manager", bg="#10151c", fg="#f2f6fa", font=("Segoe UI", 14, "bold"), padx=24, pady=14).pack()
        tk.Label(dialog, text="Review saved face images. Train the model after adding or removing enrolments.", bg="#10151c", fg="#aeb8c6", wraplength=400, justify="left").pack(anchor="w", padx=24)
        listbox = tk.Listbox(dialog, width=47, height=9, bg="#171d25", fg="#f2f6fa", selectbackground="#356da8", activestyle="none", highlightthickness=1, highlightbackground="#356da8")
        for person in self.available_people():
            folder = FACES_DIR / person
            count = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
            listbox.insert("end", f"{person}   —   {count} face images")
        listbox.pack(padx=24, pady=8)
        buttons = tk.Frame(dialog, bg="#10151c", padx=24, pady=16); buttons.pack(fill="x")
        tk.Button(buttons, text="Train face model", command=lambda: (dialog.destroy(), self.start_face_training()), bg="#19a974", fg="white", relief="flat", padx=12, pady=7).pack(side="left")
        tk.Button(buttons, text="Add person", command=lambda: (dialog.destroy(), self.open_add_person_dialog()), bg="#356da8", fg="white", relief="flat", padx=12, pady=7).pack(side="left", padx=(8, 0))
        tk.Button(buttons, text="Delete selected", command=lambda: self.show_delete_confirmation(dialog, listbox), bg="#a93d3e", fg="white", relief="flat", padx=12, pady=7).pack(side="left", padx=(8, 0))
        tk.Button(buttons, text="Close", command=dialog.destroy, bg="#273343", fg="white", relief="flat", padx=14, pady=7).pack(side="right")
        self.centre_dialog(dialog)

    def open_home_actions(self):
        dialog = self.dark_dialog("Home Assistant actions")
        data = self.load_json(ACTION_CONFIG_FILE, {"webhook_url": "", "actions": []})
        actions = [dict(item) for item in data.get("actions", []) if isinstance(item, dict)]
        tk.Label(dialog, text="Home Assistant actions", bg="#10151c", fg="#f2f6fa", font=("Segoe UI", 14, "bold"), padx=24, pady=14).pack()
        tk.Label(dialog, text="Choose a gesture and the Home Assistant action it should run. Test Mode never sends webhooks.", bg="#10151c", fg="#aeb8c6", wraplength=620, justify="left").pack(anchor="w", padx=24)
        form = tk.Frame(dialog, bg="#10151c", padx=24, pady=10)
        form.pack(fill="both", expand=True)
        url = tk.StringVar(value=data.get("webhook_url", ""))
        tk.Label(form, text="Webhook URL", bg="#10151c", fg="#f2f6fa").pack(anchor="w")
        tk.Entry(form, textvariable=url, width=72, bg="#171d25", fg="#f2f6fa", insertbackground="white").pack(fill="x", pady=(3, 10))
        listbox = tk.Listbox(form, width=76, height=8, bg="#171d25", fg="#f2f6fa", selectbackground="#356da8", activestyle="none")
        listbox.pack(fill="x", pady=(0, 10))

        gesture_var = tk.StringVar()
        id_var = tk.StringVar()
        label_var = tk.StringVar()
        entity_var = tk.StringVar()
        fields = (
            ("Gesture", gesture_var), ("Action ID", id_var),
            ("Label", label_var), ("Entity ID (optional)", entity_var),
        )
        for title, variable in fields:
            row = tk.Frame(form, bg="#10151c")
            row.pack(fill="x", pady=3)
            tk.Label(row, text=title, width=20, anchor="w", bg="#10151c", fg="#f2f6fa").pack(side="left")
            if title == "Gesture":
                control = ttk.Combobox(row, textvariable=variable, values=GESTURES, state="readonly", width=45, style="Wavely.TCombobox")
            else:
                control = tk.Entry(row, textvariable=variable, width=48, bg="#171d25", fg="#f2f6fa", insertbackground="white")
            control.pack(side="left", fill="x", expand=True)

        def refresh():
            listbox.delete(0, "end")
            for item in actions:
                listbox.insert("end", f"{item.get('gesture') or 'Unassigned'}  →  {item.get('id', '')}  |  {item.get('label', '')}  |  {item.get('entity_id', '')}")

        def selected_index():
            selection = listbox.curselection()
            return selection[0] if selection else None

        def select_action(_event=None):
            index = selected_index()
            if index is None:
                return
            item = actions[index]
            for key, variable in (("gesture", gesture_var), ("id", id_var), ("label", label_var), ("entity_id", entity_var)):
                variable.set(item.get(key, ""))

        def clear_fields():
            listbox.selection_clear(0, "end")
            for _, variable in fields:
                variable.set("")

        def read_fields():
            gesture = gesture_var.get().strip()
            action_id = id_var.get().strip()
            entity_id = entity_var.get().strip()
            if gesture not in GESTURES:
                messagebox.showwarning("Choose a gesture", "Select a gesture from the list.", parent=dialog)
                return None
            if not re.fullmatch(r"[a-z0-9_]+", action_id):
                messagebox.showwarning("Action ID", "Use lowercase letters, numbers and underscores.", parent=dialog)
                return None
            if entity_id and not re.fullmatch(r"[a-z_]+\.[a-z0-9_]+", entity_id):
                messagebox.showwarning("Entity ID", "Use an ID such as light.living_room.", parent=dialog)
                return None
            return {"gesture": gesture, "id": action_id, "label": label_var.get().strip() or action_id, "entity_id": entity_id}

        def add_or_update():
            item = read_fields()
            if item is None:
                return
            index = selected_index()
            if any(existing.get("gesture") == item["gesture"] for position, existing in enumerate(actions) if position != index):
                messagebox.showwarning("Gesture already used", "Each gesture can run one action. Edit its existing row instead.", parent=dialog)
                return
            if index is None:
                actions.append(item)
                index = len(actions) - 1
            else:
                actions[index] = {**actions[index], **item}
            refresh()
            listbox.selection_set(index)

        def remove_selected():
            index = selected_index()
            if index is not None:
                actions.pop(index)
                refresh()
                clear_fields()

        def preview():
            item = read_fields()
            if item is None:
                return
            preview_actions = list(actions)
            index = selected_index()
            if index is None:
                preview_actions.append(item)
            else:
                preview_actions[index] = item
            payload = payload_for_gesture("Example person", "Right", item["gesture"], preview_actions)
            messagebox.showinfo("Webhook preview", json.dumps(payload, indent=2), parent=dialog)

        def check_address():
            webhook = url.get().strip()
            parsed = urlparse(webhook)
            if not webhook or parsed.scheme not in ("http", "https") or not parsed.netloc:
                messagebox.showwarning("Webhook address", "Enter a complete http:// or https:// address first.", parent=dialog)
                return
            try:
                request = urllib.request.Request(webhook, method="HEAD")
                with urllib.request.urlopen(request, timeout=5) as response:
                    status = response.status
                messagebox.showinfo("Address reachable", f"Home Assistant responded successfully (HTTP {status}). No action was sent.", parent=dialog)
            except urllib.error.HTTPError as error:
                if error.code in (401, 403, 405):
                    messagebox.showinfo("Address reachable", f"The address responded (HTTP {error.code}). No action was sent.", parent=dialog)
                else:
                    messagebox.showwarning("Address check failed", f"The address responded with HTTP {error.code}. No action was sent.", parent=dialog)
            except (OSError, ValueError) as error:
                messagebox.showwarning("Address check failed", f"WAVELY could not reach the configured address.\n\n{error}", parent=dialog)

        def save_actions():
            webhook = url.get().strip()
            parsed = urlparse(webhook)
            if webhook and (parsed.scheme not in ("http", "https") or not parsed.netloc):
                messagebox.showwarning("Webhook URL", "Enter a full http:// or https:// URL.", parent=dialog)
                return
            try:
                ACTION_CONFIG_FILE.write_text(json.dumps({"webhook_url": webhook, "actions": actions}, indent=2), encoding="utf-8")
            except OSError as error:
                messagebox.showerror("Could not save", str(error), parent=dialog)
                return
            dialog.destroy()
            self.write_log("Home Assistant actions saved. Restart Vision to use the new mapping.\n")

        listbox.bind("<<ListboxSelect>>", select_action)
        refresh()
        row = tk.Frame(dialog, bg="#10151c", padx=24, pady=16)
        row.pack(fill="x")
        for title, command in (("New", clear_fields), ("Add / update", add_or_update), ("Remove", remove_selected), ("Preview", preview), ("Check address", check_address)):
            tk.Button(row, text=title, command=command, bg="#356da8", fg="white", relief="flat", padx=12, pady=7).pack(side="left", padx=(0, 7))
        tk.Button(row, text="Save", command=save_actions, bg="#19a974", fg="white", relief="flat", padx=16, pady=7).pack(side="right")
        tk.Button(row, text="Cancel", command=dialog.destroy, bg="#273343", fg="white", relief="flat", padx=16, pady=7).pack(side="right", padx=(0, 8))
        self.centre_dialog(dialog)

    def open_activity_log(self):
        dialog = self.dark_dialog("WAVELY activity")
        tk.Label(dialog, text="Activity log", bg="#10151c", fg="#f2f6fa", font=("Segoe UI", 14, "bold"), padx=24, pady=14).pack()
        tk.Label(dialog, text="A timestamped history of recognised gestures, enrolments, and reported errors.", bg="#10151c", fg="#aeb8c6").pack(anchor="w", padx=24)
        view = scrolledtext.ScrolledText(dialog, width=80, height=22, bg="#0b0e12", fg="#d7e1ec", font=("Cascadia Mono", 9), relief="flat"); view.pack(padx=16, pady=4); view.insert("1.0", ACTIVITY_FILE.read_text(encoding="utf-8") if ACTIVITY_FILE.exists() else "No activity recorded yet.\n"); view.configure(state="disabled")
        tk.Button(dialog, text="Close", command=dialog.destroy, bg="#273343", fg="white", relief="flat", padx=16, pady=7).pack(pady=14); self.centre_dialog(dialog)

    def start_test_mode(self):
        if not self.process or self.process.poll() is not None:
            self.start_vision(test_mode=True)

    def toggle_vision(self):
        if self.process and self.process.poll() is None:
            self.stop_vision()
        else:
            self.start_vision()

    def start_vision(self, test_mode=False):
        # The engine replaces this default with the actual live stream size.
        self.stream_aspect = 16 / 9
        self.aspect_file_mtime = None
        try:
            CAMERA_ASPECT_FILE.unlink()
        except FileNotFoundError:
            pass
        self.camera_message.place_forget()
        self.write_log("\nStarting WAVELY Vision...\n")
        environment = os.environ.copy()
        environment["WAVELY_EMBEDDED"] = "1"
        if test_mode:
            environment["WAVELY_TEST_MODE"] = "1"
        vision_command = (
            [str(VISION_PROGRAM)]
            if VISION_PROGRAM.exists()
            else [sys.executable, str(VISION_SCRIPT)]
        )
        self.process = subprocess.Popen(
            vision_command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.process_mode = "vision"
        threading.Thread(target=self.read_logs, daemon=True).start()
        self.toggle_button.configure(text="Stop WAVELY Vision", style="Vision.Stop.TButton")
        self.set_status("Starting gesture test mode..." if test_mode else "Starting camera...", "#f2c14e")
        self.root.after(150, self.embed_vision_window)

    def start_enrolment(self, person=None, pose_index=0):
        if self.process and self.process.poll() is None:
            return
        person = person or self.person_var.get()
        if person == "Select person...":
            person = ""
        if not person:
            messagebox.showwarning("Person required", "Enter the name of the new person before starting enrolment.", parent=self.root)
            return
        pose_name = "all poses" if pose_index == 0 else self.enrolment_pose_labels()[pose_index - 1]
        self.write_log(f"\nStarting {pose_name} enrolment for {person}...\n")
        self.active_enrolment_person = person
        self.enrolment_start_image_count = self.count_enrolment_images(person)
        self.enrolment_stopped_early = False
        # Enrolment has a 155px instruction strip above the live camera image.
        self.stream_aspect = 1280 / 875
        self.aspect_file_mtime = None
        try:
            CAMERA_ASPECT_FILE.unlink()
        except FileNotFoundError:
            pass
        self.process = subprocess.Popen(
            [str(ENROLMENT_PROGRAM), "--person", person, "--pose-index", str(pose_index)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.process_mode = "enrolment"
        threading.Thread(target=self.read_logs, daemon=True).start()
        self.toggle_button.configure(state="disabled")
        self.enrol_button.configure(
            state="normal", text="Stop Enrolment", bg="#a93d3e", activebackground="#7f2d2e"
        )
        self.set_status("Video enrolment is running", "#f2c14e")
        self.camera_message.place_forget()
        self.root.after(150, self.embed_enrolment_window)

    def stop_vision(self):
        if self.process and self.process.poll() is None:
            self.write_log("Stopping WAVELY Vision...\n")
            self.toggle_button.configure(state="disabled", text="Stopping WAVELY Vision...")
            self.set_status("Stopping camera...", "#f2c14e")
            threading.Thread(target=self.stop_process_tree, daemon=True).start()

    def stop_enrolment(self):
        if self.process_mode != "enrolment" or not self.process or self.process.poll() is not None:
            return
        self.enrolment_stopped_early = True
        self.write_log("Stopping enrolment early. Keeping images captured so far...\n")
        self.enrol_button.configure(state="disabled", text="Stopping enrolment...")
        self.set_status("Stopping enrolment...", "#f2c14e")
        threading.Thread(target=self.stop_process_tree, daemon=True).start()

    def stop_process_tree(self):
        """A one-file EXE has a launcher and a child process; stop both."""
        if not self.process:
            return
        subprocess.run(
            ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def read_logs(self):
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            self.root.after(0, self.write_log, line)

    def run_command(self, _event=None):
        command = self.command_entry.get().strip()
        if not command:
            return
        self.command_entry.delete(0, "end")
        if not self.command_history or self.command_history[-1] != command:
            self.command_history.append(command)
        self.command_history = self.command_history[-50:]
        self.history_index = len(self.command_history)
        self.save_layout()
        self.write_log(f"\n> {command}\n")

        built_in = command.casefold()
        if self.pending_deletion:
            self.complete_deletion(built_in)
            return
        if built_in == "help":
            self.show_help()
            return
        delete_match = re.fullmatch(r'delete enrolment\s+(?:["“](.+?)["”]|(.+?))', command, re.IGNORECASE)
        if delete_match:
            self.request_deletion((delete_match.group(1) or delete_match.group(2)).strip())
            return
        if built_in in {"test enrolment", "test enrollment"}:
            self.test_enrolment_complete()
            return

        threading.Thread(target=self.execute_command, args=(command,), daemon=True).start()

    def previous_command(self, _event=None):
        if not self.command_history:
            return "break"
        self.history_index = max(0, self.history_index - 1)
        self.set_command_entry(self.command_history[self.history_index])
        return "break"

    def next_command(self, _event=None):
        if not self.command_history:
            return "break"
        self.history_index = min(len(self.command_history), self.history_index + 1)
        command = "" if self.history_index == len(self.command_history) else self.command_history[self.history_index]
        self.set_command_entry(command)
        return "break"

    def set_command_entry(self, command):
        self.command_entry.delete(0, "end")
        self.command_entry.insert(0, command)
        self.command_entry.icursor("end")

    def show_help(self):
        self.write_log(
            "WAVELY COMMANDS\n"
            "===============\n"
            "help                 Show this command list\n"
            "test enrolment       Simulate a completed video enrolment\n"
            "                      and show the train-model choice\n"
            "delete enrolment \"Name\"  Request deletion of a person's enrolment\n"
            "confirm              Confirm the pending enrolment deletion\n"
            "Any other command    Run a normal Windows command\n\n"
            "Panel controls\n"
            "- Start WAVELY Vision: start or stop gesture recognition\n"
            "- Video Enrolment: capture new face-enrolment images\n"
        )

    def test_enrolment_complete(self):
        self.write_log(
            "TEST: Video enrolment completed successfully.\n"
            "TEST: 24 useful face images were saved.\n"
            "TEST: No camera or face-model files were changed.\n"
        )
        self.root.after(150, lambda: self.ask_to_train_model(test_only=True))

    def request_deletion(self, person):
        matches = {name.casefold(): name for name in self.available_people()}
        if person.casefold() not in matches:
            self.write_log(f'No enrolment was found for "{person}".\n')
            return
        self.pending_deletion = matches[person.casefold()]
        self.write_log(f'Are you sure you want to delete enrolment "{self.pending_deletion}"? Type confirm to confirm.\n')

    def complete_deletion(self, command):
        person = self.pending_deletion
        self.pending_deletion = None
        if command != "confirm":
            self.write_log(f'Deletion of "{person}" cancelled.\n')
            return
        folder = FACES_DIR / person
        shutil.rmtree(folder)
        self.write_log(f'Enrolment for "{person}" deleted. Rebuilding the face model...\n')
        self.start_face_training()

    def execute_command(self, command):
        try:
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", command],
                cwd=ROOT,
                capture_output=True,
                text=True,
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if output:
                self.root.after(0, self.write_log, output)
            self.root.after(0, self.write_log, f"[Command finished: {result.returncode}]\n")
        except OSError as error:
            self.root.after(0, self.write_log, f"[Could not run command: {error}]\n")

    def embed_vision_window(self):
        if not self.process or self.process.poll() is not None:
            return
        handle = user32.FindWindowW(None, WINDOW_TITLE)
        if not handle:
            self.root.after(150, self.embed_vision_window)
            return

        self.vision_handle = handle
        style = user32.GetWindowLongW(handle, GWL_STYLE)
        style = (style | WS_CHILD) & ~(WS_CAPTION | WS_THICKFRAME | WS_POPUP)
        user32.SetWindowLongW(handle, GWL_STYLE, style)
        user32.SetParent(handle, self.camera_host.winfo_id())
        self.resize_embedded_vision()
        self.set_status("Vision is running", "#19a974")

    def embed_enrolment_window(self):
        if not self.process or self.process.poll() is not None:
            return
        handle = user32.FindWindowW(None, ENROLMENT_WINDOW_TITLE)
        if not handle:
            self.root.after(150, self.embed_enrolment_window)
            return
        self.vision_handle = handle
        style = user32.GetWindowLongW(handle, GWL_STYLE)
        style = (style | WS_CHILD) & ~(WS_CAPTION | WS_THICKFRAME | WS_POPUP)
        user32.SetWindowLongW(handle, GWL_STYLE, style)
        user32.SetParent(handle, self.camera_host.winfo_id())
        self.resize_embedded_vision()

    def resize_embedded_vision(self, _event=None):
        if self.vision_handle:
            host_width = max(1, self.camera_host.winfo_width())
            host_height = max(1, self.camera_host.winfo_height())
            aspect = self.stream_aspect if self.stream_aspect > 0 else 16 / 9
            if host_width / host_height > aspect:
                height = host_height
                width = max(1, round(height * aspect))
            else:
                width = host_width
                height = max(1, round(width / aspect))
            x = (host_width - width) // 2
            y = (host_height - height) // 2
            user32.SetWindowPos(
                self.vision_handle, 0, x, y, width, height,
                SWP_NOZORDER | SWP_FRAMECHANGED,
            )

    def poll_camera_aspect(self):
        """Use the live stream dimensions without resizing the dashboard panel."""
        try:
            mtime = CAMERA_ASPECT_FILE.stat().st_mtime_ns
            if mtime != self.aspect_file_mtime:
                data = json.loads(CAMERA_ASPECT_FILE.read_text(encoding="utf-8"))
                width = int(data.get("width", 0))
                height = int(data.get("height", 0))
                if width > 0 and height > 0:
                    self.stream_aspect = width / height
                    self.aspect_file_mtime = mtime
                    self.resize_embedded_vision()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        self.root.after(350, self.poll_camera_aspect)

    def detach_vision_window(self):
        if self.vision_handle:
            try:
                user32.SetParent(self.vision_handle, 0)
            except OSError:
                pass
            self.vision_handle = 0

    def set_stopped_state(self):
        self.toggle_button.configure(
            state="normal",
            text="Start WAVELY Vision",
            style="Vision.Start.TButton",
        )
        self.enrol_button.configure(
            state="normal", text="Video Enrolment  ▾", bg="#356da8", activebackground="#285681"
        )
        self.set_status("Vision is off", "#cc4b4c")
        self.camera_message.place(relx=0.5, rely=0.5, anchor="center")

    def watch_process(self):
        if self.process and self.process.poll() is not None:
            finished_mode = self.process_mode
            self.detach_vision_window()
            self.set_stopped_state()
            self.process = None
            self.process_mode = None
            if finished_mode == "enrolment":
                saved = max(
                    0,
                    self.count_enrolment_images(self.active_enrolment_person or "")
                    - self.enrolment_start_image_count,
                )
                if self.enrolment_stopped_early:
                    self.write_log(f"Enrolment stopped early. {saved} new face images were kept.\n")
                else:
                    self.write_log(f"Enrolment completed. {saved} new face images were saved.\n")
                self.refresh_people()
                self.root.after(250, self.ask_to_train_model)
        self.root.after(300, self.watch_process)

    def ask_to_train_model(self, test_only=False):
        train_now = messagebox.askyesno(
            "Face enrolment complete",
            "Video enrolment has completed.\n\nTrain the face model using the new enrolment images now?",
            parent=self.root,
        )
        if test_only:
            result = "accepted" if train_now else "declined"
            self.write_log(f"TEST: Face-model training choice {result}. No model files were changed.\n")
        elif train_now:
            self.start_face_training()
        else:
            self.write_log("Face model training skipped. You can run it later from launchers\\train_face_model.bat.\n")

    def start_face_training(self):
        self.write_log("\nStarting face model training...\n")
        self.process = subprocess.Popen(
            [str(TRAINING_PROGRAM)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.process_mode = "training"
        threading.Thread(target=self.read_logs, daemon=True).start()
        self.toggle_button.configure(state="disabled")
        self.enrol_button.configure(state="disabled", text="Training model...")
        self.set_status("Training face model...", "#f2c14e")

    def close(self):
        self.save_layout()
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    WavelyDashboard().run()


