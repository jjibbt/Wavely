WAVELY - END-USER GUIDE
=======================

1. WHAT WAVELY IS
-----------------
WAVELY is a Windows computer-vision application. It uses a webcam to detect
and track hands, recognise supported hand gestures, and recognise enrolled
people using local face-recognition data.

WAVELY can:

- detect and track hands through the webcam
- recognise its supported hand gestures and gesture combinations
- associate detected hands with recognised people
- send configured gesture mappings to Home Assistant
- enrol face images for a person
- train or update the local face-recognition model
- provide spoken feedback where applicable
- configure the camera and speech audio-output device

Audio input selection is currently groundwork for future microphone features.
WAVELY does not currently use a microphone for an active feature.

2. SYSTEM REQUIREMENTS
----------------------

- Windows 64-bit
- A working webcam for vision, face-enrolment, and gesture features
- Windows camera permissions for WAVELY
- Optional: Home Assistant for Home Assistant actions

You do not need to install Python, MediaPipe, OpenCV, VS Code, or a development
environment. The required runtime components are bundled in the installed
WAVELY application.

3. INSTALLATION
---------------
1. Download and run WAVELY-Setup.exe.
2. Follow the Windows setup steps.
3. Choose a different installation directory during setup if required.
4. Start WAVELY from the Start Menu, desktop shortcut if created, or the
   installed application directory.

The default installation directory is:

  %LOCALAPPDATA%\Programs\WAVELY

Normal users should install WAVELY with WAVELY-Setup.exe. Do not run Python
source files; they are not required for normal use.

4. FIRST START
--------------
On the first start, WAVELY opens the first-run setup screen. Configure the
following options:

Camera
  Choose a camera by its friendly Windows device name. WAVELY keeps the
  corresponding OpenCV camera index internally so the existing vision pipeline
  can open it.

Audio input
  Choose a friendly Windows audio-input device if desired. This selection is
  reserved for future microphone functionality. It does not currently control
  an active microphone feature.

Audio output
  Choose the preferred Windows/SAPI speech output when WAVELY can address it.
  If that device cannot be addressed directly, WAVELY may use the Windows
  default speech output instead.

Resolution
  Select the camera resolution. Higher resolutions can provide more detail but
  use more processing power.

Camera mirroring
  Mirror the camera image so it behaves like a normal mirror.

Start WAVELY with Windows
  Start WAVELY automatically after signing in to Windows.

Gesture sensitivity
  Choose how steadily a gesture must be held before it is accepted.

Preview / test camera
  Save the settings and start the camera in gesture test mode. Test mode does
  not send Home Assistant webhooks.

Save
  Save the settings without starting the test camera.

Selected devices and settings are remembered in your WAVELY user data. Use
Refresh devices to rescan connected cameras, and Refresh audio devices to
rescan audio devices.

The dashboard also shows a compact readiness summary. Open Tools > Health check
for an actionable check of setup, camera, enrolled people, face model, Home
Assistant configuration, and the installed vision runtime.

If a previously selected camera is disconnected, WAVELY first attempts to
restore it by its Windows device identity and then by its saved name. If it is
not available, WAVELY safely selects another available camera and reports the
fallback in the settings window.

5. BASIC USE
------------
1. Launch WAVELY.
2. Open Tools > Settings and camera and select the webcam.
3. Use Preview / test camera if you need to check the image or gestures.
4. If face recognition is required, add or enrol the person through the People
   manager or Video Enrolment controls.
5. Train or update the face model after adding new face images.
6. Configure gesture mappings and Home Assistant actions if required.
7. Select Start WAVELY Vision to start the live vision and gesture system.
8. Use Tools > Settings and camera later to change the camera, audio devices,
   resolution, mirroring, startup, or gesture-sensitivity settings.

The top-right readiness summary shows whether setup is ready and identifies the
selected camera, face-model state, and Home Assistant configuration. Tools >
Health check gives the same information with suggestions for resolving missing
required components. The log panel has Copy and Clear controls; Clear only
clears the on-screen view and does not delete the saved activity history.

Use the sun/moon button beside Tools to switch between Light and Dark mode.
The choice is saved in your WAVELY user data and applies immediately, including
the native Windows title bar. Existing installations without a saved choice
default to Dark mode. WAVELY does not follow Windows appearance changes after
startup; use the WAVELY toggle when you want to change mode.

Keyboard shortcuts:

  Ctrl+,   Open Settings and camera
  F5       Open Settings and camera
  Ctrl+L   Open the activity log

Face recognition identifies people from locally enrolled face data. Gesture
recognition detects the supported gestures implemented by WAVELY and applies
the configured hold, hand, person, and combination rules. Home Assistant is
optional; gestures can still be tested without sending Home Assistant actions.

6. CAMERA AND AUDIO DEVICES
---------------------------
Camera names come from Windows Plug-and-Play camera devices. WAVELY separately
probes working OpenCV camera indices and pairs those usable indices with the
Windows names shown in the dropdown. The matching OpenCV index is retained in
settings so the existing OpenCV/MediaPipe camera pipeline remains compatible.

Refresh devices rescans connected cameras. Reconnect a camera and refresh if it
does not appear.

Audio input names and identifiers come from Windows audio endpoints, with a
native Windows wave-in fallback. Audio input is currently stored only as
future microphone groundwork and does not control an active WAVELY feature.

Audio output names are obtained from Windows SAPI output tokens when available,
with Windows audio-endpoint and native waveOut fallbacks. WAVELY speech uses a
selected SAPI output when its token can be addressed. Otherwise Windows may
route speech through its default speech output.

7. DATA LOCATIONS
-----------------
WAVELY has separate program files and user data.

Application installation

  %LOCALAPPDATA%\Programs\WAVELY

This contains the WAVELY dashboard executable, the vision, enrolment, and
training executables, bundled runtime components, launchers, configuration
templates, documentation, and installer metadata.

User data

  %LOCALAPPDATA%\Wavely

The user-data directory may contain:

  config\wavely_app_settings.json       Application and device selections
  config\home_assistant_actions.json   Home Assistant webhook and mappings
  config\face_model.yml                 Trained local face-recognition model
  config\face_labels.json               Face-model label mapping
  config\wavely_dashboard.json          Dashboard state and command history
  config\wavely_window.json             Standalone vision-window state
  config\wavely_console_window.json     Console-window state, if created
  config\active_camera_aspect.json      Generated live camera dimensions
  faces\                                Enrolled face images
  logs\wavely_activity.log              WAVELY activity log
  backups\                              WAVELY settings/import backups

The configuration directory also stores the selected camera index, Windows
camera identity/name, audio input identity/name, and audio output identity/name
in wavely_app_settings.json.

Generated and temporary state includes the active camera aspect file, window
and dashboard state, settings-import backups, and short-lived .tmp files used
while safely writing JSON. PyInstaller one-file executables may unpack temporary
runtime files under Windows %TEMP%; those are managed by PyInstaller rather
than being WAVELY user data.

When Start WAVELY with Windows is enabled, WAVELY also creates:

  %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\WAVELY Vision.cmd

This startup command is removed by the WAVELY uninstaller.

Do not confuse these installed paths with development source or build folders.
Development paths such as C:\Wavely are not installed application data.

8. BACKING UP OR MOVING TO ANOTHER PC
--------------------------------------
To preserve all WAVELY personal data, close WAVELY and back up:

  %LOCALAPPDATA%\Wavely

This preserves application settings, camera and audio selections, Home
Assistant mappings, enrolled face images, the trained face model, face labels,
logs, dashboard/window state, generated state, and WAVELY settings backups.

Install the program on the other PC with WAVELY-Setup.exe rather than copying
the installed program files from another PC. Then close WAVELY and restore the
backed-up %LOCALAPPDATA%\Wavely directory for that Windows user.

The camera number and available Windows device identities can differ between
PCs. Review the camera and audio selections after the move. Protect backups
containing enrolled faces or trained recognition data.

For only application settings and Home Assistant mappings, use Tools > Export
settings and import the resulting settings file on the other PC. That export
does not include face images, the trained face model, or logs.

9. NORMAL WINDOWS UNINSTALL
---------------------------
Use the normal Windows uninstall flow:

1. Open Windows Settings.
2. Open Apps.
3. Open Installed apps.
4. Find WAVELY.
5. Open its menu/options and choose Uninstall.

During uninstall, WAVELY asks whether to preserve or remove WAVELY user data.

Choose:

  Uninstall and keep my data

to remove the installed program while preserving:

  %LOCALAPPDATA%\Wavely

This keeps settings, camera/audio selections, Home Assistant mappings, face
images, trained models, face labels, logs, and generated state for reuse after
reinstalling WAVELY.

10. MANUAL UNINSTALLER
----------------------
If WAVELY cannot conveniently be removed through Windows Settings, run:

  %LOCALAPPDATA%\Programs\WAVELY\unins000.exe

The manual uninstaller provides the same choice to keep or remove WAVELY user
data.

11. COMPLETE REMOVAL
--------------------
Choose this option in the uninstaller:

  Remove all Wavely user data and settings

This removes the WAVELY user-data directory:

  %LOCALAPPDATA%\Wavely

It deletes WAVELY-specific settings, camera/audio selections, Home Assistant
mappings, enrolled face images, trained face models, face labels, logs, cache
and state files, and WAVELY-created settings/import backups. It does not delete
unrelated user files or shared Windows resources.

For manual complete removal:

1. Uninstall WAVELY normally, or run:

     %LOCALAPPDATA%\Programs\WAVELY\unins000.exe

2. If you intentionally want to remove all remaining WAVELY personal data,
   delete:

     %LOCALAPPDATA%\Wavely

Deleting that directory permanently removes enrolled faces, trained models,
settings, mappings, logs, and WAVELY backups. Do not delete anything outside
WAVELY-owned directories for this purpose.

12. PRIVACY AND PERSONAL DATA
-----------------------------
Enrolled face images, face labels, and the trained face-recognition model are
stored locally in:

  %LOCALAPPDATA%\Wavely

Treat this as personal data. Back it up only where appropriate and delete it
when it is no longer needed according to your own privacy and retention needs.

WAVELY sends data to Home Assistant only when the optional Home Assistant
integration is configured and a recognised gesture triggers a configured
action. Face images and the local face model are not part of the settings
export.

13. HOME ASSISTANT
------------------
Home Assistant integration is optional. Gesture mappings and the configured
webhook address are stored in:

  %LOCALAPPDATA%\Wavely\config\home_assistant_actions.json

Use Tools > Home Assistant actions to configure mappings. The installed
application includes a Home Assistant setup example under:

  %LOCALAPPDATA%\Programs\WAVELY\config\home_assistant_webhook_example.yaml

The example contains no personal webhook URL or credential. Do not share a
configured settings file containing a real webhook address unless it is safe to
do so.

14. TROUBLESHOOTING
-------------------

Camera does not appear
  Reconnect the camera, open Tools > Settings and camera, and choose Refresh
  devices. Check Windows camera permissions and reopen the settings window if
  necessary.

Wrong camera is selected
  Open Settings and camera and choose the correct friendly Windows device name.
  Save the settings and restart the vision system.

No speech from the expected device
  Open Settings and camera, choose another audio output, and check the Windows
  default speech/audio output. Some Windows audio endpoints cannot be selected
  directly by SAPI, so WAVELY may use the Windows default speech output.

Face is not recognised
  Enrol or re-enrol the person with varied poses and lighting, then train or
  update the face model before starting vision again.

Reinstall while preserving configuration
  Uninstall WAVELY and choose Uninstall and keep my data. Reinstall with
  WAVELY-Setup.exe. The existing %LOCALAPPDATA%\Wavely data will be reused.

Complete reset
  Uninstall WAVELY and choose Remove all Wavely user data and settings. This
  permanently removes the WAVELY user-data directory and its personal data.

15. VERSION AND DISTRIBUTION
----------------------------
Normal users should install and run WAVELY using WAVELY-Setup.exe. The source
files, Python environment, build scripts, and development folders are for
building or maintaining WAVELY and are not required on an end-user PC.
