from pathlib import Path
import sys
APP_ROOT = Path(sys.executable).resolve().parent.parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
import cv2
import mediapipe as mp
import os
import time
import re
import math
import numpy as np
import pyttsx3
import winsound
import threading
import argparse
import json
from wavely_paths import CONFIG_DIR, FACES_DIR, prepare_user_data
from wavely_devices import configure_speech_output


# ==========================================================
# SETTINGS
# ==========================================================
prepare_user_data()

parser = argparse.ArgumentParser(description="WAVELY video face enrolment")
parser.add_argument("--person", default="Person", help="Name of the person to enrol")
parser.add_argument(
    "--pose-index", type=int, default=0,
    help="0 captures every pose; 1 onward captures one pose from the guided list.",
)
parser.add_argument("--list-cameras", action="store_true", help="List working camera numbers as JSON")
args = parser.parse_args()
if args.list_cameras:
    available = []
    for index in range(10):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            camera = cv2.VideoCapture(index, backend)
            try:
                if camera.isOpened() and camera.read()[0]:
                    available.append(index)
                    break
            finally:
                camera.release()
    print(json.dumps({"cameras": available}))
    raise SystemExit(0)
PERSON_NAME = " ".join(args.person.split()).strip()
if not PERSON_NAME:
    raise SystemExit("A person name is required.")
safe_person_folder = re.sub(r"[^A-Za-z0-9 _-]", "", PERSON_NAME).strip()
if not safe_person_folder:
    raise SystemExit("The person name contains no usable characters.")
OUTPUT_FOLDER = os.path.join(str(FACES_DIR), safe_person_folder)

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_ASPECT_FILE = str(CONFIG_DIR / "active_camera_aspect.json")
ENROLMENT_HEADER_HEIGHT = 155
_app_settings = {}

try:
    with open(str(CONFIG_DIR / "wavely_app_settings.json"), "r", encoding="utf-8") as settings_file:
        _app_settings = json.load(settings_file)
    CAMERA_INDEX = int(_app_settings.get("camera_index", CAMERA_INDEX))
    _resolution = str(_app_settings.get("resolution", f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}")).split("x")
    if len(_resolution) == 2:
        CAMERA_WIDTH, CAMERA_HEIGHT = int(_resolution[0]), int(_resolution[1])
except (OSError, ValueError, json.JSONDecodeError):
    pass

_last_published_aspect = None


def publish_display_aspect(camera_width, camera_height):
    """Publish the full enrolment view: instructions plus camera image."""
    global _last_published_aspect
    shape = (int(camera_width), int(camera_height) + ENROLMENT_HEADER_HEIGHT)
    if shape == _last_published_aspect or shape[0] <= 0 or shape[1] <= 0:
        return
    _last_published_aspect = shape
    try:
        with open(CAMERA_ASPECT_FILE, "w", encoding="utf-8") as aspect_file:
            json.dump({"width": shape[0], "height": shape[1]}, aspect_file)
    except OSError:
        pass

# Maximum rate at which useful training images are saved.
# 0.18 sec = about 5.5 possible images per second.
CAPTURE_INTERVAL = 0.18

# First part of each pose gives you time to move into position.
SETTLE_TIME = 5.0

# The prior value was appropriate only for close-up face crops. At the
# standing-distance camera position it rejected almost every usable frame.
# Standing-distance crops naturally contain less fine detail. Keep only the
# truly unusable frames rather than rejecting the entire enrolment session.
BLUR_THRESHOLD = 2.0

# Reject near-identical consecutive face crops.
DIFFERENCE_THRESHOLD = 2.5

# Saved training face size.
FACE_OUTPUT_SIZE = 300

# Extra space around face.
FACE_PADDING = 0.28


# ==========================================================
# GUIDED POSES
# ==========================================================

POSES = [
    (
        "LOOK STRAIGHT - NEUTRAL FACE",
        7.0
    ),

    (
        "SLOWLY TURN YOUR HEAD LEFT",
        7.0
    ),

    (
        "SLOWLY TURN YOUR HEAD RIGHT",
        7.0
    ),

    (
        "LOOK SLIGHTLY UP - MOVE HEAD NATURALLY",
        7.0
    ),

    (
        "LOOK SLIGHTLY DOWN - MOVE HEAD NATURALLY",
        7.0
    ),

    (
        "SMILE - CHANGE THE SIZE OF YOUR SMILE",
        7.0
    ),

    (
        "TALK / MOVE YOUR MOUTH / CHANGE EXPRESSION",
        8.0
    ),

    (
        "MOVE A LITTLE CLOSER - KEEP MOVING NATURALLY",
        7.0
    ),

    (
        "MOVE A LITTLE FARTHER AWAY",
        7.0
    ),

    (
        "SLOWLY MOVE YOUR HEAD THROUGH DIFFERENT ANGLES",
        10.0
    )
]

# A single pose is useful for topping up a weak angle without making the
# person repeat the entire guided session.
if args.pose_index < 0 or args.pose_index > len(POSES):
    raise SystemExit(f"Pose index must be between 0 and {len(POSES)}.")
if args.pose_index:
    POSES = [POSES[args.pose_index - 1]]


# ==========================================================
# OUTPUT FOLDER
# ==========================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==========================================================
# FIND NEXT VIDEO IMAGE NUMBER
# ==========================================================

existing_numbers = []

pattern = re.compile(
    r"video_(\d+)\.jpg",
    re.IGNORECASE
)

for filename in os.listdir(
    OUTPUT_FOLDER
):
    match = pattern.fullmatch(
        filename
    )

    if match:
        existing_numbers.append(
            int(
                match.group(1)
            )
        )


if existing_numbers:
    image_number = max(
        existing_numbers
    ) + 1

else:
    image_number = 1


# ==========================================================
# MEDIAPIPE FACE DETECTION
# ==========================================================

mp_face_detection = (
    mp.solutions.face_detection
)


# ==========================================================
# PROFILE DETECTOR FALLBACK
# ==========================================================

profile_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_profileface.xml"
)


# ==========================================================
# CAMERA
# ==========================================================

def open_camera(preferred_index):
    """Try the selected camera across Windows backends, then other devices."""
    indices = [preferred_index] + [index for index in range(6) if index != preferred_index]
    for index in indices:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            candidate = cv2.VideoCapture(index, backend)
            if candidate.isOpened():
                if index != preferred_index:
                    print(f"Selected camera {preferred_index} unavailable; using camera {index}.")
                return candidate
            candidate.release()
    raise SystemExit("No camera is available. Check its connection and Windows camera permissions.")


cap = open_camera(CAMERA_INDEX)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        *"MJPG"
    )
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    print(
        "Could not open camera."
    )

    raise SystemExit


# ==========================================================
# WINDOW
# ==========================================================

window_name = (
    "WAVELY Video Face Enrolment"
)

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    1280,
    875
)


# ==========================================================
# HELPERS
# ==========================================================

def crop_square_face(
    frame,
    x,
    y,
    w,
    h
):

    frame_h, frame_w = (
        frame.shape[:2]
    )

    centre_x = (
        x + w / 2
    )

    centre_y = (
        y + h / 2
    )

    side = max(
        w,
        h
    )

    side = side * (
        1.0 + FACE_PADDING
    )

    x1 = int(
        centre_x
        -
        side / 2
    )

    y1 = int(
        centre_y
        -
        side / 2
    )

    x2 = int(
        centre_x
        +
        side / 2
    )

    y2 = int(
        centre_y
        +
        side / 2
    )

    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        frame_w,
        x2
    )

    y2 = min(
        frame_h,
        y2
    )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    crop = cv2.resize(
        crop,
        (
            FACE_OUTPUT_SIZE,
            FACE_OUTPUT_SIZE
        )
    )

    return crop


def blur_score(
    face
):

    grey = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        grey,
        cv2.CV_64F
    ).var()


def brightness_score(
    face
):

    grey = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        np.mean(
            grey
        )
    )


def face_difference(
    face_a,
    face_b
):

    if (
        face_a is None
        or
        face_b is None
    ):
        return 999.0

    a = cv2.resize(
        face_a,
        (
            100,
            100
        )
    )

    b = cv2.resize(
        face_b,
        (
            100,
            100
        )
    )

    a = cv2.cvtColor(
        a,
        cv2.COLOR_BGR2GRAY
    )

    b = cv2.cvtColor(
        b,
        cv2.COLOR_BGR2GRAY
    )

    difference = cv2.absdiff(
        a,
        b
    )

    return float(
        np.mean(
            difference
        )
    )


def detect_profile_fallback(
    frame
):

    grey = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = profile_detector.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(
            70,
            70
        )
    )

    if len(faces) > 0:

        face = max(
            faces,
            key=lambda rect:
            rect[2] * rect[3]
        )

        return tuple(
            face
        )


    # Try opposite profile by flipping image.

    flipped = cv2.flip(
        grey,
        1
    )

    faces = profile_detector.detectMultiScale(
        flipped,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(
            70,
            70
        )
    )

    if len(faces) > 0:

        x, y, w, h = max(
            faces,
            key=lambda rect:
            rect[2] * rect[3]
        )

        original_x = (
            grey.shape[1]
            -
            x
            -
            w
        )

        return (
            original_x,
            y,
            w,
            h
        )

    return None


# ==========================================================
# STARTUP
# ==========================================================

print()
print(
    "WAVELY VIDEO FACE ENROLMENT"
)

print(
    "============================"
)

print()

print(
    f"Person: {PERSON_NAME}"
)

print()

print(
    "The program will automatically capture useful"
)

print(
    "training images while you follow the instructions."
)

print()

print(
    "You do NOT need to press anything."
)

print()

print(
    "Q or ESC = stop early."
)

print()


# ==========================================================
# STATE
# ==========================================================

saved_count = 0

rejected_blurry = 0
rejected_duplicate = 0
rejected_brightness = 0

last_save_time = 0

previous_saved_face = None

def announce(message):

    print(message, flush=True)

    def speak():

        speech_engine = pyttsx3.init()
        configure_speech_output(speech_engine, _app_settings.get("audio_output_device_id", ""))

        # Prefer the user's selected Narrator voice when Windows exposes it
        # to the desktop speech engine.
        for voice in speech_engine.getProperty("voices"):

            voice_text = (
                f"{voice.id} {voice.name}"
            ).lower()

            if (
                "andrew"
                in
                voice_text
            ):

                speech_engine.setProperty(
                    "voice",
                    voice.id
                )

                break

        speech_engine.setProperty("rate", 175)
        speech_engine.say(message)
        speech_engine.runAndWait()
        speech_engine.stop()


    threading.Thread(
        target=speak,
        daemon=True
    ).start()


def capture_start_beeps():

    winsound.Beep(880, 220)


def capture_end_beep():

    winsound.Beep(440, 280)

session_start = time.time()

total_duration = sum(
    duration + SETTLE_TIME
    for _,
    duration
    in POSES
)


# ==========================================================
# FACE DETECTOR
# ==========================================================

with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.60
) as face_detection:


    # ======================================================
    # POSE LOOP
    # ======================================================

    quit_requested = False


    for pose_number, (
        pose_name,
        pose_duration
    ) in enumerate(
        POSES,
        start=1
    ):

        announce(
            pose_name.replace(
                "-",
                " "
            )
        )

        pose_start = (
            time.time()
        )

        capture_announced = False


        while True:

            now = time.time()

            pose_elapsed = (
                now
                -
                pose_start
            )

            seconds_left = max(
                0,
                int(
                    math.ceil(
                        SETTLE_TIME
                        -
                        pose_elapsed
                    )
                )
            )


            if (
                pose_elapsed >= SETTLE_TIME
                and not capture_announced
            ):

                capture_start_beeps()
                capture_announced = True

            if (
                pose_elapsed
                >=
                pose_duration + SETTLE_TIME
            ):
                break


            # ==============================================
            # CAMERA
            # ==============================================

            ret, frame = (
                cap.read()
            )

            if not ret:

                print(
                    "Could not read camera."
                )

                quit_requested = True
                break


            frame = cv2.flip(
                frame,
                1
            )


            # ==============================================
            # FACE DETECTION
            # ==============================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = (
                face_detection.process(
                    rgb
                )
            )

            detected_rect = None

            detector_name = ""


            if (
                result.detections
                and
                len(
                    result.detections
                )
                >
                0
            ):

                # Only one person is expected during enrolment.
                detection = max(
                    result.detections,
                    key=lambda detection:
                    detection.score[0]
                )

                bbox = (
                    detection
                    .location_data
                    .relative_bounding_box
                )

                frame_h, frame_w = (
                    frame.shape[:2]
                )

                x = int(
                    bbox.xmin
                    *
                    frame_w
                )

                y = int(
                    bbox.ymin
                    *
                    frame_h
                )

                w = int(
                    bbox.width
                    *
                    frame_w
                )

                h = int(
                    bbox.height
                    *
                    frame_h
                )

                detected_rect = (
                    x,
                    y,
                    w,
                    h
                )

                detector_name = (
                    "MEDIAPIPE"
                )


            # ==============================================
            # PROFILE FALLBACK
            # ==============================================

            if detected_rect is None:

                detected_rect = (
                    detect_profile_fallback(
                        frame
                    )
                )

                if (
                    detected_rect
                    is not None
                ):

                    detector_name = (
                        "PROFILE"
                    )


            face_crop = None

            quality_text = (
                "NO FACE DETECTED"
            )


            # ==============================================
            # FACE FOUND
            # ==============================================

            if (
                detected_rect
                is not None
            ):

                x, y, w, h = (
                    detected_rect
                )

                face_crop = (
                    crop_square_face(
                        frame,
                        x,
                        y,
                        w,
                        h
                    )
                )


                if face_crop is not None:

                    blur = (
                        blur_score(
                            face_crop
                        )
                    )

                    brightness = (
                        brightness_score(
                            face_crop
                        )
                    )

                    difference = (
                        face_difference(
                            face_crop,
                            previous_saved_face
                        )
                    )


                    # ======================================
                    # QUALITY STATUS
                    # ======================================

                    if (
                        blur
                        <
                        BLUR_THRESHOLD
                    ):

                        quality_text = (
                            f"TOO BLURRY ({blur:.0f})"
                        )


                    elif (
                        brightness
                        <
                        35
                        or
                        brightness
                        >
                        225
                    ):

                        quality_text = (
                            "LIGHTING NOT IDEAL"
                        )


                    elif (
                        difference
                        <
                        DIFFERENCE_THRESHOLD
                    ):

                        quality_text = (
                            "TOO SIMILAR - MOVE SLIGHTLY"
                        )


                    else:

                        quality_text = (
                            f"GOOD FACE - {detector_name}"
                        )


                    # ======================================
                    # SAVE IMAGE
                    # ======================================

                    if (
                        pose_elapsed
                        >=
                        SETTLE_TIME
                        and
                        now
                        -
                        last_save_time
                        >=
                        CAPTURE_INTERVAL
                    ):

                        if (
                            blur
                            <
                            BLUR_THRESHOLD
                        ):

                            rejected_blurry += 1


                        elif (
                            brightness
                            <
                            35
                            or
                            brightness
                            >
                            225
                        ):

                            rejected_brightness += 1


                        elif (
                            difference
                            <
                            DIFFERENCE_THRESHOLD
                        ):

                            rejected_duplicate += 1


                        else:

                            filename = (
                                f"video_"
                                f"{image_number:05d}.jpg"
                            )

                            output_path = (
                                os.path.join(
                                    OUTPUT_FOLDER,
                                    filename
                                )
                            )

                            cv2.imwrite(
                                output_path,
                                face_crop
                            )

                            previous_saved_face = (
                                face_crop.copy()
                            )

                            image_number += 1

                            saved_count += 1

                            last_save_time = (
                                now
                            )


                # ==========================================
                # DRAW FACE BOX
                # ==========================================

                x = max(
                    0,
                    x
                )

                y = max(
                    0,
                    y
                )

                cv2.rectangle(
                    frame,
                    (
                        x,
                        y
                    ),
                    (
                        x + w,
                        y + h
                    ),
                    (
                        255,
                        255,
                        255
                    ),
                    2
                )


            # ==============================================
            # DISPLAY POSE
            # ==============================================

            if (
                pose_elapsed
                <
                SETTLE_TIME
            ):

                state_text = (
                    f"GET READY: {seconds_left}"
                )

            else:

                state_text = (
                    "CAPTURING"
                )

            # The preparation countdown belongs over the live camera image so
            # it remains visible while the person moves into position.
            if pose_elapsed < SETTLE_TIME:
                countdown_text = str(seconds_left)
                text_size, _ = cv2.getTextSize(
                    countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 6
                )
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(
                    frame, countdown_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6,
                    cv2.LINE_AA,
                )


            # Guidance lives in a separate panel above the camera image,
            # leaving the complete frame unobstructed for enrolment.

            header = np.zeros(
                (
                    ENROLMENT_HEADER_HEIGHT,
                    frame.shape[1],
                    3
                ),
                dtype=np.uint8
            )

            cv2.putText(
                header,
                f"{PERSON_NAME} enrolment - Pose {pose_number}/{len(POSES)}",
                (
                    25,
                    35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                header,
                pose_name,
                (
                    25,
                    75
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                header,
                (
                    state_text
                ),
                (
                    25,
                    112
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                header,
                (
                    f"Saved: {saved_count}   "
                    f"{quality_text}"
                ),
                (
                    25,
                    145
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255
                ),
                1,
                cv2.LINE_AA
            )


            # ==============================================
            # OVERALL PROGRESS
            # ==============================================

            elapsed_total = (
                time.time()
                -
                session_start
            )

            progress = min(
                elapsed_total
                /
                total_duration,
                1.0
            )

            bar_width = (
                frame.shape[1]
                -
                50
            )

            bar_x = 25

            bar_y = (
                frame.shape[0]
                -
                30
            )

            cv2.rectangle(
                frame,
                (
                    bar_x,
                    bar_y
                ),
                (
                    bar_x
                    +
                    bar_width,
                    bar_y
                    +
                    12
                ),
                (
                    255,
                    255,
                    255
                ),
                1
            )

            cv2.rectangle(
                frame,
                (
                    bar_x,
                    bar_y
                ),
                (
                    bar_x
                    +
                    int(
                        bar_width
                        *
                        progress
                    ),
                    bar_y
                    +
                    12
                ),
                (
                    255,
                    255,
                    255
                ),
                -1
            )


            # ==============================================
            # DISPLAY
            # ==============================================

            publish_display_aspect(frame.shape[1], frame.shape[0])

            cv2.imshow(
                window_name,
                np.vstack(
                    (
                        header,
                        frame
                    )
                )
            )


            key = (
                cv2.waitKey(1)
                &
                0xFF
            )


            if (
                key
                ==
                ord("q")
                or
                key
                ==
                27
            ):

                quit_requested = True
                break


        if quit_requested:
            break

# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()


# ==========================================================
# RESULTS
# ==========================================================

print()
print(
    "WAVELY VIDEO ENROLMENT COMPLETE"
)

print(
    "================================"
)

print()

print(
    f"Useful face images saved: {saved_count}"
)

print(
    f"Rejected blurry frames: {rejected_blurry}"
)

print(
    f"Rejected near-duplicates: {rejected_duplicate}"
)

print(
    f"Rejected lighting frames: {rejected_brightness}"
)

print()


if saved_count >= 250:

    print(
        "Excellent dataset."
    )

elif saved_count >= 150:

    print(
        "Good dataset."
    )

elif saved_count >= 75:

    print(
        "Usable dataset, but another enrolment run may improve it."
    )

else:

    print(
        "Dataset is quite small. We should investigate detection quality."
    )

print()
