import cv2
import json
import math
import os
import sys
import queue
import threading
import time
import urllib.request
from datetime import datetime

import mediapipe as mp
import pyttsx3
from wavely_actions import payload_for_gesture
from wavely_devices import configure_speech_output
from wavely_paths import APP_ROOT, CONFIG_DIR as USER_CONFIG_DIR, FACES_DIR, prepare_user_data


# ==========================================================
# SETTINGS
# ==========================================================

prepare_user_data()
BASE_DIR = str(APP_ROOT)
CONFIG_DIR = str(USER_CONFIG_DIR)
DATA_DIR = str(APP_ROOT / "assets")

MODEL_FILE = os.path.join(CONFIG_DIR, "face_model.yml")
LABELS_FILE = os.path.join(CONFIG_DIR, "face_labels.json")
WINDOW_SETTINGS_FILE = os.path.join(CONFIG_DIR, "wavely_window.json")
CAMERA_ASPECT_FILE = os.path.join(CONFIG_DIR, "active_camera_aspect.json")
PROFILE_CASCADE_FILE = str(APP_ROOT / "config" / "haarcascade_profileface.xml")
EMBEDDED_MODE = os.environ.get("WAVELY_EMBEDDED") == "1"
APP_CONFIG_FILE = os.path.join(CONFIG_DIR, "wavely_app_settings.json")
ACTION_CONFIG_FILE = os.path.join(CONFIG_DIR, "home_assistant_actions.json")
try:
    with open(APP_CONFIG_FILE, "r", encoding="utf-8") as config_file:
        APP_CONFIG = json.load(config_file)
except (OSError, json.JSONDecodeError):
    APP_CONFIG = {}
TEST_MODE = os.environ.get("WAVELY_TEST_MODE") == "1"

WEBHOOK_URL = ""

CAMERA_INDEX = int(APP_CONFIG.get("camera_index", 0))

_resolution = str(APP_CONFIG.get("resolution", "1280x720")).split("x")
CAMERA_WIDTH = int(_resolution[0]) if len(_resolution) == 2 else 1280
CAMERA_HEIGHT = int(_resolution[1]) if len(_resolution) == 2 else 720
CAMERA_FPS = 30

HAND_PROCESS_WIDTH = 960
HAND_PROCESS_HEIGHT = 540

MAX_HANDS = 4
HAND_MODEL_COMPLEXITY = 0

HAND_DETECTION_CONFIDENCE = 0.60
HAND_TRACKING_CONFIDENCE = 0.60

MIRROR_FRAME = bool(APP_CONFIG.get("mirror", True))

_last_published_aspect = None


def publish_camera_aspect(width, height):
    """Tell the fixed dashboard panel how to letterbox this camera stream."""
    global _last_published_aspect
    shape = (int(width), int(height))
    if shape == _last_published_aspect or shape[0] <= 0 or shape[1] <= 0:
        return
    _last_published_aspect = shape
    try:
        with open(CAMERA_ASPECT_FILE, "w", encoding="utf-8") as aspect_file:
            json.dump({"width": shape[0], "height": shape[1]}, aspect_file)
    except OSError:
        pass

GESTURE_HOLD_TIME = max(0.25, min(1.2, float(APP_CONFIG.get("gesture_sensitivity", 0.60))))
OPEN_PALM_HOLD_TIME = 1.20
GESTURE_REARM_TIME = 1.20
HAND_STATE_TIMEOUT = 0.80

FACE_SIZE = (200, 200)
FACE_PADDING = 0.28
FACE_RECOGNITION_THRESHOLD = 78.0

IDENTITY_HOLD_TIME = 15.0
FACE_RECOGNITION_INTERVAL = 0.80
FACE_FRAME_PUBLISH_INTERVAL = 0.15

HAND_FACE_HORIZONTAL_WEIGHT = 1.5
HAND_FACE_VERTICAL_WEIGHT = 0.25


# ==========================================================
# FIXED PHYSICAL HAND SLOTS
# ==========================================================

HAND_SIDE_DEAD_ZONE = 0.045
HAND_CONTINUITY_DISTANCE = 0.22
HAND_REACQUIRE_DISTANCE = 0.55
HAND_SLOT_RECENT_TIME = 5.0
HAND_SIDE_MISMATCH_PENALTY = 0.12


# ==========================================================
# FINGER / THUMB TUNING
# ==========================================================

FINGER_PIP_EXTENDED_ANGLE = 148
FINGER_DIP_EXTENDED_ANGLE = 140

THUMB_IP_EXTENDED_ANGLE = 105

THUMB_LENGTH_RATIO = 0.38
THUMB_AWAY_RATIO = 0.42

THUMB_VERTICAL_RATIO = 0.10
THUMB_VERTICAL_VS_HORIZONTAL = 0.45

THUMB_GESTURE_GRACE = 0.70


# ==========================================================
# SWIPE TUNING
# ==========================================================

SWIPE_ARM_TIME = 0.35
SWIPE_ARM_STILLNESS = 0.035

SWIPE_HORIZONTAL_DISTANCE = 0.14
SWIPE_VERTICAL_DISTANCE = 0.13

SWIPE_HORIZONTAL_OFF_AXIS = 0.10
SWIPE_VERTICAL_OFF_AXIS = 0.11

SWIPE_REARM_STILL_TIME = 0.45
SWIPE_REARM_MOVEMENT = 0.025

SWIPE_PALM_LOSS_GRACE = 0.40


# ==========================================================
# TWO-HAND COMBINATIONS
# ==========================================================

COMBO_HOLD_TIME = 0.70

# IMPORTANT:
# Combo must genuinely disappear/change for this long
# before the combo system can trigger again.
COMBO_RELEASE_TIME = 1.00


# With two open palms, move the wrists clearly apart or together before
# the normal open-palms combo completes. Wrist positions are normalized
# (0.0 to 1.0), so this works at the configured camera resolution.
# Tuned from live-camera testing: deliberate movements often take longer
# than the static-combo hold, so accept a smaller, natural wrist change over
# a longer window before falling back to the static open-palms command.
TWO_HAND_MOTION_MIN_DISTANCE_CHANGE = 0.12
TWO_HAND_MOTION_WINDOW = 1.50
TWO_HAND_VERTICAL_MOTION_MIN_CHANGE = 0.10

# A hand at or below the detected hips stays visible but is not an active
# gesture hand. This lets one raised hand work while the other rests low.
# The boundary sits around the natural waist, above the hips.  A hand resting
# by the side therefore stays inactive until it is deliberately raised.
WAIST_GESTURE_ZONE_MARGIN = 0.02

# Show the body landmarks that MediaPipe is using for the waist-zone rule.
SHOW_BODY_SKELETON = True


# ==========================================================
# PERFORMANCE / MEDIAPIPE
# ==========================================================

cv2.setNumThreads(
    min(
        8,
        os.cpu_count() or 4
    )
)

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ==========================================================
# FACE MODEL
# ==========================================================

face_labels = {}
face_recognizer = None
if os.path.isfile(LABELS_FILE) and os.path.isfile(MODEL_FILE):
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as file:
            face_labels = json.load(file)
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(MODEL_FILE)
    except (OSError, ValueError, cv2.error) as error:
        face_labels = {}
        face_recognizer = None
        print(f"Face recognition unavailable until training completes: {error}")
else:
    print("No face model yet. Enrol a person and train the model to enable recognition.")


profile_detector = cv2.CascadeClassifier(
    PROFILE_CASCADE_FILE
)


if profile_detector.empty():

    raise SystemExit(
        "Could not load profile face detector."
    )


pose_tracker = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    enable_segmentation=False,
    min_detection_confidence=0.40,
    min_tracking_confidence=0.40
)


# ==========================================================
# GLOBAL STATE
# ==========================================================

stop_event = threading.Event()
speech_queue = queue.Queue(maxsize=4)

latest_camera_frame = None
latest_camera_frame_lock = threading.Lock()
camera_thread = None

latest_face_frame = None
latest_face_frame_lock = threading.Lock()

face_state_lock = threading.Lock()

current_faces = []
identity_memory = {}

gesture_states = {}
gesture_latches = {}

thumb_stability = {}
swipe_states = {}
combo_states = {}
two_hand_motion_states = {}

# Permanent while WAVELY is running:
# (person, Left)
# (person, Right)
hand_slots = {}

next_slot_id = 1

face_worker_last_ms = 0.0
face_worker_stats_lock = threading.Lock()


# ==========================================================
# GEOMETRY
# ==========================================================

def distance_3d(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2
        +
        (a.y - b.y) ** 2
        +
        (a.z - b.z) ** 2
    )


def point_distance(
    x1,
    y1,
    x2,
    y2
):

    return math.hypot(
        x2 - x1,
        y2 - y1
    )


def joint_angle_3d(
    a,
    b,
    c
):

    ba = (
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    )

    bc = (
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    )


    dot = sum(
        x * y
        for x, y
        in zip(
            ba,
            bc
        )
    )


    mag_a = math.sqrt(
        sum(
            x * x
            for x in ba
        )
    )


    mag_c = math.sqrt(
        sum(
            x * x
            for x in bc
        )
    )


    if (
        mag_a == 0
        or
        mag_c == 0
    ):

        return 0.0


    value = (
        dot
        /
        (
            mag_a
            *
            mag_c
        )
    )


    value = max(
        -1.0,
        min(
            1.0,
            value
        )
    )


    return math.degrees(
        math.acos(
            value
        )
    )


# ==========================================================
# SPEECH
# ==========================================================

def speech_worker():

    try:
        engine = pyttsx3.init()
    except Exception as error:
        print("TTS INITIALISATION ERROR:", error)
        return

    engine.setProperty(
        "rate",
        175
    )

    engine.setProperty(
        "volume",
        1.0
    )
    configure_speech_output(engine, APP_CONFIG.get("audio_output_device_id", ""))


    while not stop_event.is_set():

        try:

            text = speech_queue.get(
                timeout=0.25
            )

        except queue.Empty:

            continue


        try:

            print(
                f"WAVELY SPEAKING: {text}"
            )

            engine.say(
                text
            )

            engine.runAndWait()


        except Exception as error:

            print(
                "SPEECH ERROR:",
                error
            )


        finally:

            speech_queue.task_done()


    try:
        engine.stop()
    except Exception:
        pass


def speak(text):
    try:
        speech_queue.put_nowait(text)
    except queue.Full:
        print("SPEECH BUSY: announcement dropped")


def camera_capture_worker(cap):
    """Own VideoCapture and publish only the newest frame."""
    global latest_camera_frame

    while not stop_event.is_set():
        try:
            ok, frame = cap.read()
        except Exception as error:
            print("CAMERA READ ERROR:", error)
            ok, frame = False, None

        if not ok or frame is None:
            stop_event.wait(0.05)
            continue

        with latest_camera_frame_lock:
            latest_camera_frame = frame


def get_latest_camera_frame():
    with latest_camera_frame_lock:
        if latest_camera_frame is None:
            return None
        return latest_camera_frame.copy()


def speak_current_time():

    now = datetime.now()

    hour = (
        now.strftime("%I")
        .lstrip("0")
    )

    minute = now.strftime(
        "%M"
    )

    period = now.strftime(
        "%p"
    )


    if minute == "00":

        spoken_time = (
            f"{hour} {period}"
        )

    else:

        spoken_time = (
            f"{hour}:{minute} {period}"
        )


    speak(
        f"The time is {spoken_time}"
    )


# ==========================================================
# HOME ASSISTANT
# ==========================================================

def home_assistant_worker(
    person,
    hand,
    gesture
):

    try:
        with open(ACTION_CONFIG_FILE, "r", encoding="utf-8") as actions_file:
            action_config = json.load(actions_file)
    except (OSError, json.JSONDecodeError):
        action_config = {}
    payload = payload_for_gesture(person, hand, gesture, action_config.get("actions", []))
    webhook_url = action_config.get("webhook_url", WEBHOOK_URL).strip()

    if TEST_MODE:
        print(f"TEST MODE: {person} - {hand} Hand - {gesture}")
        return

    if not webhook_url:
        print("Home Assistant action skipped: add a webhook URL in settings.")
        return


    request = urllib.request.Request(
        webhook_url,

        data=json.dumps(
            payload
        ).encode(
            "utf-8"
        ),

        headers={
            "Content-Type":
            "application/json"
        },

        method="POST"
    )


    try:

        with urllib.request.urlopen(
            request,
            timeout=2
        ):

            print(
                f"HOME ASSISTANT: "
                f"{person} - "
                f"{hand} Hand - "
                f"{gesture}"
            )


    except Exception as error:

        print(
            "HOME ASSISTANT ERROR:",
            error
        )


def send_to_home_assistant(
    person,
    hand,
    gesture
):

    threading.Thread(
        target=home_assistant_worker,

        args=(
            person,
            hand,
            gesture
        ),

        daemon=True
    ).start()


# ==========================================================
# FACE HELPERS
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


    side = (
        max(
            w,
            h
        )
        *
        (
            1.0
            +
            FACE_PADDING
        )
    )


    x1 = max(
        0,
        int(
            centre_x
            -
            side / 2
        )
    )


    y1 = max(
        0,
        int(
            centre_y
            -
            side / 2
        )
    )


    x2 = min(
        frame_w,
        int(
            centre_x
            +
            side / 2
        )
    )


    y2 = min(
        frame_h,
        int(
            centre_y
            +
            side / 2
        )
    )


    crop = frame[
        y1:y2,
        x1:x2
    ]


    if crop.size == 0:

        return None


    return crop


def prepare_face(face):

    grey = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )


    grey = cv2.resize(
        grey,
        FACE_SIZE
    )


    return cv2.equalizeHist(
        grey
    )


def detect_profile(frame):

    grey = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = (
        profile_detector.detectMultiScale(
            grey,
            1.1,
            4,
            minSize=(
                70,
                70
            )
        )
    )


    if len(faces):

        return (
            tuple(
                max(
                    faces,
                    key=lambda rect:
                    rect[2]
                    *
                    rect[3]
                )
            ),
            "PROFILE"
        )


    flipped = cv2.flip(
        grey,
        1
    )


    faces = (
        profile_detector.detectMultiScale(
            flipped,
            1.1,
            4,
            minSize=(
                70,
                70
            )
        )
    )


    if len(faces):

        x, y, w, h = max(
            faces,
            key=lambda rect:
            rect[2]
            *
            rect[3]
        )


        return (
            (
                grey.shape[1]
                -
                x
                -
                w,
                y,
                w,
                h
            ),
            "PROFILE FLIPPED"
        )


    return (
        None,
        None
    )


# ==========================================================
# FACE IDENTITY MEMORY
# ==========================================================

def remember_identity(
    name,
    x,
    y,
    w,
    h
):

    with face_state_lock:

        identity_memory[
            name
        ] = {

            "confirmed_time":
            time.monotonic(),

            "centre_x":
            x + w / 2,

            "centre_y":
            y + h / 2,

            "x":
            x,

            "y":
            y,

            "w":
            w,

            "h":
            h

        }


def update_identity_position(
    name,
    x,
    y,
    w,
    h
):

    with face_state_lock:

        if (
            name
            not in
            identity_memory
        ):

            return


        identity_memory[
            name
        ].update(
            {

                "centre_x":
                x + w / 2,

                "centre_y":
                y + h / 2,

                "x":
                x,

                "y":
                y,

                "w":
                w,

                "h":
                h

            }
        )


def get_recent_identities():

    now = (
        time.monotonic()
    )


    with face_state_lock:

        for name in list(
            identity_memory
        ):

            if (
                now
                -
                identity_memory[
                    name
                ][
                    "confirmed_time"
                ]
                >
                IDENTITY_HOLD_TIME
            ):

                del identity_memory[
                    name
                ]


        return {

            name:
            data.copy()

            for name, data
            in identity_memory.items()

        }


def get_current_faces():

    with face_state_lock:

        return [
            face.copy()
            for face
            in current_faces
        ]


def find_nearby_recent_identity(
    x,
    y,
    w,
    h
):

    active = (
        get_recent_identities()
    )


    if not active:

        return None


    centre_x = (
        x + w / 2
    )

    centre_y = (
        y + h / 2
    )


    best_name = None
    best_distance = None


    for name, data in (
        active.items()
    ):

        distance = math.hypot(

            centre_x
            -
            data[
                "centre_x"
            ],

            centre_y
            -
            data[
                "centre_y"
            ]

        )


        if (
            best_distance is None
            or
            distance < best_distance
        ):

            best_name = (
                name
            )

            best_distance = (
                distance
            )


    if (
        best_distance is not None
        and
        best_distance < 300
    ):

        return (
            best_name
        )


    return None


def get_person_face_x_normalised(
    person,
    frame_width
):

    for face in (
        get_current_faces()
    ):

        if (
            face[
                "name"
            ]
            ==
            person
        ):

            return (
                face[
                    "centre_x"
                ]
                /
                frame_width
            )


    recent = (
        get_recent_identities()
    )


    if (
        person
        in
        recent
    ):

        return (
            recent[
                person
            ][
                "centre_x"
            ]
            /
            frame_width
        )


    return None


# ==========================================================
# FACE RECOGNITION
# ==========================================================

def recognise_faces_in_frame(
    frame,
    face_detection
):

    detected = []


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = (
        face_detection.process(
            rgb
        )
    )


    if results.detections:

        frame_h, frame_w = (
            frame.shape[:2]
        )

        publish_camera_aspect(frame_w, frame_h)


        for detection in (
            results.detections
        ):

            box = (
                detection
                .location_data
                .relative_bounding_box
            )


            detected.append(
                {

                    "x":
                    max(
                        0,
                        int(
                            box.xmin
                            *
                            frame_w
                        )
                    ),

                    "y":
                    max(
                        0,
                        int(
                            box.ymin
                            *
                            frame_h
                        )
                    ),

                    "w":
                    int(
                        box.width
                        *
                        frame_w
                    ),

                    "h":
                    int(
                        box.height
                        *
                        frame_h
                    ),

                    "detector":
                    "MEDIAPIPE"

                }
            )


    if not detected:

        rect, kind = (
            detect_profile(
                frame
            )
        )


        if (
            rect
            is not None
        ):

            x, y, w, h = (
                rect
            )


            detected.append(
                {

                    "x":
                    x,

                    "y":
                    y,

                    "w":
                    w,

                    "h":
                    h,

                    "detector":
                    kind

                }
            )


    recognised = []


    for detection in (
        detected
    ):

        x = detection[
            "x"
        ]

        y = detection[
            "y"
        ]

        w = detection[
            "w"
        ]

        h = detection[
            "h"
        ]


        crop = (
            crop_square_face(
                frame,
                x,
                y,
                w,
                h
            )
        )


        if crop is None:

            continue


        label, confidence = (
            face_recognizer.predict(prepare_face(crop))
            if face_recognizer is not None else (None, float("inf"))
        )


        predicted_name = (
            face_labels.get(
                str(
                    label
                ),
                "Unknown"
            )
        )


        if (
            confidence
            <=
            FACE_RECOGNITION_THRESHOLD
        ):

            name = (
                predicted_name
            )

            held = (
                False
            )


            remember_identity(
                name,
                x,
                y,
                w,
                h
            )


        else:

            held_name = (
                find_nearby_recent_identity(
                    x,
                    y,
                    w,
                    h
                )
            )


            if (
                held_name
                is not None
            ):

                name = (
                    held_name
                )

                held = (
                    True
                )


                update_identity_position(
                    name,
                    x,
                    y,
                    w,
                    h
                )


            else:

                name = (
                    "Unknown"
                )

                held = (
                    False
                )


        recognised.append(
            {

                "name":
                name,

                "confidence":
                confidence,

                "held":
                held,

                "x":
                x,

                "y":
                y,

                "w":
                w,

                "h":
                h,

                "centre_x":
                x + w / 2,

                "centre_y":
                y + h / 2,

                "detector":
                detection[
                    "detector"
                ]

            }
        )


    return (
        recognised
    )


def face_worker():

    global current_faces
    global face_worker_last_ms


    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.55
    ) as detector:


        while not stop_event.is_set():

            start = (
                time.monotonic()
            )


            with latest_face_frame_lock:

                if (
                    latest_face_frame
                    is None
                ):

                    frame = (
                        None
                    )

                else:

                    frame = (
                        latest_face_frame.copy()
                    )


            if (
                frame
                is not None
            ):

                try:

                    faces = (
                        recognise_faces_in_frame(
                            frame,
                            detector
                        )
                    )


                    with face_state_lock:

                        current_faces = (
                            faces
                        )


                except Exception as error:

                    print(
                        "FACE WORKER ERROR:",
                        error
                    )


            get_recent_identities()


            elapsed = (
                time.monotonic()
                -
                start
            )


            with face_worker_stats_lock:

                face_worker_last_ms = (
                    elapsed
                    *
                    1000
                )


            stop_event.wait(
                max(
                    0.0,
                    FACE_RECOGNITION_INTERVAL
                    -
                    elapsed
                )
            )


# ==========================================================
# ASSIGN PERSON TO HAND
# ==========================================================

def assign_person_to_hand(
    wrist_x,
    wrist_y,
    frame_width,
    frame_height
):

    candidates = {}


    for name, data in (
        get_recent_identities().items()
    ):

        candidates[
            name
        ] = (

            data[
                "centre_x"
            ],

            data[
                "centre_y"
            ]

        )


    for face in (
        get_current_faces()
    ):

        if (
            face[
                "name"
            ]
            !=
            "Unknown"
        ):

            candidates[
                face[
                    "name"
                ]
            ] = (

                face[
                    "centre_x"
                ],

                face[
                    "centre_y"
                ]

            )


    if not candidates:

        return (
            "Unknown"
        )


    if (
        len(
            candidates
        )
        ==
        1
    ):

        return (
            next(
                iter(
                    candidates
                )
            )
        )


    hand_x = (
        wrist_x
        *
        frame_width
    )

    hand_y = (
        wrist_y
        *
        frame_height
    )


    return min(

        candidates,

        key=lambda name:
        (

            abs(
                hand_x
                -
                candidates[
                    name
                ][0]
            )
            *
            HAND_FACE_HORIZONTAL_WEIGHT

            +

            abs(
                hand_y
                -
                candidates[
                    name
                ][1]
            )
            *
            HAND_FACE_VERTICAL_WEIGHT

        )

    )


# ==========================================================
# FIXED LEFT / RIGHT HAND SLOTS
# ==========================================================

def get_hand_slot(
    person,
    hand
):

    return (
        hand_slots.get(
            (
                person,
                hand
            )
        )
    )


def create_or_update_hand_slot(
    person,
    hand,
    x,
    y
):

    global next_slot_id


    key = (
        person,
        hand
    )

    now = (
        time.monotonic()
    )


    if (
        key
        not in
        hand_slots
    ):

        hand_slots[
            key
        ] = {

            "slot_id":
            next_slot_id,

            "person":
            person,

            "hand":
            hand,

            "x":
            x,

            "y":
            y,

            "last_seen":
            now

        }


        print(
            f"HAND SLOT CREATED: "
            f"{person} {hand} "
            f"(slot {next_slot_id})"
        )


        next_slot_id += 1


    else:

        hand_slots[
            key
        ][
            "x"
        ] = x


        hand_slots[
            key
        ][
            "y"
        ] = y


        hand_slots[
            key
        ][
            "last_seen"
        ] = now


    return (
        hand_slots[
            key
        ]
    )


def physical_side_from_face(
    wrist_x,
    face_x
):

    if face_x is None:

        return None


    if (
        wrist_x
        <
        face_x
        -
        HAND_SIDE_DEAD_ZONE
    ):

        if MIRROR_FRAME:

            return (
                "Left"
            )

        return (
            "Right"
        )


    if (
        wrist_x
        >
        face_x
        +
        HAND_SIDE_DEAD_ZONE
    ):

        if MIRROR_FRAME:

            return (
                "Right"
            )

        return (
            "Left"
        )


    return None


def slot_cost(
    item,
    hand,
    face_x
):

    slot = (
        get_hand_slot(
            item[
                "person"
            ],
            hand
        )
    )


    if (
        slot
        is not None
    ):

        age = (
            time.monotonic()
            -
            slot[
                "last_seen"
            ]
        )


        if (
            age
            <=
            HAND_SLOT_RECENT_TIME
        ):

            cost = (
                point_distance(
                    item[
                        "wrist_x"
                    ],
                    item[
                        "wrist_y"
                    ],
                    slot[
                        "x"
                    ],
                    slot[
                        "y"
                    ]
                )
            )


        else:

            cost = (
                0.35
            )


    else:

        cost = (
            0.30
        )


    side_guess = (
        physical_side_from_face(
            item[
                "wrist_x"
            ],
            face_x
        )
    )


    if (
        side_guess
        is not None

        and

        side_guess
        !=
        hand
    ):

        cost += (
            HAND_SIDE_MISMATCH_PENALTY
        )


    return (
        cost
    )


def assign_hand_slots_for_person(
    items,
    frame_width
):

    if not items:

        return


    person = (
        items[0][
            "person"
        ]
    )


    face_x = (
        get_person_face_x_normalised(
            person,
            frame_width
        )
    )


    # Only keep strongest two hand detections per person.
    if (
        len(
            items
        )
        >
        2
    ):

        ordered_by_confidence = sorted(

            items,

            key=lambda item:
            item[
                "raw_hand_confidence"
            ],

            reverse=True

        )


        active_items = (
            ordered_by_confidence[
                :2
            ]
        )


        for item in (
            ordered_by_confidence[
                2:
            ]
        ):

            item[
                "ignore"
            ] = (
                True
            )


    else:

        active_items = (
            items
        )


    # ======================================================
    # TWO HANDS VISIBLE
    # ======================================================

    if (
        len(
            active_items
        )
        ==
        2
    ):

        first = (
            active_items[0]
        )

        second = (
            active_items[1]
        )


        left_slot = (
            get_hand_slot(
                person,
                "Left"
            )
        )

        right_slot = (
            get_hand_slot(
                person,
                "Right"
            )
        )


        # First time both hands are visible.
        if (
            left_slot
            is None

            and

            right_slot
            is None
        ):

            ordered = sorted(

                active_items,

                key=lambda item:
                item[
                    "wrist_x"
                ]

            )


            if MIRROR_FRAME:

                ordered[0][
                    "stable_hand"
                ] = (
                    "Left"
                )

                ordered[1][
                    "stable_hand"
                ] = (
                    "Right"
                )


            else:

                ordered[0][
                    "stable_hand"
                ] = (
                    "Right"
                )

                ordered[1][
                    "stable_hand"
                ] = (
                    "Left"
                )


        else:

            direct_cost = (

                slot_cost(
                    first,
                    "Left",
                    face_x
                )

                +

                slot_cost(
                    second,
                    "Right",
                    face_x
                )

            )


            crossed_cost = (

                slot_cost(
                    first,
                    "Right",
                    face_x
                )

                +

                slot_cost(
                    second,
                    "Left",
                    face_x
                )

            )


            if (
                direct_cost
                <=
                crossed_cost
            ):

                first[
                    "stable_hand"
                ] = (
                    "Left"
                )

                second[
                    "stable_hand"
                ] = (
                    "Right"
                )


            else:

                first[
                    "stable_hand"
                ] = (
                    "Right"
                )

                second[
                    "stable_hand"
                ] = (
                    "Left"
                )


        for item in (
            active_items
        ):

            create_or_update_hand_slot(
                person,
                item[
                    "stable_hand"
                ],
                item[
                    "wrist_x"
                ],
                item[
                    "wrist_y"
                ]
            )


        return


    # ======================================================
    # ONE HAND VISIBLE
    # ======================================================

    item = (
        active_items[0]
    )

    now = (
        time.monotonic()
    )

    recent_candidates = []


    for hand in (
        "Left",
        "Right"
    ):

        slot = (
            get_hand_slot(
                person,
                hand
            )
        )


        if (
            slot
            is None
        ):

            continue


        age = (
            now
            -
            slot[
                "last_seen"
            ]
        )


        if (
            age
            >
            HAND_SLOT_RECENT_TIME
        ):

            continue


        distance = (
            point_distance(
                item[
                    "wrist_x"
                ],
                item[
                    "wrist_y"
                ],
                slot[
                    "x"
                ],
                slot[
                    "y"
                ]
            )
        )


        recent_candidates.append(
            (
                distance,
                hand
            )
        )


    chosen = None


    if recent_candidates:

        (
            nearest_distance,
            nearest_hand
        ) = min(
            recent_candidates,
            key=lambda entry:
            entry[0]
        )


        if (
            nearest_distance
            <=
            HAND_REACQUIRE_DISTANCE
        ):

            chosen = (
                nearest_hand
            )


    if (
        chosen
        is None
    ):

        chosen = (
            physical_side_from_face(
                item[
                    "wrist_x"
                ],
                face_x
            )
        )


    if (
        chosen
        is None
    ):

        chosen = (
            item[
                "raw_hand"
            ]
        )


    item[
        "stable_hand"
    ] = (
        chosen
    )


    create_or_update_hand_slot(
        person,
        chosen,
        item[
            "wrist_x"
        ],
        item[
            "wrist_y"
        ]
    )


def assign_hand_slots(
    items,
    frame_width
):

    groups = {}


    for item in (
        items
    ):

        item[
            "ignore"
        ] = (
            False
        )


        groups.setdefault(
            item[
                "person"
            ],
            []
        ).append(
            item
        )


    for person_items in (
        groups.values()
    ):

        if (
            person_items[0][
                "person"
            ]
            ==
            "Unknown"
        ):

            for item in (
                person_items
            ):

                item[
                    "stable_hand"
                ] = (
                    item[
                        "raw_hand"
                    ]
                )


            continue


        assign_hand_slots_for_person(
            person_items,
            frame_width
        )


# ==========================================================
# FINGER / GESTURE CLASSIFICATION
# ==========================================================

def finger_is_extended(
    landmarks,
    mcp,
    pip,
    dip,
    tip
):

    return (

        joint_angle_3d(
            landmarks[
                mcp
            ],
            landmarks[
                pip
            ],
            landmarks[
                dip
            ]
        )
        >=
        FINGER_PIP_EXTENDED_ANGLE

        and

        joint_angle_3d(
            landmarks[
                pip
            ],
            landmarks[
                dip
            ],
            landmarks[
                tip
            ]
        )
        >=
        FINGER_DIP_EXTENDED_ANGLE

    )


def get_finger_states(
    hand_landmarks
):

    landmarks = (
        hand_landmarks.landmark
    )


    return {

        "index":
        finger_is_extended(
            landmarks,
            5,
            6,
            7,
            8
        ),

        "middle":
        finger_is_extended(
            landmarks,
            9,
            10,
            11,
            12
        ),

        "ring":
        finger_is_extended(
            landmarks,
            13,
            14,
            15,
            16
        ),

        "pinky":
        finger_is_extended(
            landmarks,
            17,
            18,
            19,
            20
        )

    }


def classify_gesture(
    hand_landmarks
):

    landmarks = (
        hand_landmarks.landmark
    )


    fingers = (
        get_finger_states(
            hand_landmarks
        )
    )


    index = (
        fingers[
            "index"
        ]
    )

    middle = (
        fingers[
            "middle"
        ]
    )

    ring = (
        fingers[
            "ring"
        ]
    )

    pinky = (
        fingers[
            "pinky"
        ]
    )


    open_count = sum(
        map(
            int,
            (
                index,
                middle,
                ring,
                pinky
            )
        )
    )


    palm_size = (
        distance_3d(
            landmarks[0],
            landmarks[9]
        )
    )


    thumb_length = (
        distance_3d(
            landmarks[2],
            landmarks[4]
        )
    )


    thumb_away = (
        distance_3d(
            landmarks[4],
            landmarks[5]
        )
    )


    thumb_angle = (
        joint_angle_3d(
            landmarks[2],
            landmarks[3],
            landmarks[4]
        )
    )


    thumb_extended = (

        palm_size > 0

        and

        thumb_length
        >
        palm_size
        *
        THUMB_LENGTH_RATIO

        and

        thumb_away
        >
        palm_size
        *
        THUMB_AWAY_RATIO

        and

        thumb_angle
        >=
        THUMB_IP_EXTENDED_ANGLE

    )


    thumb_dx = (
        landmarks[4].x
        -
        landmarks[2].x
    )


    thumb_dy = (
        landmarks[4].y
        -
        landmarks[2].y
    )


    vertical_enough = (

        abs(
            thumb_dy
        )

        >

        abs(
            thumb_dx
        )
        *
        THUMB_VERTICAL_VS_HORIZONTAL

    )


    vertical_threshold = max(

        0.018,

        palm_size
        *
        THUMB_VERTICAL_RATIO

    )


    obvious_peace = (

        index
        and
        middle
        and
        not ring
        and
        not pinky

    )


    thumb_fingers_ok = (

        open_count
        <=
        2

        and

        not index

        and

        not obvious_peace

    )


    # ======================================================
    # THUMBS DOWN
    # ======================================================

    if (
        thumb_fingers_ok

        and

        thumb_extended

        and

        vertical_enough

        and

        thumb_dy
        >
        vertical_threshold
    ):

        return (
            "THUMBS_DOWN"
        )


    # ======================================================
    # THUMBS UP
    # ======================================================

    if (
        thumb_fingers_ok

        and

        thumb_extended

        and

        vertical_enough

        and

        thumb_dy
        <
        -
        vertical_threshold
    ):

        return (
            "THUMBS_UP"
        )


    if (
        obvious_peace
    ):

        return (
            "PEACE"
        )


    if (
        index
        and
        not middle
        and
        not ring
        and
        not pinky
    ):

        return (
            "POINT"
        )


    if (
        open_count
        ==
        4
    ):

        return (
            "OPEN_PALM"
        )


    if (
        open_count
        ==
        0
    ):

        return (
            "FIST"
        )


    return (
        "OTHER"
    )


# ==========================================================
# THUMB TEMPORAL STABILISER
# ==========================================================

def stabilise_thumb_gesture(
    person,
    hand,
    raw_gesture
):

    key = (
        person,
        hand
    )

    now = (
        time.monotonic()
    )


    state = (
        thumb_stability.setdefault(
            key,
            {

                "gesture":
                None,

                "last_seen":
                0.0

            }
        )
    )


    if (
        raw_gesture
        in
        (
            "THUMBS_UP",
            "THUMBS_DOWN"
        )
    ):

        state[
            "gesture"
        ] = (
            raw_gesture
        )

        state[
            "last_seen"
        ] = (
            now
        )


        return (
            raw_gesture
        )


    held = (
        state[
            "gesture"
        ]
    )


    if (
        held

        and

        now
        -
        state[
            "last_seen"
        ]
        <=
        THUMB_GESTURE_GRACE

        and

        raw_gesture
        in
        (
            "FIST",
            "OTHER",
            "POINT"
        )
    ):

        return (
            held
        )


    if (
        now
        -
        state[
            "last_seen"
        ]
        >
        THUMB_GESTURE_GRACE
    ):

        state[
            "gesture"
        ] = (
            None
        )


    return (
        raw_gesture
    )


# ==========================================================
# TWO-HAND COMBINATION SYSTEM
# ==========================================================

def get_combo_name(
    left_gesture,
    right_gesture
):

    if (
        left_gesture
        is None

        or

        right_gesture
        is None
    ):

        return None


    if (
        left_gesture
        ==
        "OPEN_PALM"

        and

        right_gesture
        ==
        "OPEN_PALM"
    ):

        return (
            "BOTH_OPEN_PALMS"
        )


    if (
        left_gesture
        ==
        "FIST"

        and

        right_gesture
        ==
        "FIST"
    ):

        return (
            "BOTH_FISTS"
        )


    if (
        left_gesture
        ==
        "THUMBS_UP"

        and

        right_gesture
        ==
        "THUMBS_UP"
    ):

        return (
            "BOTH_THUMBS_UP"
        )


    if (
        left_gesture
        ==
        "THUMBS_DOWN"

        and

        right_gesture
        ==
        "THUMBS_DOWN"
    ):

        return (
            "BOTH_THUMBS_DOWN"
        )


    if (
        left_gesture
        ==
        "OPEN_PALM"

        and

        right_gesture
        ==
        "FIST"
    ):

        return (
            "LEFT_OPEN_RIGHT_FIST"
        )


    if (
        left_gesture
        ==
        "FIST"

        and

        right_gesture
        ==
        "OPEN_PALM"
    ):

        return (
            "LEFT_FIST_RIGHT_OPEN"
        )


    return None


def update_two_hand_motion_state(
    person,
    left_gesture,
    right_gesture,
    left_wrist_x,
    right_wrist_x,
    left_wrist_y,
    right_wrist_y,
    both_hands_visible
):

    now = (
        time.monotonic()
    )


    state = (
        two_hand_motion_states.setdefault(
            person,
            {

                "baseline_distance":
                None,

                "baseline_y":
                None,

                "start":
                None,

                "latched":
                False,

                "name":
                None

            }
        )
    )


    both_open_palms = (
        both_hands_visible

        and

        left_gesture
        ==
        "OPEN_PALM"

        and

        right_gesture
        ==
        "OPEN_PALM"
    )


    if (
        not both_open_palms
    ):

        state[
            "baseline_distance"
        ] = (
            None
        )

        state[
            "baseline_y"
        ] = (
            None
        )

        state[
            "start"
        ] = (
            None
        )

        state[
            "latched"
        ] = (
            False
        )

        state[
            "name"
        ] = (
            None
        )

        return (
            None,
            0.0
        )


    distance = abs(
        right_wrist_x
        -
        left_wrist_x
    )

    average_y = (
        left_wrist_y
        +
        right_wrist_y
    ) / 2.0


    if (
        state[
            "latched"
        ]
    ):

        distance_change = (
            distance
            -
            state[
                "baseline_distance"
            ]
        )

        vertical_change = (
            average_y
            -
            state[
                "baseline_y"
            ]
        )


        if (
            state["name"] != "HANDS_DOWN"
            and vertical_change >= TWO_HAND_VERTICAL_MOTION_MIN_CHANGE
        ):
            state["baseline_distance"] = distance
            state["baseline_y"] = average_y
            state["name"] = "HANDS_DOWN"
            return ("HANDS_DOWN", 1.0)


        if (
            state["name"] != "HANDS_UP"
            and vertical_change <= -TWO_HAND_VERTICAL_MOTION_MIN_CHANGE
        ):
            state["baseline_distance"] = distance
            state["baseline_y"] = average_y
            state["name"] = "HANDS_UP"
            return ("HANDS_UP", 1.0)


        if (
            state[
                "name"
            ]
            !=
            "CLOSE_HANDS"

            and

            distance_change
            <=
            -TWO_HAND_MOTION_MIN_DISTANCE_CHANGE
        ):

            state[
                "baseline_distance"
            ] = (
                distance
            )

            state[
                "name"
            ] = (
                "CLOSE_HANDS"
            )


            return (
                "CLOSE_HANDS",
                1.0
            )


        if (
            state[
                "name"
            ]
            !=
            "SPREAD_HANDS"

            and

            distance_change
            >=
            TWO_HAND_MOTION_MIN_DISTANCE_CHANGE
        ):

            state[
                "baseline_distance"
            ] = (
                distance
            )

            state[
                "name"
            ] = (
                "SPREAD_HANDS"
            )


            return (
                "SPREAD_HANDS",
                1.0
            )

        return (
            state[
                "name"
            ],
            1.0
        )


    if (
        state[
            "baseline_distance"
        ]
        is None
    ):

        state[
            "baseline_distance"
        ] = (
            distance
        )

        state[
            "baseline_y"
        ] = (
            average_y
        )

        state[
            "start"
        ] = (
            now
        )

        return (
            None,
            0.0
        )


    elapsed = (
        now
        -
        state[
            "start"
        ]
    )


    distance_change = (
        distance
        -
        state[
            "baseline_distance"
        ]
    )

    vertical_change = (
        average_y
        -
        state[
            "baseline_y"
        ]
    )


    if (
        vertical_change
        <=
        -TWO_HAND_VERTICAL_MOTION_MIN_CHANGE

        and

        elapsed
        <=
        TWO_HAND_MOTION_WINDOW
    ):

        state["latched"] = True
        state["name"] = "HANDS_UP"
        state["baseline_distance"] = distance
        state["baseline_y"] = average_y

        return (
            "HANDS_UP",
            1.0
        )


    if (
        vertical_change
        >=
        TWO_HAND_VERTICAL_MOTION_MIN_CHANGE

        and

        elapsed
        <=
        TWO_HAND_MOTION_WINDOW
    ):

        state["latched"] = True
        state["name"] = "HANDS_DOWN"
        state["baseline_distance"] = distance
        state["baseline_y"] = average_y

        return (
            "HANDS_DOWN",
            1.0
        )


    if (
        distance_change
        >=
        TWO_HAND_MOTION_MIN_DISTANCE_CHANGE

        and

        elapsed
        <=
        TWO_HAND_MOTION_WINDOW
    ):

        state[
            "latched"
        ] = (
            True
        )

        state[
            "name"
        ] = (
            "SPREAD_HANDS"
        )

        state[
            "baseline_distance"
        ] = (
            distance
        )


        return (
            "SPREAD_HANDS",
            1.0
        )


    if (
        distance_change
        <=
        -TWO_HAND_MOTION_MIN_DISTANCE_CHANGE

        and

        elapsed
        <=
        TWO_HAND_MOTION_WINDOW
    ):

        state[
            "latched"
        ] = (
            True
        )

        state[
            "name"
        ] = (
            "CLOSE_HANDS"
        )

        state[
            "baseline_distance"
        ] = (
            distance
        )


        return (
            "CLOSE_HANDS",
            1.0
        )


    # A stationary open-palms pose remains the normal static combo.
    # Do not keep an old baseline alive after that combo has had time to fire.
    if (
        elapsed
        >
        TWO_HAND_MOTION_WINDOW
    ):

        state[
            "baseline_distance"
        ] = (
            distance
        )

        state[
            "start"
        ] = (
            now
        )


    return (
        None,
        min(
            max(
                abs(
                    distance_change
                ),
                0.0
            )
            /
            TWO_HAND_MOTION_MIN_DISTANCE_CHANGE,
            1.0
        )
    )


def update_combo_state(
    person,
    left_gesture,
    right_gesture,
    both_hands_visible,
    motion_candidate=None
):

    now = (
        time.monotonic()
    )


    if (
        motion_candidate
        is not None
    ):

        candidate = (
            motion_candidate
        )


    elif (
        both_hands_visible
    ):

        candidate = (
            get_combo_name(
                left_gesture,
                right_gesture
            )
        )


    else:

        candidate = (
            None
        )


    state = (
        combo_states.setdefault(
            person,
            {

                "candidate":
                None,

                "start":
                None,

                "latched":
                None,

                "release_start":
                None

            }
        )
    )


    # ======================================================
    # PREVIOUS COMBO IS LATCHED
    # ======================================================

    if (
        state[
            "latched"
        ]
        is not None
    ):

        # Motion gestures are directional. A clear reversal between open
        # palms is a new command, so do not make the user lower both hands
        # and wait through the static-combo release period first.
        if (
            motion_candidate
            in
            (
                "SPREAD_HANDS",
                "CLOSE_HANDS",
                "HANDS_UP",
                "HANDS_DOWN"
            )

            and

            motion_candidate
            !=
            state[
                "latched"
            ]
        ):

            state[
                "latched"
            ] = (
                motion_candidate
            )

            state[
                "candidate"
            ] = (
                motion_candidate
            )

            state[
                "start"
            ] = (
                now
            )

            state[
                "release_start"
            ] = (
                None
            )


            print(
                f"COMBO TRIGGERED: "
                f"{person} - "
                f"{motion_candidate}"
            )


            send_to_home_assistant(
                person,
                "Both",
                motion_candidate
            )


            return (
                motion_candidate,
                1.0,
                True
            )

        # Still genuinely holding the exact same combo.
        if (
            both_hands_visible

            and

            candidate
            ==
            state[
                "latched"
            ]
        ):

            state[
                "release_start"
            ] = (
                None
            )


            return (
                state[
                    "latched"
                ],
                1.0,
                True
            )


        # Combo is broken or changed.
        if (
            state[
                "release_start"
            ]
            is None
        ):

            state[
                "release_start"
            ] = (
                now
            )


        release_elapsed = (
            now
            -
            state[
                "release_start"
            ]
        )


        # IMPORTANT:
        # Keep two-hand blocking active for the entire
        # release period. This prevents accidental individual
        # fist/thumb/swipe actions during transitions.
        if (
            release_elapsed
            <
            COMBO_RELEASE_TIME
        ):

            return (
                state[
                    "latched"
                ],
                1.0,
                True
            )


        print(
            f"COMBO REARMED: "
            f"{person} - "
            f"{state['latched']}"
        )


        state[
            "latched"
        ] = (
            None
        )


        state[
            "candidate"
        ] = (
            None
        )


        state[
            "start"
        ] = (
            None
        )


        state[
            "release_start"
        ] = (
            None
        )


    # ======================================================
    # ONLY ONE HAND VISIBLE
    # ======================================================

    if (
        not both_hands_visible
    ):

        state[
            "candidate"
        ] = (
            None
        )

        state[
            "start"
        ] = (
            None
        )


        return (
            None,
            0.0,
            False
        )


    # ======================================================
    # TWO HANDS ARE VISIBLE, BUT NO VALID COMBO YET
    #
    # THIS STILL BLOCKS ALL SINGLE-HAND ACTIONS.
    # ======================================================

    if (
        candidate
        is None
    ):

        state[
            "candidate"
        ] = (
            None
        )

        state[
            "start"
        ] = (
            None
        )


        return (
            None,
            0.0,
            True
        )


    # ======================================================
    # NEW COMBO CANDIDATE
    # ======================================================

    if (
        candidate
        !=
        state[
            "candidate"
        ]
    ):

        state[
            "candidate"
        ] = (
            candidate
        )

        state[
            "start"
        ] = (
            now
            -
            (
                COMBO_HOLD_TIME
                if motion_candidate is not None
                else 0.0
            )
        )


        return (
            candidate,
            0.0,
            True
        )


    if (
        state[
            "start"
        ]
        is None
    ):

        state[
            "start"
        ] = (
            now
        )


    elapsed = (
        now
        -
        state[
            "start"
        ]
    )


    progress = min(
        elapsed
        /
        COMBO_HOLD_TIME,
        1.0
    )


    # ======================================================
    # COMBO TRIGGER
    # ======================================================

    if (
        progress
        >=
        1.0

        and

        state[
            "latched"
        ]
        is None
    ):

        state[
            "latched"
        ] = (
            candidate
        )


        state[
            "release_start"
        ] = (
            None
        )


        print(
            f"COMBO TRIGGERED: "
            f"{person} - "
            f"{candidate}"
        )


        send_to_home_assistant(
            person,
            "Both",
            candidate
        )


    return (
        candidate,
        progress,
        True
    )


def clear_swipe_state(
    person,
    hand
):

    swipe_states.pop(
        f"{person}|{hand}",
        None
    )


# ==========================================================
# SWIPE STATE MACHINE
# ==========================================================

def swipe_key(
    person,
    hand
):

    return (
        f"{person}|{hand}"
    )


def get_swipe_state(
    person,
    hand
):

    key = (
        swipe_key(
            person,
            hand
        )
    )


    if (
        key
        not in
        swipe_states
    ):

        swipe_states[
            key
        ] = {

            "mode":
            "WAITING_STILL",

            "last_x":
            None,

            "last_y":
            None,

            "still_start":
            None,

            "anchor_x":
            None,

            "anchor_y":
            None,

            "rearm_start":
            None,

            "last_open_palm_time":
            0.0

        }


    return (
        swipe_states[
            key
        ]
    )


def reset_swipe_state(
    state,
    now
):

    state.update(
        {

            "mode":
            "WAITING_STILL",

            "last_x":
            None,

            "last_y":
            None,

            "still_start":
            now,

            "anchor_x":
            None,

            "anchor_y":
            None,

            "rearm_start":
            None

        }
    )


def process_swipe(
    person,
    hand,
    x,
    y,
    raw_open_palm
):

    state = (
        get_swipe_state(
            person,
            hand
        )
    )


    now = (
        time.monotonic()
    )


    if (
        raw_open_palm
    ):

        state[
            "last_open_palm_time"
        ] = (
            now
        )


    palm_recent = (

        raw_open_palm

        or

        now
        -
        state[
            "last_open_palm_time"
        ]
        <=
        SWIPE_PALM_LOSS_GRACE

    )


    if (
        not palm_recent
    ):

        reset_swipe_state(
            state,
            now
        )


        return (
            None,
            "WAITING_STILL",
            False
        )


    if (
        state[
            "last_x"
        ]
        is None
    ):

        state[
            "last_x"
        ] = (
            x
        )

        state[
            "last_y"
        ] = (
            y
        )


        if (
            state[
                "still_start"
            ]
            is None
        ):

            state[
                "still_start"
            ] = (
                now
            )


        return (
            None,
            state[
                "mode"
            ],
            True
        )


    step_distance = (
        point_distance(
            state[
                "last_x"
            ],
            state[
                "last_y"
            ],
            x,
            y
        )
    )


    state[
        "last_x"
    ] = (
        x
    )

    state[
        "last_y"
    ] = (
        y
    )


    # ======================================================
    # WAITING FOR STATIONARY OPEN PALM
    # ======================================================

    if (
        state[
            "mode"
        ]
        ==
        "WAITING_STILL"
    ):

        if (
            step_distance
            <=
            SWIPE_ARM_STILLNESS
        ):

            if (
                state[
                    "still_start"
                ]
                is None
            ):

                state[
                    "still_start"
                ] = (
                    now
                )


            if (
                now
                -
                state[
                    "still_start"
                ]
                >=
                SWIPE_ARM_TIME
            ):

                state[
                    "mode"
                ] = (
                    "ARMED"
                )

                state[
                    "anchor_x"
                ] = (
                    x
                )

                state[
                    "anchor_y"
                ] = (
                    y
                )


                print(
                    f"SWIPE ARMED: "
                    f"{person} - "
                    f"{hand} Hand"
                )


        else:

            state[
                "still_start"
            ] = (
                now
            )


        return (
            None,
            state[
                "mode"
            ],
            True
        )


    # ======================================================
    # SWIPE READY
    # ======================================================

    if (
        state[
            "mode"
        ]
        ==
        "ARMED"
    ):

        dx = (
            x
            -
            state[
                "anchor_x"
            ]
        )

        dy = (
            y
            -
            state[
                "anchor_y"
            ]
        )


        if (
            abs(
                dx
            )
            >=
            SWIPE_HORIZONTAL_DISTANCE

            and

            abs(
                dy
            )
            <=
            SWIPE_HORIZONTAL_OFF_AXIS
        ):

            if (
                dx > 0
            ):

                gesture = (
                    "SWIPE_RIGHT"
                )

            else:

                gesture = (
                    "SWIPE_LEFT"
                )


            state[
                "mode"
            ] = (
                "LOCKED"
            )

            state[
                "rearm_start"
            ] = (
                None
            )


            return (
                gesture,
                "LOCKED",
                True
            )


        if (
            abs(
                dy
            )
            >=
            SWIPE_VERTICAL_DISTANCE

            and

            abs(
                dx
            )
            <=
            SWIPE_VERTICAL_OFF_AXIS
        ):

            if (
                dy < 0
            ):

                gesture = (
                    "SWIPE_UP"
                )

            else:

                gesture = (
                    "SWIPE_DOWN"
                )


            state[
                "mode"
            ] = (
                "LOCKED"
            )

            state[
                "rearm_start"
            ] = (
                None
            )


            return (
                gesture,
                "LOCKED",
                True
            )


        return (
            None,
            "ARMED",
            True
        )


    # ======================================================
    # LOCKED AFTER SWIPE
    # ======================================================

    if (
        state[
            "mode"
        ]
        ==
        "LOCKED"
    ):

        if (
            step_distance
            <=
            SWIPE_REARM_MOVEMENT
        ):

            if (
                state[
                    "rearm_start"
                ]
                is None
            ):

                state[
                    "rearm_start"
                ] = (
                    now
                )


            if (
                now
                -
                state[
                    "rearm_start"
                ]
                >=
                SWIPE_REARM_STILL_TIME
            ):

                state[
                    "mode"
                ] = (
                    "ARMED"
                )

                state[
                    "anchor_x"
                ] = (
                    x
                )

                state[
                    "anchor_y"
                ] = (
                    y
                )

                state[
                    "rearm_start"
                ] = (
                    None
                )


                print(
                    f"SWIPE REARMED: "
                    f"{person} - "
                    f"{hand} Hand"
                )


        else:

            state[
                "rearm_start"
            ] = (
                None
            )


        return (
            None,
            state[
                "mode"
            ],
            True
        )


    return (
        None,
        state[
            "mode"
        ],
        True
    )


# ==========================================================
# STATIC GESTURE HOLD / ANTI-REPEAT
# ==========================================================

def gesture_state_key(
    person,
    hand
):

    return (
        f"{person}|{hand}"
    )


def gesture_required_hold(
    gesture
):

    if (
        gesture
        ==
        "OPEN_PALM"
    ):

        return (
            OPEN_PALM_HOLD_TIME
        )


    return (
        GESTURE_HOLD_TIME
    )


def update_gesture_state(
    person,
    hand,
    gesture
):

    key = (
        gesture_state_key(
            person,
            hand
        )
    )

    now = (
        time.monotonic()
    )


    if (
        key
        not in
        gesture_states
    ):

        gesture_states[
            key
        ] = {

            "gesture":
            gesture,

            "start":
            now,

            "last_seen":
            now

        }


        return (
            key,
            0.0,
            False
        )


    state = (
        gesture_states[
            key
        ]
    )


    # A tracking gap breaks the hold timer.  Gesture latches remain
    # independent, so a previously fired command still cannot repeat.
    if (
        now
        -
        state[
            "last_seen"
        ]
        >
        HAND_STATE_TIMEOUT
    ):

        gesture_states[
            key
        ] = {

            "gesture":
            gesture,

            "start":
            now,

            "last_seen":
            now

        }

        return (
            key,
            0.0,
            False
        )


    state[
        "last_seen"
    ] = (
        now
    )


    if (
        gesture
        !=
        state[
            "gesture"
        ]
    ):

        state[
            "gesture"
        ] = (
            gesture
        )

        state[
            "start"
        ] = (
            now
        )


        return (
            key,
            0.0,
            False
        )


    elapsed = (
        now
        -
        state[
            "start"
        ]
    )


    required = (
        gesture_required_hold(
            gesture
        )
    )


    progress = min(
        elapsed
        /
        required,
        1.0
    )


    stable = (
        gesture
        !=
        "OTHER"

        and

        elapsed
        >=
        required
    )


    return (
        key,
        progress,
        stable
    )


def cleanup_hand_states(
    seen_keys
):

    now = (
        time.monotonic()
    )


    for key in list(
        gesture_states
    ):

        if (
            key
            not in
            seen_keys

            and

            now
            -
            gesture_states[
                key
            ][
                "last_seen"
            ]
            >
            HAND_STATE_TIMEOUT
        ):

            del gesture_states[
                key
            ]


def update_gesture_latches(
    observed_gestures
):

    now = (
        time.monotonic()
    )


    for key in list(
        gesture_latches
    ):

        if (
            key
            in
            observed_gestures
        ):

            gesture_latches[
                key
            ] = (
                now
            )


        elif (
            now
            -
            gesture_latches[
                key
            ]
            >=
            GESTURE_REARM_TIME
        ):

            person, hand, gesture = (
                key
            )


            print(
                f"REARMED: "
                f"{person} - "
                f"{hand} Hand - "
                f"{gesture}"
            )


            del gesture_latches[
                key
            ]


def try_trigger_gesture(
    person,
    hand,
    gesture,
    use_latch=True
):

    if (
        person
        ==
        "Unknown"
    ):

        return


    if (
        use_latch
    ):

        key = (
            person,
            hand,
            gesture
        )


        if (
            key
            in
            gesture_latches
        ):

            return


        gesture_latches[
            key
        ] = (
            time.monotonic()
        )


    print(
        f"TRIGGERED: "
        f"{person} - "
        f"{hand} Hand - "
        f"{gesture}"
    )


    send_to_home_assistant(
        person,
        hand,
        gesture
    )


    if (
        gesture
        ==
        "POINT"
    ):

        speak_current_time()


    else:

        print(
            f"ACTION: "
            f"{person} "
            f"{gesture}"
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

    raise SystemExit(
        "Could not open camera."
    )

camera_thread = threading.Thread(
    target=camera_capture_worker,
    args=(cap,),
    name="camera-capture",
    daemon=True,
)
camera_thread.start()


# ==========================================================
# THREADS / WINDOW
# ==========================================================

threading.Thread(
    target=speech_worker,
    daemon=True
).start()


threading.Thread(
    target=face_worker,
    daemon=True
).start()


window_name = (
    "WAVELY Vision"
)


cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

if not EMBEDDED_MODE:
    window_settings = {}
    try:
        with open(WINDOW_SETTINGS_FILE, "r", encoding="utf-8") as file:
            window_settings = json.load(file)
    except (OSError, json.JSONDecodeError):
        pass

    window_width = int(window_settings.get("width", 1280))
    window_height = int(window_settings.get("height", 720))
    cv2.resizeWindow(window_name, max(640, window_width), max(480, window_height))

    if "x" in window_settings and "y" in window_settings:
        cv2.moveWindow(window_name, int(window_settings["x"]), int(window_settings["y"]))


fps = 0.0
fps_counter = 0

fps_start = (
    time.monotonic()
)

last_face_publish = (
    0.0
)

last_window_save = 0.0


# ==========================================================
# STARTUP
# ==========================================================

print()

print(
    "WAVELY VISION"
)

print(
    "============="
)

print()

print(
    "Face recognition enabled."
)

print(
    "Fixed Left / Right hand slots enabled."
)

print(
    "Single-hand gestures and swipes enabled."
)

print(
    "STRICT TWO-HAND MODE enabled."
)

print(
    "When both hands are visible, individual actions are suppressed."
)

print(
    "Combo release time: 1.0 second."
)

print()

print(
    "Q or ESC = quit"
)

print()


# ==========================================================
# MAIN LOOP
# ==========================================================

with mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=MAX_HANDS,

    model_complexity=
    HAND_MODEL_COMPLEXITY,

    min_detection_confidence=
    HAND_DETECTION_CONFIDENCE,

    min_tracking_confidence=
    HAND_TRACKING_CONFIDENCE

) as hands:


    while True:

        frame = get_latest_camera_frame()
        if frame is None:
            cv2.waitKey(1)
            continue


        if MIRROR_FRAME:

            frame = cv2.flip(
                frame,
                1
            )


        frame_h, frame_w = (
            frame.shape[:2]
        )


        publish_camera_aspect(frame_w, frame_h)


        now = (
            time.monotonic()
        )


        # ==================================================
        # FACE THREAD
        # ==================================================

        if (
            now
            -
            last_face_publish
            >=
            FACE_FRAME_PUBLISH_INTERVAL
        ):

            with latest_face_frame_lock:

                latest_face_frame = (
                    frame.copy()
                )


            last_face_publish = (
                now
            )


        # ==================================================
        # HAND INFERENCE
        # ==================================================

        small = cv2.resize(
            frame,
            (
                HAND_PROCESS_WIDTH,
                HAND_PROCESS_HEIGHT
            ),
            interpolation=
            cv2.INTER_AREA
        )


        rgb = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2RGB
        )


        rgb.flags.writeable = (
            False
        )


        result = (
            hands.process(
                rgb
            )
        )

        pose_result = pose_tracker.process(
            rgb
        )

        waist_y = 1.0
        waist_tracking = False

        if pose_result.pose_landmarks:
            pose_points = pose_result.pose_landmarks.landmark
            left_shoulder = pose_points[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_points[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = pose_points[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_points[mp_pose.PoseLandmark.RIGHT_HIP]

            # A single reliable hip is enough.  Requiring both hips at 55%
            # was too strict at room distance, which silently made every
            # hand active whenever one hip was partially obscured.
            visible_hips = [
                hip.y
                for hip in (left_hip, right_hip)
                if hip.visibility >= 0.25
            ]

            if visible_hips:
                hip_y = sum(visible_hips) / len(visible_hips)
                visible_shoulders = [
                    shoulder.y
                    for shoulder in (left_shoulder, right_shoulder)
                    if shoulder.visibility >= 0.25
                ]

                if visible_shoulders:
                    shoulder_y = sum(visible_shoulders) / len(visible_shoulders)
                    # The true waist is part-way down the torso, rather than
                    # at hip level.  This makes a naturally resting hand fall
                    # below the active gesture zone.
                    waist_y = shoulder_y + ((hip_y - shoulder_y) * 0.48)
                else:
                    waist_y = hip_y
                waist_tracking = True


        rgb.flags.writeable = (
            True
        )

        # Draw the tracked body structure before the hand labels.  This shows
        # the shoulders, elbows, wrists, hips, knees and ankles used by the
        # pose tracker without covering the gesture information.
        if SHOW_BODY_SKELETON and pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 175, 40), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(80, 220, 255), thickness=2, circle_radius=2),
            )

        # This makes the hip zone test visible during a live test.  Green
        # means the detected line is being used; amber means pose tracking is
        # unavailable and no hand is excluded for that frame.
        if waist_tracking:
            waist_line_y = int(max(0.0, min(1.0, waist_y)) * frame_h)
            cv2.line(frame, (0, waist_line_y), (frame_w, waist_line_y), (0, 220, 90), 2)
            cv2.putText(frame, "GESTURE ZONE: ABOVE GREEN LINE", (18, max(28, waist_line_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 90), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "GESTURE ZONE: HIP TRACKING LOST", (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 180, 255), 2, cv2.LINE_AA)


        seen_keys = set()
        observed = set()

        hand_items = []


        # ==================================================
        # RAW HANDS
        # ==================================================

        if (
            result.multi_hand_landmarks

            and

            result.multi_handedness
        ):

            for (
                landmarks,
                handedness
            ) in zip(

                result.multi_hand_landmarks,
                result.multi_handedness

            ):

                classification = (
                    handedness
                    .classification[0]
                )


                wrist = (
                    landmarks
                    .landmark[0]
                )


                person = (
                    assign_person_to_hand(
                        wrist.x,
                        wrist.y,
                        frame_w,
                        frame_h
                    )
                )


                hand_items.append(
                    {

                        "landmarks":
                        landmarks,

                        "raw_hand":
                        classification.label,

                        "raw_hand_confidence":
                        classification.score,

                        "wrist_x":
                        wrist.x,

                        "wrist_y":
                        wrist.y,

                        "gesture_active":
                        wrist.y
                        <
                        waist_y
                        +
                        WAIST_GESTURE_ZONE_MARGIN,

                        "person":
                        person,

                        "stable_hand":
                        None,

                        "ignore":
                        False

                    }
                )


        # ==================================================
        # PHYSICAL LEFT / RIGHT SLOTS
        # ==================================================

        assign_hand_slots(
            hand_items,
            frame_w
        )


        hand_items = [

            item

            for item
            in hand_items

            if not item.get(
                "ignore",
                False
            )

        ]


        # ==================================================
        # CLASSIFY GESTURES
        # ==================================================

        for item in (
            hand_items
        ):

            item[
                "raw_gesture"
            ] = (
                classify_gesture(
                    item[
                        "landmarks"
                    ]
                )
            )


            item[
                "gesture"
            ] = (
                stabilise_thumb_gesture(
                    item[
                        "person"
                    ],
                    item[
                        "stable_hand"
                    ],
                    item[
                        "raw_gesture"
                    ]
                )
            )


        # ==================================================
        # TWO-HAND MODE
        #
        # If both physical hands are visible for a person,
        # ALL individual gestures/swipes are immediately
        # disabled for that person.
        # ==================================================

        combo_frame = {}


        people_in_frame = {

            item[
                "person"
            ]

            for item
            in hand_items

            if (
                item[
                    "person"
                ]
                !=
                "Unknown"
            )

        }


        people_to_check = (

            set(
                combo_states.keys()
            )

            |

            set(
                two_hand_motion_states.keys()
            )

            |

            people_in_frame

        )


        for person in (
            people_to_check
        ):

            left_item = next(

                (

                    item

                    for item
                    in hand_items

                    if (
                        item[
                            "person"
                        ]
                        ==
                        person

                        and

                        item[
                            "stable_hand"
                        ]
                        ==
                        "Left"

                        and

                        item[
                            "gesture_active"
                        ]
                    )

                ),

                None

            )


            right_item = next(

                (

                    item

                    for item
                    in hand_items

                    if (
                        item[
                            "person"
                        ]
                        ==
                        person

                        and

                        item[
                            "stable_hand"
                        ]
                        ==
                        "Right"

                        and

                        item[
                            "gesture_active"
                        ]
                    )

                ),

                None

            )


            both_visible = (

                left_item
                is not None

                and

                right_item
                is not None

            )


            if (
                both_visible
            ):

                left_gesture = (
                    left_item[
                        "gesture"
                    ]
                )

                right_gesture = (
                    right_item[
                        "gesture"
                    ]
                )


                motion_candidate, _ = (
                    update_two_hand_motion_state(
                        person,
                        left_gesture,
                        right_gesture,
                        left_item[
                            "wrist_x"
                        ],
                        right_item[
                            "wrist_x"
                        ],
                        left_item[
                            "wrist_y"
                        ],
                        right_item[
                            "wrist_y"
                        ],
                        True
                    )
                )


            else:

                left_gesture = (
                    None
                )

                right_gesture = (
                    None
                )


                motion_candidate, _ = (
                    update_two_hand_motion_state(
                        person,
                        left_gesture,
                        right_gesture,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        False
                    )
                )


            (
                combo_name,
                combo_progress,
                two_hand_blocking
            ) = (
                update_combo_state(
                    person,
                    left_gesture,
                    right_gesture,
                    both_visible,
                    motion_candidate
                )
            )


            if (
                two_hand_blocking
            ):

                combo_frame[
                    person
                ] = {

                    "name":
                    combo_name,

                    "progress":
                    combo_progress,

                    "left":
                    left_gesture,

                    "right":
                    right_gesture

                }


                # Kill any old swipe anchors immediately.
                clear_swipe_state(
                    person,
                    "Left"
                )

                clear_swipe_state(
                    person,
                    "Right"
                )


        # ==================================================
        # DRAW TWO-HAND MODE STATUS
        # ==================================================

        combo_y = (
            105
        )


        for person, combo_data in (
            combo_frame.items()
        ):

            combo_name = (
                combo_data[
                    "name"
                ]
            )


            combo_progress = (
                combo_data[
                    "progress"
                ]
            )


            if (
                combo_name
                is not None
            ):

                combo_text = (
                    combo_name.replace(
                        "_",
                        " "
                    )
                )


                display_line = (
                    f"{person} TWO-HAND: "
                    f"{combo_text} "
                    f"{int(combo_progress * 100)}%"
                )


            else:

                left_text = (
                    combo_data[
                        "left"
                    ]
                    or
                    "-"
                )

                right_text = (
                    combo_data[
                        "right"
                    ]
                    or
                    "-"
                )


                display_line = (
                    f"{person} TWO-HAND MODE: "
                    f"L={left_text} "
                    f"R={right_text}"
                )


            cv2.putText(
                frame,
                display_line,
                (
                    25,
                    combo_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.67,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )


            combo_y += (
                30
            )


        # ==================================================
        # PROCESS EACH HAND
        # ==================================================

        for item in (
            hand_items
        ):

            if (
                not item[
                    "gesture_active"
                ]
            ):
                wrist_x = int(item["wrist_x"] * frame_w)
                wrist_y = int(item["wrist_y"] * frame_h)
                cv2.putText(frame, "BELOW WAIST - IGNORED", (wrist_x - 95, wrist_y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 80, 255), 2, cv2.LINE_AA)
                continue

            landmarks = (
                item[
                    "landmarks"
                ]
            )


            raw_hand = (
                item[
                    "raw_hand"
                ]
            )


            person = (
                item[
                    "person"
                ]
            )


            hand_label = (
                item[
                    "stable_hand"
                ]
            )


            wrist_x = (
                item[
                    "wrist_x"
                ]
            )


            wrist_y = (
                item[
                    "wrist_y"
                ]
            )


            gesture = (
                item[
                    "gesture"
                ]
            )


            two_hand_mode = (
                person
                in
                combo_frame
            )


            # ==================================================
            # STRICT TWO-HAND MODE
            # ==================================================

            if (
                two_hand_mode
            ):

                display_gesture = (
                    gesture
                )


                # Critical change:
                # individual gesture cannot trigger.
                action_gesture = (
                    "OTHER"
                )


                # No swipe processing whatsoever.
                clear_swipe_state(
                    person,
                    hand_label
                )


            # ==================================================
            # NORMAL SINGLE-HAND MODE
            # ==================================================

            else:

                raw_open = (
                    gesture
                    ==
                    "OPEN_PALM"
                )


                swipe_gesture = (
                    None
                )

                swipe_mode = (
                    "WAITING_STILL"
                )

                palm_recent = (
                    False
                )


                if (
                    person
                    !=
                    "Unknown"
                ):

                    (
                        swipe_gesture,
                        swipe_mode,
                        palm_recent
                    ) = (
                        process_swipe(
                            person,
                            hand_label,
                            wrist_x,
                            wrist_y,
                            raw_open
                        )
                    )


                if (
                    swipe_gesture
                    is not None
                ):

                    try_trigger_gesture(
                        person,
                        hand_label,
                        swipe_gesture,
                        use_latch=False
                    )


                if (
                    swipe_gesture
                    is not None
                ):

                    display_gesture = (
                        swipe_gesture
                    )

                    action_gesture = (
                        "OTHER"
                    )


                elif (
                    palm_recent

                    and

                    swipe_mode
                    ==
                    "ARMED"
                ):

                    display_gesture = (
                        "SWIPE_READY"
                    )


                    if (
                        gesture
                        ==
                        "OPEN_PALM"
                    ):

                        action_gesture = (
                            "OPEN_PALM"
                        )


                    else:

                        action_gesture = (
                            "OTHER"
                        )


                elif (
                    palm_recent

                    and

                    swipe_mode
                    ==
                    "LOCKED"
                ):

                    display_gesture = (
                        "SWIPE_LOCKED"
                    )

                    action_gesture = (
                        "OTHER"
                    )


                else:

                    display_gesture = (
                        gesture
                    )

                    action_gesture = (
                        gesture
                    )


            # ==================================================
            # STATIC HOLD
            # ==================================================

            (
                state_key,
                progress,
                stable
            ) = (
                update_gesture_state(
                    person,
                    hand_label,
                    action_gesture
                )
            )


            seen_keys.add(
                state_key
            )


            if (
                person
                !=
                "Unknown"

                and

                action_gesture
                !=
                "OTHER"
            ):

                observed.add(
                    (
                        person,
                        hand_label,
                        action_gesture
                    )
                )


            if (
                stable
            ):

                try_trigger_gesture(
                    person,
                    hand_label,
                    action_gesture,
                    use_latch=True
                )


            # ==================================================
            # DRAW HAND
            # ==================================================

            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            hand_x = int(
                wrist_x
                *
                frame_w
            )


            hand_y = int(
                wrist_y
                *
                frame_h
            )


            if (
                two_hand_mode
            ):

                display_prefix = (
                    "2H | "
                )

            else:

                display_prefix = (
                    ""
                )


            cv2.putText(
                frame,
                (
                    f"{display_prefix}"
                    f"{person} "
                    f"{hand_label} Hand: "
                    f"{display_gesture.replace('_', ' ')}"
                ),
                (
                    hand_x - 90,
                    hand_y - 48
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )


            slot = (
                get_hand_slot(
                    person,
                    hand_label
                )
            )


            if (
                slot
                is not None
            ):

                cv2.putText(
                    frame,
                    (
                        f"Slot "
                        f"{slot['slot_id']} "
                        f"{hand_label} "
                        f"(MediaPipe: {raw_hand})"
                    ),
                    (
                        hand_x - 90,
                        hand_y - 70
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (
                        255,
                        255,
                        255
                    ),
                    1,
                    cv2.LINE_AA
                )


            # ==================================================
            # HOLD BAR
            # ==================================================

            bar_x = (
                hand_x - 90
            )

            bar_y = (
                hand_y - 25
            )

            bar_width = (
                170
            )

            bar_height = (
                10
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
                    bar_height
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
                    bar_height
                ),
                (
                    255,
                    255,
                    255
                ),
                -1
            )


        # ==================================================
        # CLEANUP
        # ==================================================

        cleanup_hand_states(
            seen_keys
        )


        update_gesture_latches(
            observed
        )


        # ==================================================
        # DRAW FACES
        # ==================================================

        for face in (
            get_current_faces()
        ):

            x = (
                face[
                    "x"
                ]
            )

            y = (
                face[
                    "y"
                ]
            )

            w = (
                face[
                    "w"
                ]
            )

            h = (
                face[
                    "h"
                ]
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


            if (
                face[
                    "held"
                ]
            ):

                face_text = (
                    f"{face['name']} "
                    f"(HELD)"
                )


            else:

                face_text = (
                    f"{face['name']} "
                    f"{face['confidence']:.1f}"
                )


            cv2.putText(
                frame,
                face_text,
                (
                    x,
                    max(
                        y - 30,
                        30
                    )
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
                frame,
                face[
                    "detector"
                ],
                (
                    x,
                    max(
                        y - 7,
                        52
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (
                    255,
                    255,
                    255
                ),
                1,
                cv2.LINE_AA
            )


        # ==================================================
        # IDENTITY STATUS
        # ==========================================================

        y_line = (
            70
        )


        for name, data in (
            get_recent_identities().items()
        ):

            remaining = max(

                0.0,

                IDENTITY_HOLD_TIME
                -
                (
                    time.monotonic()
                    -
                    data[
                        "confirmed_time"
                    ]
                )

            )


            cv2.putText(
                frame,
                (
                    f"Identity: "
                    f"{name} "
                    f"({remaining:.1f}s hold)"
                ),
                (
                    25,
                    y_line
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


            y_line += (
                26
            )


        # ==================================================
        # FPS
        # ==========================================================

        fps_counter += 1


        elapsed = (
            time.monotonic()
            -
            fps_start
        )


        if (
            elapsed
            >=
            1.0
        ):

            fps = (
                fps_counter
                /
                elapsed
            )


            fps_counter = (
                0
            )


            fps_start = (
                time.monotonic()
            )


        with face_worker_stats_lock:

            face_ms = (
                face_worker_last_ms
            )


        cv2.putText(
            frame,
            (
                f"WAVELY Vision   "
                f"FPS: {fps:.1f}   "
                f"Face: {face_ms:.0f}ms   "
                f"Hand slots: {len(hand_slots)}"
            ),
            (
                25,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )


        if TEST_MODE:
            cv2.putText(frame, "GESTURE TEST MODE - HOME ASSISTANT OFF", (18, frame_h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 255), 2, cv2.LINE_AA)

        cv2.imshow(
            window_name,
            frame
        )

        # OpenCV exposes the current image-window rectangle, including manual
        # resize and move actions.  Save it periodically so an unexpected
        # camera stop does not lose the user's preferred layout.
        now = time.monotonic()
        if not EMBEDDED_MODE and now - last_window_save >= 1.0:
            try:
                window_x, window_y, saved_width, saved_height = cv2.getWindowImageRect(window_name)
                with open(WINDOW_SETTINGS_FILE, "w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "x": window_x,
                            "y": window_y,
                            "width": saved_width,
                            "height": saved_height,
                        },
                        file,
                        indent=2,
                    )
                last_window_save = now
            except cv2.error:
                pass


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if (
            key == ord("q")

            or

            key == 27
        ):

            break


# ==========================================================
# SHUTDOWN
# ==========================================================

# Save one final camera-window rectangle on exit.  The periodic save above
# covers normal use; this also remembers a layout from a short test run.
if not EMBEDDED_MODE:
    try:
        window_x, window_y, saved_width, saved_height = cv2.getWindowImageRect(window_name)
        if saved_width > 100 and saved_height > 100:
            with open(WINDOW_SETTINGS_FILE, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "x": window_x,
                        "y": window_y,
                        "width": saved_width,
                        "height": saved_height,
                    },
                    file,
                    indent=2,
                )
    except cv2.error:
        pass

stop_event.set()

if camera_thread is not None:
    camera_thread.join(timeout=1.0)

cap.release()

cv2.destroyAllWindows()

pose_tracker.close()

print()

print(
    "WAVELY Vision stopped."
)
