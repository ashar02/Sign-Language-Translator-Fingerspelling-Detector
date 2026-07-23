"""
Sign Language Translator - Flask Application

This module provides a Flask-based web application for translating
American Sign Language (ASL) fingerspelling to text and vice versa.
"""

from flask import Flask, render_template, jsonify, request, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
import sys
import signal
import atexit
import logging
import base64
import threading
from typing import Optional, Generator, Any
from dotenv import load_dotenv

# Load environment variables (project root .env, then optional local override)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
load_dotenv()

# Import custom modules
from functions.text_fix import generate_sentences
from functions.voice import text_to_speech
from functions.text_to_sign import text_to_sign_language
from functions.speech_to_text import speech_to_text
# base64 already imported above

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Model and detection settings
MIN_DETECTION_CONFIDENCE: float = float(os.getenv('MIN_DETECTION_CONFIDENCE', '0.3'))
NUM_HAND_LANDMARKS: int = 21
FEATURE_VECTOR_SIZE: int = 42  # 21 landmarks * 2 coordinates (x, y)

# Stabilization settings (can be configured via environment variables)
STABILITY_THRESHOLD: int = int(os.getenv('STABILITY_THRESHOLD', '5'))
STABILITY_TIME_WINDOW: float = float(os.getenv('STABILITY_TIME_WINDOW', '1.0'))
STABILIZATION_DELAY: float = float(os.getenv('STABILIZATION_DELAY', '2.0'))

# Input validation
MAX_TEXT_LENGTH: int = 500

# Server / traffic (python app.py and start.sh / gunicorn)
HOST: str = os.getenv('HOST', '0.0.0.0')
PORT: int = int(os.getenv('PORT', '5000'))
# http (default) or https — https requires SSL_CERT_FILE and SSL_KEY_FILE
TRAFFIC: str = os.getenv('TRAFFIC', 'http').strip().lower()
SSL_CERT_FILE: str = os.getenv('SSL_CERT_FILE', '').strip()
SSL_KEY_FILE: str = os.getenv('SSL_KEY_FILE', '').strip()

# UI visibility (default: show)
SHOW_HEADER: bool = os.getenv('SHOW_HEADER', 'true').lower() in ('1', 'true', 'yes', 'on', 'show')
SHOW_FOOTER: bool = os.getenv('SHOW_FOOTER', 'true').lower() in ('1', 'true', 'yes', 'on', 'show')
SHOW_SIGN_TO_TEXT: bool = os.getenv('SHOW_SIGN_TO_TEXT', 'true').lower() in ('1', 'true', 'yes', 'on', 'show')
SHOW_TEXT_TO_SIGN: bool = os.getenv('SHOW_TEXT_TO_SIGN', 'true').lower() in ('1', 'true', 'yes', 'on', 'show')

# Speak provider: browser (free Web Speech / Google voices) or elevenlabs
SPEAK_PROVIDER: str = os.getenv('SPEAK_PROVIDER', 'browser').strip().lower()
if SPEAK_PROVIDER not in ('browser', 'elevenlabs'):
    SPEAK_PROVIDER = 'browser'

# Camera source: browser (client webcam) or server (OpenCV /dev/video0)
CAMERA_SOURCE: str = os.getenv('CAMERA_SOURCE', 'browser').strip().lower()
if CAMERA_SOURCE not in ('browser', 'server'):
    CAMERA_SOURCE = 'browser'

# Inference location for browser camera: client (fast) or server (frame upload)
INFERENCE_MODE: str = os.getenv('INFERENCE_MODE', 'client').strip().lower()
if INFERENCE_MODE not in ('client', 'server'):
    INFERENCE_MODE = 'client'

# Paths - resolved relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'model.p')
LOG_FILE = os.path.join(BASE_DIR, 'app.log')


def _resolve_path(path: str) -> str:
    """Resolve a path; relative paths are tried from project root, then CWD."""
    if not path:
        return ''
    if os.path.isabs(path):
        return path
    from_root = os.path.join(PROJECT_ROOT, path)
    if os.path.isfile(from_root):
        return from_root
    return os.path.abspath(path)


def get_ssl_context() -> Optional[tuple[str, str]]:
    """Return (cert, key) when HTTPS is enabled; otherwise None.

    TRAFFIC=https requires both SSL_CERT_FILE and SSL_KEY_FILE.
    If TRAFFIC=http but both cert paths are set and exist, HTTPS is still used.
    """
    cert = _resolve_path(SSL_CERT_FILE)
    key = _resolve_path(SSL_KEY_FILE)
    want_https = TRAFFIC in ('https', 'ssl', 'tls')

    if want_https:
        if not SSL_CERT_FILE or not SSL_KEY_FILE:
            logger.error("TRAFFIC=https requires SSL_CERT_FILE and SSL_KEY_FILE in .env")
            sys.exit(1)
        if not os.path.isfile(cert):
            logger.error(f"SSL certificate not found: {cert}")
            sys.exit(1)
        if not os.path.isfile(key):
            logger.error(f"SSL private key not found: {key}")
            sys.exit(1)
        return (cert, key)

    if cert and key and os.path.isfile(cert) and os.path.isfile(key):
        logger.info("SSL cert/key found; enabling HTTPS even though TRAFFIC is not https")
        return (cert, key)

    return None


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================

app = Flask(__name__)

# Try to import and configure rate limiting
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )
    RATE_LIMITING_ENABLED = True
    logger.info("Rate limiting enabled")
except ImportError:
    logger.warning("flask-limiter not installed. Rate limiting disabled. Install with: pip install flask-limiter")
    RATE_LIMITING_ENABLED = False
    limiter = None


# =============================================================================
# MODEL AND MEDIAPIPE INITIALIZATION
# =============================================================================

def load_model() -> Any:
    """Load the trained ML model from pickle file.

    Returns:
        The loaded model object.

    Raises:
        SystemExit: If model cannot be loaded.
    """
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model_dict = pickle.load(f)
        logger.info("Model loaded successfully")
        return model_dict['model']
    except FileNotFoundError:
        logger.error(f"Model file not found at: {MODEL_PATH}")
        logger.error("Please ensure the model file exists. Run train_classifier.py to generate it.")
        sys.exit(1)
    except (pickle.UnpicklingError, KeyError) as e:
        logger.error(f"Error loading model - file may be corrupted: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        sys.exit(1)


def initialize_mediapipe() -> mp.solutions.hands.Hands:
    """Initialize MediaPipe hands detection.

    Returns:
        Configured MediaPipe Hands object.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
    )
    logger.info(f"MediaPipe initialized with confidence threshold: {MIN_DETECTION_CONFIDENCE}")
    return hands


# Initialize model and MediaPipe
model = load_model()
mp_hands = mp.solutions.hands
hands = initialize_mediapipe()
hands_lock = threading.Lock()


def process_detection_frame(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run hand detection + prediction on a BGR frame.

    Args:
        frame: OpenCV BGR image.

    Returns:
        Tuple of (annotated_frame, result_dict).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with hands_lock:
        results = hands.process(frame_rgb)

    predicted_char = ''
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            features = process_hand_landmarks(hand_landmarks)
            if len(features) == FEATURE_VECTOR_SIZE:
                predicted_char, confidence = predict_character(features)
                is_stable, stable_pred = detector.check_sign_stability(predicted_char)
                if is_stable and stable_pred:
                    detector.process_stable_prediction(stable_pred)
            break

    is_buffer_stable = len(detector.stability_buffer) >= STABILITY_THRESHOLD
    if predicted_char or detector.stable_char or detector.detected_sentence:
        frame = draw_overlays(
            frame,
            detector.stable_char or predicted_char,
            detector.detected_sentence,
            is_buffer_stable,
        )

    return frame, {
        'prediction': detector.stable_char or predicted_char,
        'current_prediction': predicted_char,
        'confidence': round(float(confidence), 1),
        'raw_text': ' '.join(detector.detected_sentence) if detector.detected_sentence else '',
        'is_stable': is_buffer_stable,
        'is_recording': detector.is_recording,
        'hand_detected': bool(results.multi_hand_landmarks),
    }

# Label mapping: 0-25 -> a-z
labels_dict = {i: chr(97 + i) for i in range(26)}


# =============================================================================
# SIGN LANGUAGE DETECTOR CLASS
# =============================================================================

class SignLanguageDetector:
    """Encapsulates the state and logic for sign language detection.

    This class manages the detection state, stability checking, and
    sentence building for the sign-to-text conversion process.
    """

    def __init__(self) -> None:
        """Initialize the detector with default state."""
        self.reset()

    def reset(self) -> None:
        """Reset the detector state to initial values."""
        self.detected_sentence: list[str] = []
        self.is_recording: bool = False
        self.last_confirmed_char: str = ""
        self.last_detection_time: float = time.time()
        self.stable_char: str = ""
        self.current_meaningful_sentence: str = ""
        self.stability_buffer: list[tuple[str, float]] = []

    def start_recording(self) -> None:
        """Start recording mode and reset sentence."""
        self.is_recording = True
        self.detected_sentence = []
        self.stability_buffer = []
        self.last_confirmed_char = ""
        self.stable_char = ""
        logger.info("Recording started")

    def stop_recording(self) -> tuple[str, str]:
        """Stop recording and generate meaningful sentence.

        Returns:
            Tuple of (raw_text, meaningful_sentence)
        """
        self.is_recording = False
        raw_text = ' '.join(self.detected_sentence)

        if raw_text:
            try:
                self.current_meaningful_sentence = generate_sentences(raw_text)
                logger.info(f"Generated sentence: {self.current_meaningful_sentence}")
            except Exception as e:
                logger.error(f"Error generating sentence: {e}")
                self.current_meaningful_sentence = raw_text
        else:
            self.current_meaningful_sentence = ""

        logger.info(f"Recording stopped. Raw: '{raw_text}', Processed: '{self.current_meaningful_sentence}'")
        return raw_text, self.current_meaningful_sentence

    def check_sign_stability(self, prediction: str) -> tuple[bool, Optional[str]]:
        """Check if a sign prediction is stable over time.

        Args:
            prediction: The predicted character.

        Returns:
            Tuple of (is_stable, stable_prediction or None)
        """
        current_time = time.time()

        # Remove old predictions outside the time window
        self.stability_buffer = [
            (pred, t) for pred, t in self.stability_buffer
            if current_time - t < STABILITY_TIME_WINDOW
        ]

        # Add new prediction
        self.stability_buffer.append((prediction, current_time))

        # Check if we have enough predictions and they're all the same
        if len(self.stability_buffer) >= STABILITY_THRESHOLD:
            recent_predictions = [pred for pred, _ in self.stability_buffer[-STABILITY_THRESHOLD:]]
            if all(pred == recent_predictions[0] for pred in recent_predictions):
                return True, recent_predictions[0]

        return False, None

    def process_stable_prediction(self, prediction: str) -> None:
        """Process a stable prediction and add to sentence if appropriate.

        Args:
            prediction: The stable predicted character.
        """
        current_time = time.time()
        self.stable_char = prediction

        # Add to sentence if recording and enough time has passed
        if (self.is_recording and
            prediction != self.last_confirmed_char and
            current_time - self.last_detection_time >= STABILIZATION_DELAY):

            self.detected_sentence.append(prediction)
            self.last_confirmed_char = prediction
            self.last_detection_time = current_time
            logger.debug(f"Added character: {prediction}, Sentence: {self.detected_sentence}")


    def append_confirmed_character(self, prediction: str) -> None:
        """Append a client-confirmed character while recording."""
        if not self.is_recording or not prediction:
            return
        char = prediction.strip().upper()
        if len(char) != 1 or not char.isalpha():
            return
        self.stable_char = char
        if char != self.last_confirmed_char:
            self.detected_sentence.append(char)
            self.last_confirmed_char = char
            self.last_detection_time = time.time()
            logger.debug(f"Client appended character: {char}, Sentence: {self.detected_sentence}")

    def replace_detected_sentence(self, letters: list[str]) -> None:
        """Replace detected sentence from client-side inference letters."""
        cleaned = []
        for item in letters or []:
            ch = str(item).strip().upper()
            if len(ch) == 1 and ch.isalpha():
                cleaned.append(ch)
        self.detected_sentence = cleaned
        if cleaned:
            self.stable_char = cleaned[-1]
            self.last_confirmed_char = cleaned[-1]


# Create global detector instance
detector = SignLanguageDetector()


# =============================================================================
# VIDEO PROCESSING FUNCTIONS
# =============================================================================

def process_hand_landmarks(hand_landmarks) -> list[float]:
    """Extract normalized feature vector from hand landmarks.

    Args:
        hand_landmarks: MediaPipe hand landmarks object.

    Returns:
        List of normalized x, y coordinates (42 values).
    """
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

    min_x, min_y = min(x_coords), min(y_coords)

    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x - min_x)
        features.append(lm.y - min_y)

    return features


def predict_character(features: list[float]) -> tuple[str, float]:
    """Predict character from feature vector.

    Args:
        features: Normalized feature vector (42 values).

    Returns:
        Tuple of (predicted_character, confidence)
    """
    if len(features) != FEATURE_VECTOR_SIZE:
        return "", 0.0

    prediction = model.predict([np.asarray(features)])
    predicted_char = labels_dict[int(prediction[0])]

    # Get confidence if available
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba([np.asarray(features)])[0][int(prediction[0])] * 100
    else:
        confidence = 100.0

    return predicted_char.upper(), confidence


def draw_overlays(frame: np.ndarray, stable_char: str,
                  sentence: list[str], is_stable: bool) -> np.ndarray:
    """Draw text overlays on the video frame.

    Args:
        frame: OpenCV image frame.
        stable_char: Currently stable character.
        sentence: List of detected characters.
        is_stable: Whether current detection is stable.

    Returns:
        Frame with overlays drawn.
    """
    # Magenta (BGR) for stronger contrast on the camera feed
    overlay_color = (255, 0, 255)
    left = 180

    # Stable character
    cv2.putText(frame, f"Stable Character: {stable_char}", (left, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2)

    # Current sentence
    cv2.putText(frame, f"Sentence: {' '.join(sentence)}", (left, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2)

    # Stability status
    status_text = "Stable" if is_stable else "Unstable"
    status_color = overlay_color if is_stable else (0, 0, 255)
    cv2.putText(frame, f"Sign Status: {status_text}", (left, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    return frame


def create_error_frame(message: str) -> bytes:
    """Create a frame with an error message.

    Args:
        message: Error message to display.

    Returns:
        JPEG encoded error frame.
    """
    # Create a dark frame with error message
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)  # Dark gray background

    # Add error message
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 0.7, 2)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (480 + text_size[1]) // 2
    cv2.putText(frame, message, (text_x, text_y), font, 0.7, (200, 200, 200), 2)

    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes() if ret else b''


def generate_frames() -> Generator[bytes, None, None]:
    """Generate video frames with sign language detection.

    Yields:
        JPEG encoded frames for streaming.
    """
    cap = None
    consecutive_failures = 0
    max_failures = 30  # Allow up to 30 consecutive failures before reconnecting
    reconnect_attempts = 0
    max_reconnect_attempts = 5

    while reconnect_attempts < max_reconnect_attempts:
        try:
            # Try to open camera
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.error("Failed to open camera")
                    reconnect_attempts += 1
                    error_frame = create_error_frame(f"Camera not available. Retrying... ({reconnect_attempts}/{max_reconnect_attempts})")
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
                    time.sleep(2)
                    continue

                logger.info("Camera opened successfully")
                consecutive_failures = 0
                reconnect_attempts = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame ({consecutive_failures}/{max_failures})")

                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive frame failures, reconnecting camera...")
                        if cap is not None:
                            cap.release()
                            cap = None
                        reconnect_attempts += 1
                        break  # Break inner loop to reconnect

                    # Yield a placeholder frame while waiting
                    time.sleep(0.1)
                    continue

                # Reset failure counter on successful read
                consecutive_failures = 0

                frame, _ = process_detection_frame(frame)

                # Encode and yield frame
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except GeneratorExit:
            # Client disconnected, clean up
            logger.info("Client disconnected from video feed")
            break
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            reconnect_attempts += 1
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(1)

    # Final cleanup
    if cap is not None:
        cap.release()
        logger.info("Camera released")


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the main page."""
    return render_template(
        'index.html',
        show_header=SHOW_HEADER,
        show_footer=SHOW_FOOTER,
        show_sign_to_text=SHOW_SIGN_TO_TEXT,
        show_text_to_sign=SHOW_TEXT_TO_SIGN,
        single_panel=SHOW_SIGN_TO_TEXT ^ SHOW_TEXT_TO_SIGN,
        speak_provider=SPEAK_PROVIDER,
        camera_source=CAMERA_SOURCE,
        inference_mode=INFERENCE_MODE,
        stability_threshold=STABILITY_THRESHOLD,
        stability_time_window=STABILITY_TIME_WINDOW,
        stabilization_delay=STABILIZATION_DELAY,
    )


@app.route('/health')
def health():
    """Health check endpoint for Docker/Kubernetes."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'rate_limiting': RATE_LIMITING_ENABLED
    })


@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Accept a browser webcam frame and return detection results + annotated image."""
    try:
        image_bytes = None

        if request.files.get('frame'):
            image_bytes = request.files['frame'].read()
        elif request.is_json and request.json:
            data_url = request.json.get('image', '')
            if ',' in data_url:
                data_url = data_url.split(',', 1)[1]
            if data_url:
                image_bytes = base64.b64decode(data_url)

        if not image_bytes:
            return jsonify({'status': 'error', 'message': 'No frame provided'}), 400

        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

        annotated, result = process_detection_frame(frame)

        ok, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return jsonify({'status': 'error', 'message': 'Failed to encode frame'}), 500

        result.update({
            'status': 'success',
            'annotated_image': 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('ascii'),
        })
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in predict_frame: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start recording sign language gestures."""
    detector.start_recording()
    return jsonify({'status': 'success', 'message': 'Recording started'})


@app.route('/stop_recording', methods=['POST'])
def stop_recording_route():
    """Stop recording and process the detected sentence."""
    # Client inference may send the final letter list explicitly
    if request.is_json and request.json:
        letters = request.json.get('letters')
        if isinstance(letters, list):
            detector.replace_detected_sentence(letters)

    raw_text, meaningful_sentence = detector.stop_recording()
    return jsonify({
        'status': 'success',
        'raw_text': raw_text,
        'meaningful_sentence': meaningful_sentence
    })


@app.route('/append_character', methods=['POST'])
def append_character():
    """Append a character confirmed by client-side inference."""
    if not request.is_json or not request.json:
        return jsonify({'status': 'error', 'message': 'JSON required'}), 400
    char = (request.json.get('character') or '').strip()
    detector.append_confirmed_character(char)
    return jsonify({
        'status': 'success',
        'raw_text': ' '.join(detector.detected_sentence),
        'prediction': detector.stable_char,
    })


# Apply rate limiting only if available
if RATE_LIMITING_ENABLED and limiter:
    stop_recording_route = limiter.limit("10 per minute")(stop_recording_route)


@app.route('/get_current_prediction')
def get_current_prediction():
    """Get live prediction and raw letter stream while recording."""
    raw_text = ' '.join(detector.detected_sentence) if detector.detected_sentence else ''
    return jsonify({
        'prediction': detector.stable_char,
        'raw_text': raw_text,
        'is_recording': detector.is_recording,
    })


@app.route('/speak_text', methods=['POST'])
def speak_text():
    """Generate TTS audio for browser playback (ElevenLabs provider)."""
    try:
        if SPEAK_PROVIDER != 'elevenlabs':
            return jsonify({
                'status': 'error',
                'message': 'Server TTS disabled. SPEAK_PROVIDER is set to browser.'
            })

        text = ''
        if request.is_json and request.json:
            text = (request.json.get('text') or '').strip()
        if not text:
            text = (detector.current_meaningful_sentence or '').strip()

        # Speak only the guessed phrase from "ORIGINAL / Guess"
        if ' / ' in text:
            text = text.split(' / ', 1)[1].strip()

        if not text:
            return jsonify({
                'status': 'error',
                'message': 'No text available to speak. Please record some signs first.'
            })

        audio_bytes = text_to_speech(text)
        if not audio_bytes:
            return jsonify({
                'status': 'error',
                'message': 'Failed to generate audio. Check ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env.'
            })

        logger.info(f"Generated TTS for browser playback: {text[:80]}")
        return jsonify({
            'status': 'success',
            'message': 'Audio ready',
            'audio': base64.b64encode(audio_bytes).decode('ascii'),
            'mime_type': 'audio/mpeg',
        })
    except Exception as e:
        logger.error(f"Error in speak_text: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to play audio: {str(e)}'
        })


@app.route('/convert_text', methods=['POST'])
def convert_text():
    """Convert input text to sign language images."""
    # Validate request
    if not request.is_json:
        return jsonify({
            'status': 'error',
            'message': 'Request must be JSON'
        }), 400

    text = request.json.get('text', '').strip()

    # Validate input
    if not text:
        return jsonify({
            'status': 'error',
            'message': 'No text provided. Please enter some text to convert.'
        })

    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({
            'status': 'error',
            'message': f'Text too long. Maximum {MAX_TEXT_LENGTH} characters allowed.'
        })

    # Filter to only allow letters and spaces
    filtered_text = ''.join(c for c in text if c.isalpha() or c.isspace())

    if not filtered_text:
        return jsonify({
            'status': 'error',
            'message': 'Please enter English letters only (A-Z).'
        })

    try:
        images_data = text_to_sign_language(filtered_text)
        logger.info(f"Converted text to sign: {filtered_text}")
        return jsonify({
            'status': 'success',
            'message': 'Text converted successfully',
            'images': images_data
        })
    except Exception as e:
        logger.error(f"Error in convert_text: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to convert text: {str(e)}'
        })


# Apply rate limiting only if available
if RATE_LIMITING_ENABLED and limiter:
    convert_text = limiter.limit("10 per minute")(convert_text)


@app.route('/convert_speech_to_sign', methods=['POST'])
def convert_speech_to_sign():
    """Convert speech input to sign language images."""
    try:
        # Convert speech to text
        logger.info("Starting speech recognition...")
        text = speech_to_text()

        if text is None:
            return jsonify({
                'status': 'error',
                'message': 'Could not understand speech. Please try again and speak clearly.'
            })

        logger.info(f"Recognized speech: {text}")

        # Convert text to sign language
        images_data = text_to_sign_language(text)

        return jsonify({
            'status': 'success',
            'text': text,
            'images': images_data
        })

    except Exception as e:
        logger.error(f"Error in convert_speech_to_sign: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Speech conversion failed: {str(e)}'
        })


# =============================================================================
# CLEANUP AND SHUTDOWN HANDLERS
# =============================================================================

def cleanup() -> None:
    """Clean up resources on shutdown."""
    global hands
    try:
        if hands:
            hands.close()
            logger.info("MediaPipe hands closed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def signal_handler(sig, frame) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {sig}, shutting down...")
    cleanup()
    sys.exit(0)


# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    logger.info("Starting Sign Language Translator...")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Host: {HOST}")
    logger.info(f"Port: {PORT}")
    logger.info(f"Traffic: {TRAFFIC}")
    logger.info(f"Camera source: {CAMERA_SOURCE}")
    logger.info(f"Inference mode: {INFERENCE_MODE}")
    logger.info(f"Speak provider: {SPEAK_PROVIDER}")
    logger.info(f"Stability threshold: {STABILITY_THRESHOLD}")
    logger.info(f"Stabilization delay: {STABILIZATION_DELAY}s")

    ssl_context = get_ssl_context()
    if ssl_context:
        logger.info(f"HTTPS enabled (cert={ssl_context[0]})")
    else:
        logger.info("HTTP enabled (no SSL certs)")

    # Run the Flask app
    # Note: debug=True is for development only
    # For production, use a WSGI server like gunicorn via start.sh
    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
        ssl_context=ssl_context,
    )
