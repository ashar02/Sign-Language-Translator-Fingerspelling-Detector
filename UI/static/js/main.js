/**
 * SignBridge — ASL Translator
 * Main JavaScript Controller
 *
 * Handles all client-side interactions including:
 * - Recording control for sign-to-text translation
 * - Text-to-sign conversion with animated display
 * - Speech-to-sign conversion
 * - UI state management, toasts, and loading states
 */

// =============================================================================
// STATE MANAGEMENT
// =============================================================================

let recording = false;
let currentImageIndex = 0;
let imagesData = [];
let imageInterval = null;
let speechRecording = false;

// =============================================================================
// DOM ELEMENTS (cached for performance)
// =============================================================================

const elements = {
    get loadingOverlay() { return document.getElementById('loading-overlay'); },
    get toast() { return document.getElementById('toast'); },
    get toastMessage() { return document.getElementById('toast-message'); },
    get recordBtn() { return document.getElementById('record-btn'); },
    get stopBtn() { return document.getElementById('stop-btn'); },
    get speakBtn() { return document.getElementById('speak-btn'); },
    get convertTextBtn() { return document.getElementById('convert-text-btn'); },
    get convertSpeechBtn() { return document.getElementById('convert-speech-btn'); },
    get predictionBox() { return document.getElementById('prediction-box'); },
    get outputBox() { return document.getElementById('output-box'); },
    get textInput() { return document.getElementById('text-input'); },
    get charCount() { return document.getElementById('char-count'); },
    get signDisplay() { return document.getElementById('sign-display'); },
    get recordingIndicator() { return document.getElementById('recording-indicator'); },
    get progressBar() { return document.getElementById('progress-bar'); },
    get signProgress() { return document.getElementById('sign-progress'); }
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Show the loading overlay with animation
 */
function showLoading() {
    const overlay = elements.loadingOverlay;
    if (overlay) {
        overlay.classList.add('active');
        overlay.setAttribute('aria-hidden', 'false');
    }
}

/**
 * Hide the loading overlay
 */
function hideLoading() {
    const overlay = elements.loadingOverlay;
    if (overlay) {
        overlay.classList.remove('active');
        overlay.setAttribute('aria-hidden', 'true');
    }
}

/**
 * Show a toast notification
 * @param {string} message - The message to display
 * @param {string} type - 'error', 'success', or 'info'
 * @param {number} duration - How long to show the toast (ms)
 */
function showToast(message, type = 'error', duration = 5000) {
    const toast = elements.toast;
    const toastMessage = elements.toastMessage;

    if (toast && toastMessage) {
        toastMessage.textContent = message;
        toast.className = `toast toast-${type} show`;

        // Auto-hide after duration
        setTimeout(() => {
            hideToast();
        }, duration);
    }
}

/**
 * Hide the toast notification
 */
function hideToast() {
    const toast = elements.toast;
    if (toast) {
        toast.classList.remove('show');
    }
}

/**
 * Disable a button and show loading state
 * @param {string} buttonId - The button element ID
 */
function setButtonLoading(buttonId) {
    const button = document.getElementById(buttonId);
    if (button) {
        button.disabled = true;
        button.classList.add('loading');
    }
}

/**
 * Re-enable a button and restore original state
 * @param {string} buttonId - The button element ID
 */
function resetButton(buttonId) {
    const button = document.getElementById(buttonId);
    if (button) {
        button.disabled = false;
        button.classList.remove('loading');
    }
}

/**
 * Make a fetch request with error handling
 * @param {string} url - The URL to fetch
 * @param {object} options - Fetch options
 * @returns {Promise<object>} - The JSON response
 */
async function fetchWithErrorHandling(url, options = {}) {
    try {
        const response = await fetch(url, options);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `Server error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        if (error.name === 'TypeError') {
            throw new Error('Network error. Please check your connection.');
        }
        throw error;
    }
}

/**
 * Update the progress bar
 * @param {number} current - Current position
 * @param {number} total - Total items
 */
function updateProgressBar(current, total) {
    const progressBar = elements.progressBar;
    const signProgress = elements.signProgress;

    if (progressBar && signProgress) {
        const percentage = total > 0 ? (current / total) * 100 : 0;
        progressBar.style.width = `${percentage}%`;

        // Show/hide progress bar container
        if (total > 0) {
            signProgress.classList.add('active');
        } else {
            signProgress.classList.remove('active');
        }
    }
}

// =============================================================================
// SIGN TO TEXT FUNCTIONS
// =============================================================================

/**
 * Show recording indicator
 */
function showRecordingIndicator() {
    const indicator = elements.recordingIndicator;
    if (indicator) {
        indicator.classList.add('active');
    }
}

/**
 * Hide recording indicator
 */
function hideRecordingIndicator() {
    const indicator = elements.recordingIndicator;
    if (indicator) {
        indicator.classList.remove('active');
    }
}

/**
 * Start recording sign language gestures
 */
async function startRecording() {
    if (recording) {
        showToast('Already recording!', 'info');
        return;
    }

    try {
        setButtonLoading('record-btn');

        const data = await fetchWithErrorHandling('/start_recording', {
            method: 'POST'
        });

        if (data.status === 'success') {
            recording = true;
            showToast('Recording started! For double letters (LL), briefly move your hand away, then sign again.', 'success', 4500);

            // Update UI to show recording state
            elements.recordBtn?.classList.add('recording');
            showRecordingIndicator();

            // Inference runs only while recording
            startInferenceWhileRecording();

            if (elements.predictionBox) {
                elements.predictionBox.textContent = '—';
            }
            if (elements.outputBox) {
                elements.outputBox.textContent = 'Listening for signs...';
            }
        } else {
            showToast(data.message || 'Failed to start recording', 'error');
        }
    } catch (error) {
        console.error('Error starting recording:', error);
        showToast(error.message || 'Failed to start recording', 'error');
    } finally {
        resetButton('record-btn');
    }
}

/**
 * Stop recording and process the detected sentence
 */
async function stopRecording() {
    if (!recording) {
        showToast('Not currently recording', 'info');
        return;
    }

    try {
        setButtonLoading('stop-btn');
        showLoading();

        const payload = {};
        if (window.ClientInference && getInferenceMode() === 'client') {
            const state = window.ClientInference.getState();
            payload.letters = state.letters || [];
        }

        // Stop inference as soon as recording ends (camera preview stays on)
        stopInferenceWhileIdle();

        const data = await fetchWithErrorHandling('/stop_recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        recording = false;
        elements.recordBtn?.classList.remove('recording');
        hideRecordingIndicator();

        if (elements.predictionBox) {
            elements.predictionBox.textContent = '—';
        }

        if (data.status === 'success') {
            const outputBox = elements.outputBox;

            if (data.meaningful_sentence) {
                if (outputBox) outputBox.textContent = data.meaningful_sentence;
                showToast('Recording processed successfully!', 'success', 3000);
            } else if (data.raw_text) {
                if (outputBox) outputBox.textContent = data.raw_text;
                showToast('No text correction available', 'info');
            } else {
                if (outputBox) outputBox.textContent = 'No signs detected. Try again.';
                showToast('No signs were detected', 'info');
            }
        } else {
            showToast(data.message || 'Failed to process recording', 'error');
        }
    } catch (error) {
        console.error('Error stopping recording:', error);
        showToast(error.message || 'Failed to stop recording', 'error');
        recording = false;
        hideRecordingIndicator();
        if (elements.predictionBox) {
            elements.predictionBox.textContent = '—';
        }
    } finally {
        resetButton('stop-btn');
        hideLoading();
    }
}

/**
 * Update prediction + live raw text while recording (server-camera fallback poll)
 */
function updatePrediction() {
    if (!recording) return;
    if (getCameraSource() === 'browser') return; // browser mode updates via /predict_frame

    fetch('/get_current_prediction')
        .then(response => response.json())
        .then(data => {
            applyPredictionUi(data);
        })
        .catch(error => {
            console.error('Error fetching prediction:', error);
        });
}

function applyPredictionUi(data) {
    const predictionBox = elements.predictionBox;
    const pred = data.prediction || data.current_prediction;
    if (predictionBox && pred) {
        predictionBox.textContent = pred;
        predictionBox.classList.add('pulse');
        setTimeout(() => predictionBox.classList.remove('pulse'), 300);
    }

    const outputBox = elements.outputBox;
    if (outputBox && recording) {
        if (data.raw_text) {
            outputBox.textContent = data.raw_text;
        } else if (
            outputBox.textContent === 'Your translated text will appear here...' ||
            outputBox.textContent === 'Listening for signs...'
        ) {
            outputBox.textContent = 'Listening for signs...';
        }
    }
}

// Poll for live letters while recording (server camera mode only)
setInterval(updatePrediction, 400);

/**
 * From "HELO / Hello." return only the guessed part for speech.
 * While recording / letter-only text, returns the text as-is.
 */
function getSpeakableText(text) {
    const value = (text || '').trim();
    if (!value) return '';
    if (value.includes(' / ')) {
        return value.split(' / ').slice(1).join(' / ').trim();
    }
    return value;
}

/**
 * Clear translated text (UI + client/server letter buffers).
 */
async function clearTranslatedText() {
    const outputBox = elements.outputBox;
    const placeholder = 'Your translated text will appear here...';
    const current = outputBox?.textContent?.trim() || '';

    if (
        !current ||
        current === placeholder ||
        current === 'Listening for signs...'
    ) {
        showToast('Nothing to clear.', 'info');
        return;
    }

    if (outputBox) {
        outputBox.textContent = recording ? 'Listening for signs...' : placeholder;
    }
    if (elements.predictionBox) {
        elements.predictionBox.textContent = '—';
    }
    if (window.ClientInference && typeof window.ClientInference.clearSentence === 'function') {
        window.ClientInference.clearSentence();
    }

    try {
        await fetch('/clear_translated_text', { method: 'POST' });
        showToast('Translated text cleared', 'success', 2000);
    } catch (error) {
        console.warn('Could not clear server text state:', error);
        showToast('Translated text cleared', 'success', 2000);
    }
}

/**
 * Speak translated text aloud.
 * Default: free browser Web Speech API (Google voices in Chrome).
 * Optional: ElevenLabs when SPEAK_PROVIDER=elevenlabs.
 * Speaks only the guessed phrase after "ORIGINAL / Guess".
 */
async function speakText() {
    const outputBox = elements.outputBox;
    const fullText = outputBox?.textContent?.trim();
    const text = getSpeakableText(fullText);

    if (!fullText || fullText === 'Your translated text will appear here...' || fullText === 'No signs detected. Try again.' || fullText === 'Listening for signs...') {
        showToast('No text to speak. Record some signs first.', 'info');
        return;
    }

    if (!text) {
        showToast('No guessed word to speak yet.', 'info');
        return;
    }

    const provider = (window.APP_CONFIG && window.APP_CONFIG.speakProvider) || 'browser';

    try {
        setButtonLoading('speak-btn');

        if (provider === 'elevenlabs') {
            const data = await fetchWithErrorHandling('/speak_text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (data.status === 'success' && data.audio) {
                const mime = data.mime_type || 'audio/mpeg';
                const audio = new Audio(`data:${mime};base64,${data.audio}`);
                await audio.play();
                showToast('Playing audio', 'success', 2000);
            } else {
                showToast(data.message || 'Failed to play audio', 'error');
            }
            return;
        }

        // Free browser TTS (uses system / Google voices in Chrome)
        if (!window.speechSynthesis) {
            showToast('Speech is not supported in this browser. Try Chrome.', 'error');
            return;
        }

        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 1;
        utterance.pitch = 1;

        const pickVoice = () => {
            const voices = window.speechSynthesis.getVoices();
            const preferred = voices.find(v => /en(-|_)US/i.test(v.lang) && /google/i.test(v.name))
                || voices.find(v => /en(-|_)US/i.test(v.lang))
                || voices.find(v => /^en/i.test(v.lang));
            if (preferred) utterance.voice = preferred;
        };
        pickVoice();
        if (!utterance.voice) {
            window.speechSynthesis.onvoiceschanged = () => {
                pickVoice();
            };
        }

        await new Promise((resolve, reject) => {
            utterance.onend = resolve;
            utterance.onerror = (event) => reject(new Error(event.error || 'Speech failed'));
            window.speechSynthesis.speak(utterance);
        });

        showToast('Playing audio', 'success', 2000);
    } catch (error) {
        console.error('Error speaking text:', error);
        showToast(error.message || 'Failed to play audio', 'error');
    } finally {
        resetButton('speak-btn');
    }
}

// =============================================================================
// TEXT TO SIGN FUNCTIONS
// =============================================================================

/**
 * Reset sign display to placeholder state
 */
function resetSignDisplay() {
    const signDisplay = elements.signDisplay;
    if (signDisplay) {
        signDisplay.innerHTML = `
            <div class="sign-placeholder">
                <svg viewBox="0 0 80 80" fill="none" stroke="currentColor" stroke-width="1">
                    <path d="M40 10C55 15 65 30 60 50C55 70 25 75 15 55C5 35 15 15 35 10C38 9 38 9 40 10Z" opacity="0.3"/>
                    <circle cx="35" cy="40" r="4" fill="currentColor" opacity="0.3"/>
                </svg>
                <p>Sign gestures will appear here</p>
            </div>
        `;
    }
    updateProgressBar(0, 0);
}

/**
 * Convert text input to sign language images
 */
async function convertText() {
    const textInput = elements.textInput;
    const text = textInput?.value?.trim();

    if (!text) {
        showToast('Please enter some text to convert', 'info');
        return;
    }

    if (text.length > 500) {
        showToast('Text is too long. Maximum 500 characters allowed.', 'error');
        return;
    }

    try {
        setButtonLoading('convert-text-btn');
        showLoading();

        const data = await fetchWithErrorHandling('/convert_text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (data.status === 'success') {
            // Stop any existing animation
            if (imageInterval) {
                clearInterval(imageInterval);
            }

            imagesData = data.images;
            currentImageIndex = 0;

            if (imagesData.length > 0) {
                showToast(`Converting ${imagesData.length} characters to sign language`, 'success', 2000);
                updateProgressBar(0, imagesData.length);
                imageInterval = setInterval(displayNextImage, 1000);
            } else {
                showToast('No valid characters to convert', 'info');
                resetSignDisplay();
            }
        } else {
            showToast(data.message || 'Failed to convert text', 'error');
        }
    } catch (error) {
        console.error('Error converting text:', error);
        showToast(error.message || 'Failed to convert text', 'error');
    } finally {
        resetButton('convert-text-btn');
        hideLoading();
    }
}

/**
 * Convert speech to sign language
 */
async function convertSpeech() {
    if (speechRecording) {
        showToast('Already recording speech', 'info');
        return;
    }

    try {
        speechRecording = true;
        setButtonLoading('convert-speech-btn');
        showLoading();

        // Clear the text input to show we're listening
        if (elements.textInput) {
            elements.textInput.value = '';
            updateCharCount();
        }

        const data = await fetchWithErrorHandling('/convert_speech_to_sign', {
            method: 'POST'
        });

        if (data.status === 'success') {
            // Show recognized text in the input
            if (elements.textInput) {
                elements.textInput.value = data.text;
                updateCharCount();
            }

            // Start sign animation
            if (imageInterval) {
                clearInterval(imageInterval);
            }

            imagesData = data.images;
            currentImageIndex = 0;

            if (imagesData.length > 0) {
                showToast(`Recognized: "${data.text}"`, 'success', 3000);
                updateProgressBar(0, imagesData.length);
                imageInterval = setInterval(displayNextImage, 1000);
            }
        } else {
            showToast(data.message || 'Could not understand speech', 'error');
        }
    } catch (error) {
        console.error('Error converting speech:', error);
        showToast(error.message || 'Speech conversion failed', 'error');
    } finally {
        speechRecording = false;
        resetButton('convert-speech-btn');
        hideLoading();
    }
}

/**
 * Display the next sign language image in the sequence
 */
function displayNextImage() {
    const signDisplay = elements.signDisplay;

    if (currentImageIndex < imagesData.length) {
        const imageData = imagesData[currentImageIndex];

        // Update progress bar
        updateProgressBar(currentImageIndex + 1, imagesData.length);

        if (imageData.image) {
            signDisplay.innerHTML = `
                <div class="sign-content">
                    <img src="data:image/png;base64,${imageData.image}"
                         alt="Sign language gesture for letter ${imageData.character}"
                         class="sign-image">
                    <span class="sign-letter">${imageData.character}</span>
                </div>
            `;
        } else {
            // Space character
            signDisplay.innerHTML = `
                <div class="sign-content space-indicator">
                    <div class="space-visual">
                        <svg viewBox="0 0 60 20" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M5 10h50" stroke-dasharray="4 4"/>
                        </svg>
                    </div>
                    <span class="sign-letter">SPACE</span>
                </div>
            `;
        }

        // Add animation class
        signDisplay.classList.add('transitioning');
        setTimeout(() => signDisplay.classList.remove('transitioning'), 300);

        currentImageIndex++;
    } else {
        // Animation complete
        clearInterval(imageInterval);
        imageInterval = null;
        currentImageIndex = 0;

        // Show completion message
        setTimeout(() => {
            signDisplay.innerHTML = `
                <div class="sign-complete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                        <polyline points="22,4 12,14.01 9,11.01"/>
                    </svg>
                    <p>Conversion complete!</p>
                    <span>Enter new text to continue</span>
                </div>
            `;
            updateProgressBar(0, 0);
        }, 500);
    }
}

// =============================================================================
// CHARACTER COUNT
// =============================================================================

/**
 * Update the character count display
 */
function updateCharCount() {
    const textInput = elements.textInput;
    const charCount = elements.charCount;

    if (textInput && charCount) {
        const length = textInput.value.length;
        charCount.textContent = length;

        // Visual feedback when approaching limit
        const counter = charCount.parentElement;
        if (counter) {
            counter.classList.toggle('warning', length > 450);
            counter.classList.toggle('danger', length >= 500);
        }
    }
}

// =============================================================================
// KEYBOARD SHORTCUTS
// =============================================================================

/**
 * Handle keyboard shortcuts
 * @param {KeyboardEvent} event
 */
function handleKeyboardShortcuts(event) {
    // Ctrl/Cmd+Enter to convert text
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const activeElement = document.activeElement;
        if (activeElement && activeElement.id === 'text-input') {
            event.preventDefault();
            convertText();
        }
    }

    // Escape to stop recording
    if (event.key === 'Escape' && recording) {
        stopRecording();
    }

    // Space to start recording when focused on record button
    if (event.key === ' ' && document.activeElement?.id === 'record-btn' && !recording) {
        event.preventDefault();
        startRecording();
    }
}

// =============================================================================
// ACCESSIBILITY
// =============================================================================

/**
 * Announce to screen readers
 * @param {string} message - Message to announce
 */
function announceToScreenReader(message) {
    const announcement = document.createElement('div');
    announcement.setAttribute('role', 'status');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    document.body.appendChild(announcement);

    setTimeout(() => {
        document.body.removeChild(announcement);
    }, 1000);
}

// =============================================================================
// VIDEO / CAMERA MANAGEMENT
// =============================================================================

let videoFeedRetryCount = 0;
const MAX_VIDEO_RETRIES = 10;
const VIDEO_RETRY_DELAY = 3000;
let videoFeedCheckInterval = null;
let browserStream = null;
let browserPredictTimer = null;
let browserPredictInFlight = false;
const BROWSER_PREDICT_INTERVAL_MS = 180;

function getCameraSource() {
    return (window.APP_CONFIG && window.APP_CONFIG.cameraSource) || 'browser';
}

function getInferenceMode() {
    return (window.APP_CONFIG && window.APP_CONFIG.inferenceMode) || 'client';
}

function setCameraStatus(message, hide = false) {
    const status = document.getElementById('camera-status');
    if (!status) return;
    if (hide) {
        status.classList.add('is-hidden');
        status.textContent = '';
        return;
    }
    status.classList.remove('is-hidden');
    status.textContent = message;
}

/**
 * Initialize camera (browser webcam or server MJPEG)
 */
function initializeVideoFeed() {
    if (getCameraSource() === 'browser') {
        initializeBrowserCamera();
        return;
    }

    const cameraFeed = document.getElementById('camera-feed');
    if (!cameraFeed) return;

    refreshVideoFeed();
    cameraFeed.onerror = function() {
        console.error('Video feed error, attempting reconnect...');
        handleVideoFeedError();
    };
    startVideoFeedMonitor();
}

/**
 * Use the visitor's webcam and send frames to the server for inference
 */
async function initializeBrowserCamera() {
    const video = document.getElementById('camera-video');
    if (!video) return;

    if (!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
        setCameraStatus('Camera requires HTTPS (or localhost). Set TRAFFIC=https with SSL certs, then reload.');
        showToast('Browser camera needs HTTPS on remote hosts', 'error', 8000);
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraStatus('Camera API not supported in this browser. Try Chrome.');
        return;
    }

    try {
        setCameraStatus('Requesting camera permission…');
        browserStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });
        video.srcObject = browserStream;
        await video.play();

        if (getInferenceMode() === 'client') {
            setCameraStatus('Loading on-device model…');
            await window.ClientInference.init();
            const annotated = document.getElementById('camera-annotated');
            if (annotated) annotated.hidden = true;
            setCameraStatus('', true);
            showToast('Camera ready — press Record to start detection', 'success', 3000);
        } else {
            setCameraStatus('', true);
            showToast('Camera ready — press Record to start detection', 'success', 3000);
        }
    } catch (error) {
        console.error('Browser camera error:', error);
        const msg = error.name === 'NotAllowedError'
            ? 'Camera permission denied. Allow camera access and refresh.'
            : `Could not open camera: ${error.message || error.name}`;
        setCameraStatus(msg);
        showToast(msg, 'error', 6000);
    }
}

function onClientInferenceUpdate(data) {
    applyPredictionUi({
        prediction: data.prediction,
        current_prediction: data.currentPrediction,
        raw_text: data.rawText,
    });

    if (data.addedCharacter) {
        // Keep server sentence in sync for stop/speak fallbacks
        fetch('/append_character', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ character: data.addedCharacter })
        }).catch(() => {});
    }
}

/**
 * Start detection only while recording is active.
 */
function startInferenceWhileRecording() {
    if (getCameraSource() !== 'browser') return;

    if (getInferenceMode() === 'client' && window.ClientInference) {
        const video = document.getElementById('camera-video');
        const overlay = document.getElementById('camera-overlay-canvas');
        if (!video || !overlay) return;
        window.ClientInference.setRecording(true);
        window.ClientInference.start(video, overlay, onClientInferenceUpdate);
        return;
    }

    // Server-side frame upload mode
    startBrowserPredictionLoop();
}

/**
 * Stop detection; keep webcam preview running.
 */
function stopInferenceWhileIdle() {
    stopBrowserPredictionLoop();

    if (window.ClientInference) {
        window.ClientInference.setRecording(false);
        window.ClientInference.stop();
    }

    const overlay = document.getElementById('camera-overlay-canvas');
    if (overlay) {
        const ctx = overlay.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
    }

    const annotated = document.getElementById('camera-annotated');
    if (annotated) {
        annotated.hidden = true;
        annotated.removeAttribute('src');
    }
}

function startBrowserPredictionLoop() {
    stopBrowserPredictionLoop();
    browserPredictTimer = setInterval(sendBrowserFrameForPrediction, BROWSER_PREDICT_INTERVAL_MS);
}

function stopBrowserPredictionLoop() {
    if (browserPredictTimer) {
        clearInterval(browserPredictTimer);
        browserPredictTimer = null;
    }
}

async function sendBrowserFrameForPrediction() {
    if (browserPredictInFlight) return;

    const video = document.getElementById('camera-video');
    const canvas = document.getElementById('camera-capture');
    if (!video || !canvas || video.readyState < 2) return;

    browserPredictInFlight = true;
    try {
        const maxWidth = 480;
        const scale = Math.min(1, maxWidth / video.videoWidth);
        const width = Math.max(1, Math.round(video.videoWidth * scale));
        const height = Math.max(1, Math.round(video.videoHeight * scale));
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, width, height);

        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.7));
        if (!blob) return;

        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        const response = await fetch('/predict_frame', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok || data.status === 'error') {
            throw new Error(data.message || 'Prediction failed');
        }

        const annotated = document.getElementById('camera-annotated');
        if (annotated && data.annotated_image) {
            annotated.src = data.annotated_image;
            annotated.hidden = false;
        }

        applyPredictionUi(data);
    } catch (error) {
        // Keep camera preview even if a single prediction fails
        console.warn('Frame prediction error:', error.message || error);
    } finally {
        browserPredictInFlight = false;
    }
}

/**
 * Refresh the camera feed
 */
function refreshVideoFeed() {
    if (getCameraSource() === 'browser') {
        stopBrowserPredictionLoop();
        if (window.ClientInference) {
            window.ClientInference.stop();
        }
        if (browserStream) {
            browserStream.getTracks().forEach(track => track.stop());
            browserStream = null;
        }
        const annotated = document.getElementById('camera-annotated');
        if (annotated) {
            annotated.hidden = true;
            annotated.removeAttribute('src');
        }
        initializeBrowserCamera();
        return;
    }

    const cameraFeed = document.getElementById('camera-feed');
    if (!cameraFeed) return;

    const timestamp = new Date().getTime();
    cameraFeed.src = `/video_feed?t=${timestamp}`;
    console.log('Video feed refreshed');
}

/**
 * Handle video feed errors with retry logic (server mode)
 */
function handleVideoFeedError() {
    if (getCameraSource() === 'browser') return;

    videoFeedRetryCount++;

    if (videoFeedRetryCount <= MAX_VIDEO_RETRIES) {
        console.log(`Retrying video feed (${videoFeedRetryCount}/${MAX_VIDEO_RETRIES})...`);
        showToast(`Camera reconnecting... (${videoFeedRetryCount}/${MAX_VIDEO_RETRIES})`, 'info', 2000);

        setTimeout(() => {
            refreshVideoFeed();
        }, VIDEO_RETRY_DELAY);
    } else {
        console.error('Max video feed retries reached');
        showToast('Camera connection lost. Please refresh the page.', 'error', 10000);
    }
}

/**
 * Monitor video feed for stale frames (server mode)
 */
function startVideoFeedMonitor() {
    const cameraFeed = document.getElementById('camera-feed');
    if (!cameraFeed || getCameraSource() === 'browser') return;

    let lastLoadTime = Date.now();

    cameraFeed.onload = function() {
        lastLoadTime = Date.now();
        videoFeedRetryCount = 0;
    };

    if (videoFeedCheckInterval) {
        clearInterval(videoFeedCheckInterval);
    }

    videoFeedCheckInterval = setInterval(() => {
        if (Date.now() - lastLoadTime > 10000) {
            console.warn('Video feed appears stale, refreshing...');
            refreshVideoFeed();
            lastLoadTime = Date.now();
        }
    }, 5000);
}

/**
 * Clean up video feed monitoring / browser camera
 */
function cleanupVideoFeed() {
    if (videoFeedCheckInterval) {
        clearInterval(videoFeedCheckInterval);
        videoFeedCheckInterval = null;
    }
    stopBrowserPredictionLoop();
    if (window.ClientInference) {
        window.ClientInference.stop();
    }
    if (browserStream) {
        browserStream.getTracks().forEach(track => track.stop());
        browserStream = null;
    }
}

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Set up character count listener
    const textInput = elements.textInput;
    if (textInput) {
        textInput.addEventListener('input', updateCharCount);

        // Auto-resize textarea (optional enhancement)
        textInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });
    }

    // Set up keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);

    // Initialize character count
    updateCharCount();

    // Initialize progress bar
    updateProgressBar(0, 0);

    // Initialize video feed with auto-reconnect
    initializeVideoFeed();

    // Add focus styles for keyboard navigation
    document.body.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-nav');
        }
    });

    document.body.addEventListener('mousedown', () => {
        document.body.classList.remove('keyboard-nav');
    });

    // Clean up on page unload
    window.addEventListener('beforeunload', cleanupVideoFeed);

    console.log('SignBridge initialized successfully');
});

// =============================================================================
// EXPOSE FUNCTIONS GLOBALLY (for onclick handlers)
// =============================================================================

window.startRecording = startRecording;
window.stopRecording = stopRecording;
window.clearTranslatedText = clearTranslatedText;
window.speakText = speakText;
window.convertText = convertText;
window.convertSpeech = convertSpeech;
window.hideToast = hideToast;
window.refreshVideoFeed = refreshVideoFeed;
