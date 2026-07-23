/**
 * Client-side ASL detection: MediaPipe Hands + local RandomForest.
 */

const ClientInference = (() => {
    let handLandmarker = null;
    let rafId = null;
    let lastVideoTime = -1;
    let running = false;

    const stabilityBuffer = [];
    let lastConfirmedChar = '';
    let lastDetectionTime = 0;
    let stableChar = '';
    let detectedSentence = [];
    let isRecording = false;

    const HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20],
        [5, 9], [9, 13], [13, 17],
    ];

    function config() {
        const cfg = window.APP_CONFIG || {};
        return {
            stabilityThreshold: cfg.stabilityThreshold || 5,
            stabilityTimeWindow: cfg.stabilityTimeWindow || 1.0,
            stabilizationDelay: cfg.stabilizationDelay || 2.0,
        };
    }

    async function init() {
        if (!window.SignClassifier) {
            throw new Error('SignClassifier missing');
        }
        await window.SignClassifier.load('/static/model/sign_rf.json');

        const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm');
        const { FilesetResolver, HandLandmarker } = vision;
        const fileset = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
        handLandmarker = await HandLandmarker.createFromOptions(fileset, {
            baseOptions: {
                modelAssetPath:
                    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numHands: 1,
            minHandDetectionConfidence: 0.3,
            minHandPresenceConfidence: 0.3,
            minTrackingConfidence: 0.3,
        });
        return true;
    }

    function setRecording(value) {
        isRecording = !!value;
        if (isRecording) {
            detectedSentence = [];
            stabilityBuffer.length = 0;
            lastConfirmedChar = '';
            lastDetectionTime = 0;
            stableChar = '';
        }
    }

    function getState() {
        return {
            stableChar,
            rawText: detectedSentence.join(' '),
            letters: [...detectedSentence],
            isRecording,
        };
    }

    function checkStability(prediction) {
        const { stabilityThreshold, stabilityTimeWindow } = config();
        const now = performance.now() / 1000;
        while (stabilityBuffer.length && now - stabilityBuffer[0][1] >= stabilityTimeWindow) {
            stabilityBuffer.shift();
        }
        stabilityBuffer.push([prediction, now]);
        if (stabilityBuffer.length >= stabilityThreshold) {
            const recent = stabilityBuffer.slice(-stabilityThreshold).map(x => x[0]);
            if (recent.every(p => p === recent[0])) {
                return recent[0];
            }
        }
        return null;
    }

    function processStable(prediction) {
        const { stabilizationDelay } = config();
        const now = performance.now() / 1000;
        stableChar = prediction;
        if (
            isRecording &&
            prediction !== lastConfirmedChar &&
            now - lastDetectionTime >= stabilizationDelay
        ) {
            detectedSentence.push(prediction);
            lastConfirmedChar = prediction;
            lastDetectionTime = now;
            return true;
        }
        return false;
    }

    function drawLandmarks(ctx, landmarks, width, height) {
        ctx.clearRect(0, 0, width, height);
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#00FF7F';
        ctx.fillStyle = '#FF1493';

        // Mirror X to match CSS-mirrored video; keep text unmirrored on this canvas
        const mx = (p) => (1 - p.x) * width;
        const my = (p) => p.y * height;

        for (const [a, b] of HAND_CONNECTIONS) {
            const pa = landmarks[a];
            const pb = landmarks[b];
            ctx.beginPath();
            ctx.moveTo(mx(pa), my(pa));
            ctx.lineTo(mx(pb), my(pb));
            ctx.stroke();
        }
        for (const p of landmarks) {
            ctx.beginPath();
            ctx.arc(mx(p), my(p), 4, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function wrapLine(ctx, text, maxWidth) {
        if (ctx.measureText(text).width <= maxWidth) return [text];

        const words = text.split('');
        // Fingerspelling text is often uninterrupted letters; wrap by characters
        const lines = [];
        let current = '';
        for (const ch of words) {
            const next = current + ch;
            if (current && ctx.measureText(next).width > maxWidth) {
                lines.push(current);
                current = ch;
            } else {
                current = next;
            }
        }
        if (current) lines.push(current);
        return lines.length ? lines : [text];
    }

    function drawHud(ctx, width, height, pred, isStable) {
        const rawLines = [
            `Char: ${stableChar || pred || '-'}`,
            `Text: ${detectedSentence.join('') || '-'}`,
            `Status: ${isStable ? 'Stable' : 'Unstable'}`,
        ];
        const paddingX = 10;
        const paddingY = 8;
        const lineHeight = 22;
        const maxBarWidth = Math.min(220, Math.floor(width * 0.42));
        const contentWidth = maxBarWidth - paddingX * 2;
        ctx.font = '600 15px Outfit, sans-serif';

        const displayLines = [];
        for (const line of rawLines) {
            const wrapped = wrapLine(ctx, line, contentWidth);
            for (let i = 0; i < wrapped.length; i++) {
                displayLines.push({
                    text: wrapped[i],
                    isStatus: line.startsWith('Status:'),
                });
            }
        }

        let textWidth = 0;
        for (const item of displayLines) {
            textWidth = Math.max(textWidth, ctx.measureText(item.text).width);
        }

        const barWidth = Math.min(maxBarWidth, Math.ceil(textWidth + paddingX * 2));
        const barHeight = paddingY * 2 + lineHeight * displayLines.length;
        const barX = width - barWidth - 12;
        const barY = 12;

        ctx.fillStyle = 'rgba(255, 255, 255, 0.72)';
        ctx.fillRect(barX, barY, barWidth, barHeight);

        let y = barY + paddingY + 14;
        for (const item of displayLines) {
            if (item.isStatus && !isStable) ctx.fillStyle = '#C62828';
            else ctx.fillStyle = '#C2185B';
            ctx.fillText(item.text, barX + paddingX, y);
            y += lineHeight;
        }
    }

    function start(video, canvas, onUpdate) {
        if (!handLandmarker || !window.SignClassifier.isReady()) {
            throw new Error('Client inference not initialized');
        }
        stop();
        running = true;

        const loop = () => {
            if (!running) return;
            rafId = requestAnimationFrame(loop);

            if (!video || video.readyState < 2) return;
            if (video.currentTime === lastVideoTime) return;
            lastVideoTime = video.currentTime;

            const nowMs = performance.now();
            const result = handLandmarker.detectForVideo(video, nowMs);
            const ctx = canvas.getContext('2d');
            const width = canvas.width = video.clientWidth || video.videoWidth;
            const height = canvas.height = video.clientHeight || video.videoHeight;

            let prediction = '';
            let added = false;
            let isStable = false;

            if (result.landmarks && result.landmarks.length > 0) {
                const landmarks = result.landmarks[0];
                drawLandmarks(ctx, landmarks, width, height);
                const pred = window.SignClassifier.predictFromLandmarks(landmarks);
                prediction = pred.label;
                const stable = checkStability(prediction);
                isStable = !!stable;
                if (stable) {
                    added = processStable(stable);
                }
            } else {
                ctx.clearRect(0, 0, width, height);
            }

            drawHud(ctx, width, height, prediction, isStable);

            if (typeof onUpdate === 'function') {
                onUpdate({
                    prediction: stableChar || prediction,
                    currentPrediction: prediction,
                    rawText: detectedSentence.join(' '),
                    letters: [...detectedSentence],
                    addedCharacter: added ? lastConfirmedChar : null,
                    isStable,
                    handDetected: !!(result.landmarks && result.landmarks.length),
                });
            }
        };

        rafId = requestAnimationFrame(loop);
    }

    function stop() {
        running = false;
        if (rafId) {
            cancelAnimationFrame(rafId);
            rafId = null;
        }
    }

    return { init, start, stop, setRecording, getState };
})();

window.ClientInference = ClientInference;
