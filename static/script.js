import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

let handLandmarker;
let video = document.getElementById("cam");
let canvas = document.getElementById("overlay");
let ctx = canvas.getContext("2d");
let running = false;
let stream;

async function initMP() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        },
        runningMode: "VIDEO",
        numHands: 1
    });

    console.log("MediaPipe Initialized");
}

// -------------------------
// START / STOP CAMERA
// -------------------------
async function toggleCamera() {
    if (!running) {
        await startCamera();
        startBtn.textContent = "Stop Camera";
        startBtn.classList.add("btn-danger");
    } else {
        stopCamera();
        startBtn.textContent = "Start Camera";
        startBtn.classList.remove("btn-danger");
    }
}

async function startCamera() {
    await initMP();

    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    video.onloadeddata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        running = true;
        detectFrame();
    };
}

function stopCamera() {
    running = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("prediction").textContent = "—";
    document.getElementById("confBar").style.width = "0%";
}

// -------------------------
// FRAME LOOP
// -------------------------
async function detectFrame() {
    if (!running) return;

    const results = handLandmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.landmarks && results.landmarks.length > 0) {
        const lm = results.landmarks[0];
        drawLandmarks(lm);
        processROI(lm);
    }

    requestAnimationFrame(detectFrame);
}

function drawLandmarks(landmarks) {
    ctx.fillStyle = "lime";
    landmarks.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 4, 0, 2 * Math.PI);
        ctx.fill();
    });
}

// -------------------------
// FIXED CROPPING (ACCURATE)
// -------------------------
function processROI(landmarks) {
    const vidW = video.videoWidth;
    const vidH = video.videoHeight;

    const xs = landmarks.map(p => p.x * vidW);
    const ys = landmarks.map(p => p.y * vidH);

    let xmin = Math.min(...xs);
    let xmax = Math.max(...xs);
    let ymin = Math.min(...ys);
    let ymax = Math.max(...ys);

    const hand_w = xmax - xmin;
    const hand_h = ymax - ymin;
    const scale = 1.5;
    xmin = Math.max(0, xmin - hand_w * (scale - 1) / 2);
    xmax = Math.min(vidW, xmax + hand_w * (scale - 1) / 2);
    ymin = Math.max(0, ymin - hand_h * (scale - 1) / 2);
    ymax = Math.min(vidH, ymax + hand_h * (scale - 1) / 2);
    drawBBox(xmin, ymin, xmax, ymax);
    cropAndSend(xmin, ymin, xmax, ymax);
}

function drawBBox(xmin, ymin, xmax, ymax) {
    ctx.strokeStyle = "red";
    ctx.lineWidth = 3;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
}

function cropAndSend(xmin, ymin, xmax, ymax) {
    const w = xmax - xmin;
    const h = ymax - ymin;

    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = w;
    cropCanvas.height = h;

    const cx = cropCanvas.getContext("2d");
    cx.drawImage(video, xmin, ymin, w, h, 0, 0, w, h);

    prepareForPrediction(cropCanvas);
}

// -------------------------
let lastPredict = 0;
const PREDICT_INTERVAL = 200;
function prepareForPrediction(cropCanvas) {
    const now = performance.now();
    if (now - lastPredict < PREDICT_INTERVAL) return;
    lastPredict = now;
    const TARGET = 224;
    const modelSelect = document.getElementById("model");

    const resizedCanvas = document.createElement("canvas");
    resizedCanvas.width = TARGET;
    resizedCanvas.height = TARGET;

    const rc = resizedCanvas.getContext("2d");
    rc.drawImage(cropCanvas, 0, 0, TARGET, TARGET);

    resizedCanvas.toBlob(blob => {
        const fd = new FormData();
        fd.append("frame", blob, "frame.jpg");
        fd.append("model", modelSelect.value);

        fetch("/predict", {
            method: "POST",
            body: fd
        })
        .then(res => res.json())
        .then(data => {
            const letter = data.prediction || "—";
            const conf = data.confidence ? (data.confidence * 100).toFixed(1) : 0;

            document.getElementById("prediction").innerHTML =
                `${letter} <span style="font-size:0.4em; color:#555;">(${conf}%)</span>`;

            document.getElementById("confBar").style.width = conf + "%";
        });
    }, "image/jpeg", 0.8);
}

document.getElementById("startBtn").onclick = toggleCamera;