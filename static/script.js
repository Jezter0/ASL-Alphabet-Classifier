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

// ==========================
// INIT MEDIAPIPE
// ==========================
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

// ==========================
// START / STOP CAMERA
// ==========================
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

// ==========================
// FRAME LOOP
// ==========================
async function detectFrame() {
    if (!running) return;

    const results = handLandmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        drawLandmarks(landmarks);

        const model = document.getElementById("model").value;

        if (model === "landmark") {
            sendLandmarks(landmarks);
        } else {
            processROI(landmarks);
        }
    }

    requestAnimationFrame(detectFrame);
}

// ==========================
// DRAW LANDMARKS
// ==========================
function drawLandmarks(landmarks) {
    ctx.fillStyle = "lime";
    landmarks.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 4, 0, 2 * Math.PI);
        ctx.fill();
    });
}

// ==========================
// LANDMARK → SERVER
// ==========================
function flattenLandmarks(landmarks) {
    const arr = [];
    landmarks.forEach(p => arr.push(p.x, p.y, p.z));
    return arr;
}

let lastPredict = 0;
const PREDICT_INTERVAL = 200;

function sendLandmarks(landmarks) {
    const now = performance.now();
    if (now - lastPredict < PREDICT_INTERVAL) return;
    lastPredict = now;

    const fd = new FormData();
    fd.append("model", "landmark");
    fd.append("landmarks", JSON.stringify(flattenLandmarks(landmarks)));

    fetch("/predict", {
        method: "POST",
        body: fd
    })
        .then(res => res.json())
        .then(updateUI);
}

// ==========================
// IMAGE MODEL PIPELINE
// ==========================
function processROI(landmarks) {
    const vidW = video.videoWidth;
    const vidH = video.videoHeight;

    const xs = landmarks.map(p => p.x * vidW);
    const ys = landmarks.map(p => p.y * vidH);

    let xmin = Math.min(...xs);
    let xmax = Math.max(...xs);
    let ymin = Math.min(...ys);
    let ymax = Math.max(...ys);

    const scale = 1.5;
    const w = xmax - xmin;
    const h = ymax - ymin;

    xmin = Math.max(0, xmin - w * (scale - 1) / 2);
    xmax = Math.min(vidW, xmax + w * (scale - 1) / 2);
    ymin = Math.max(0, ymin - h * (scale - 1) / 2);
    ymax = Math.min(vidH, ymax + h * (scale - 1) / 2);

    drawBBox(xmin, ymin, xmax, ymax);
    cropAndSend(xmin, ymin, xmax, ymax);
}

function drawBBox(xmin, ymin, xmax, ymax) {
    ctx.strokeStyle = "red";
    ctx.lineWidth = 3;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
}

function cropAndSend(xmin, ymin, xmax, ymax) {
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = xmax - xmin;
    cropCanvas.height = ymax - ymin;

    const cx = cropCanvas.getContext("2d");
    cx.drawImage(video, xmin, ymin, cropCanvas.width, cropCanvas.height, 0, 0, cropCanvas.width, cropCanvas.height);

    sendImage(cropCanvas);
}

function sendImage(canvas) {
    const now = performance.now();
    if (now - lastPredict < PREDICT_INTERVAL) return;
    lastPredict = now;

    canvas.toBlob(blob => {
        const fd = new FormData();
        fd.append("frame", blob, "frame.jpg");
        fd.append("model", document.getElementById("model").value);

        fetch("/predict", {
            method: "POST",
            body: fd
        })
            .then(res => res.json())
            .then(updateUI);
    }, "image/jpeg", 0.8);
}

// ==========================
// UI UPDATE
// ==========================
function updateUI(data) {
    const letter = data.prediction || "—";
    const conf = data.confidence ? (data.confidence * 100).toFixed(1) : 0;

    document.getElementById("prediction").innerHTML =
        `${letter} <span style="font-size:0.4em; color:#555;">(${conf}%)</span>`;

    document.getElementById("confBar").style.width = conf + "%";
}

// ==========================
document.getElementById("startBtn").onclick = toggleCamera;