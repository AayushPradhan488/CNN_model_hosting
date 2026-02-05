const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!drawing) return;
    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(e.offsetX, e.offsetY, 10, 0, Math.PI * 2);
    ctx.fill();
}

function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function predictCanvas() {
    const dataURL = canvas.toDataURL("image/png");

    fetch("/predict-canvas", {
        method: "POST",
        body: JSON.stringify({ image: dataURL }),
        headers: { "Content-Type": "application/json" }
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerText =
            `Prediction: ${data.prediction} (confidence: ${(data.confidence * 100).toFixed(2)}%)`;
    });
}

function predictUpload() {
    const fileInput = document.getElementById("uploadInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a file first");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict-upload", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("uploadResult").innerText =
            `Prediction: ${data.prediction} (confidence: ${(data.confidence * 100).toFixed(2)}%)`;
    })
    .catch(err => {
        console.error(err);
        alert("Prediction failed");
    });
}
