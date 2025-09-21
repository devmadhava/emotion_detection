const labels = ["negative", "neutral", "positive"];
const colors  = ["red", "#0068FF", "green"];

let session = null;
let selectedImage = null;

document.getElementById("loader").style.display = "flex";
document.getElementById("app").style.display = "none";

// Load ONNX model on page load
async function loadModel() {
    // document.getElementById("loader").style.display = "flex";
    // document.getElementById("app").style.display = "none";
    const loader = document.getElementById("loader");
    const app = document.getElementById("app");
    const progressFill = document.getElementById("progressFill");

    loader.style.display = "flex";
    app.style.display = "none";

    const response = await fetch("model/emotion_model.onnx.gz");
    if (!response.ok) throw new Error("Failed to fetch model");

    // Progress Bar Download Logic
    const contentLength = response.headers.get("content-length");
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    const reader = response.body.getReader();
    let received = 0;
    let chunks = [];

    // Changing width
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;

        if (total) {
            const percent = Math.round((received / total) * 100);
            progressFill.style.width = percent + "%";
        }
    }

    // Merge chunks
    // Load Modal
    let compressedData = new Uint8Array(received);
    let position = 0;
    for (let chunk of chunks) {
        compressedData.set(chunk, position);
        position += chunk.length;
    }

    // const compressedData = await response.arrayBuffer();
    // Decompress
    const decompressedData = pako.ungzip(new Uint8Array(compressedData));

    // Create ONNX session from bytes
    session = await ort.InferenceSession.create(decompressedData);
    console.log("ONNX model loaded");

    loader.style.display = "none";
    app.style.display = "flex";
}

loadModel();

// Preview uploaded image
document.getElementById("fileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    
    if (file) {
        document.getElementById("fileName").textContent = file.name;

        const reader = new FileReader();
        reader.onload = function (ev) {
            const img = document.getElementById("preview");
            img.src = ev.target.result;
            selectedImage = img;
        };
        reader.readAsDataURL(file);
    }
});

// Run ONNX model on click
document.getElementById("runBtn").addEventListener("click", async () => {
    if (!selectedImage) {
        alert("Please upload an image first.");
        return;
    }

    const tensor = await imageToTensor(selectedImage, 224, 224);
    const predictions = await runModel(tensor);

    const expVals = predictions.map((v) => Math.exp(v));
    const sumExp = expVals.reduce((a, b) => a + b, 0);
    const probs = expVals.map((v) => v / sumExp);

    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const resultText = `\nProbabilities:\n` + 
    labels.map((label, i) => `${label}: ${(probs[i] * 100).toFixed(2)}%`).join("\n");

    document.getElementById("result").style.color = colors[maxIndex];
    document.getElementById("result").textContent = `${labels[maxIndex]}`;
    document.getElementById("output").textContent = resultText;
});

async function imageToTensor(img, width, height) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = width;
    canvas.height = height;

    ctx.drawImage(img, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height).data;
    const floatArray = new Float32Array(3 * width * height);

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < width * height; i++) {
        floatArray[i] = (imageData[i * 4] / 255.0 - mean[0]) / std[0]; // R
        floatArray[i + width * height] =
            (imageData[i * 4 + 1] / 255.0 - mean[1]) / std[1]; // G
        floatArray[i + 2 * width * height] =
            (imageData[i * 4 + 2] / 255.0 - mean[2]) / std[2]; // B
    }

    console.log("Tranformed")
    return new ort.Tensor("float32", floatArray, [1, 3, height, width]);
}


async function runModel(imageTensor) {
    document.getElementById("runBtn").textContent = "Loading...";

    const feeds = { input: imageTensor };
    const results = await session.run(feeds); // Use already loaded session

    document.getElementById("runBtn").textContent = "Run Model";
    return Array.from(results["output"].data);
}