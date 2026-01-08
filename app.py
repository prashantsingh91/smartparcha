import io
import os
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment or .env")

genai.configure(api_key=API_KEY)

app = FastAPI(
    title="SmartParcha OCR API",
    description=(
        "OCR and understanding for medical prescriptions using Gemini. "
        "Upload a JPG/PNG or PDF prescription to extract patient and medication details."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PrescriptionResponse(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    phone_number: Optional[str] = None
    medicines_prescribed: Optional[str] = None
    tests_written: Optional[str] = None
    raw_text: Optional[str] = None


SYSTEM_PROMPT = """
You are a senior medical document expert specializing in reading and understanding handwritten prescriptions.

Task:
- Carefully read the attached medical prescription image or PDF.
- Extract the following fields as accurately as possible:
  - name
  - age
  - gender
  - phone_number
  - medicines_prescribed
  - tests_written
- If some information is missing or unclear, set that field to null and do NOT hallucinate.

Output:
- Respond strictly as a compact JSON object with keys:
  {
    "name": string or null,
    "age": string or null,
    "gender": string or null,
    "phone_number": string or null,
    "medicines_prescribed": string or null,
    "tests_written": string or null,
    "raw_text": string or null  // your best-effort full transcription of the prescription
  }
- Do NOT add explanations, markdown, comments, or any text outside of the JSON.
"""


def _build_vision_model(model_name=None):
    """
    Build a vision model using Gemini 2.5.
    """
    # Default to Gemini 2.5 Flash if no model specified
    if model_name is None:
        model_name = "models/gemini-2.5-flash"
    
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        raise RuntimeError(f"Could not initialize Gemini model '{model_name}': {e}")


@app.post("/ocr/prescription", response_model=PrescriptionResponse)
async def ocr_prescription(file: UploadFile = File(...)):
    """
    Upload a medical prescription (JPG/PNG/PDF) and extract key fields.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload JPG, JPEG, PNG, or PDF.",
        )

    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Prepare file for Gemini
        mime_type = "application/pdf" if filename_lower.endswith(".pdf") else "image/jpeg"
        if filename_lower.endswith(".png"):
            mime_type = "image/png"

        uploaded_file = {
            "mime_type": mime_type,
            "data": data,
        }

        prompt = SYSTEM_PROMPT.strip()

        # Use Gemini 2.5 models (prioritize 2.5-flash for speed, 2.5-pro for accuracy)
        models_to_try = [
            "models/gemini-2.5-flash",      # Primary: Fast and efficient for vision tasks
            "models/gemini-2.5-pro",        # Fallback: More accurate but slower
        ]
        response = None
        last_error = None
        
        for model_name in models_to_try:
            try:
                model = _build_vision_model(model_name)
                response = model.generate_content(
                    [prompt, uploaded_file],
                    request_options={"timeout": 120},
                )
                break  # Success, exit loop
            except Exception as e:
                last_error = e
                continue
        
        if response is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process with any available model. Tried: {models_to_try}. Last error: {str(last_error)}. Visit /models to see available models."
            )

        text = response.text if hasattr(response, "text") else str(response)
        if not text:
            raise HTTPException(status_code=500, detail="Empty response from Gemini")

        # Attempt to parse JSON from the model output
        import json

        cleaned = text.strip()
        # Try to handle if model returns fenced code block
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # Strip potential language prefix like ```json
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.strip()

        try:
            data_json = json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: try to locate the first { .. } block
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                data_json = json.loads(cleaned[start : end + 1])
            else:
                raise

        # Map keys safely
        return PrescriptionResponse(
            name=data_json.get("name"),
            age=data_json.get("age"),
            gender=data_json.get("gender"),
            phone_number=data_json.get("phone_number"),
            medicines_prescribed=data_json.get("medicines_prescribed"),
            tests_written=data_json.get("tests_written"),
            raw_text=data_json.get("raw_text"),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to process prescription: {exc}") from exc


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    """List available Gemini models for debugging."""
    try:
        models = genai.list_models()
        model_info = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_info.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "supported_methods": list(model.supported_generation_methods)
                })
        return {"available_models": model_info}
    except Exception as e:
        return {"error": str(e), "message": "Could not list models"}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main UI page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartParcha - Prescription OCR</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 800px;
            width: 100%;
            padding: 40px;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-align: center;
        }
        
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 20px;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        
        .upload-area.dragover {
            border-color: #764ba2;
            background: #e8ebff;
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .upload-text {
            color: #667eea;
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .upload-hint {
            color: #999;
            font-size: 0.9em;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
            margin-top: 20px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .preview {
            margin-top: 30px;
            text-align: center;
        }
        
        .preview img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .results {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9ff;
            border-radius: 15px;
            display: none;
        }
        
        .results.show {
            display: block;
        }
        
        .results h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .result-item {
            background: white;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .result-label {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 5px;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }
        
        .result-value {
            color: #333;
            font-size: 1.1em;
        }
        
        .result-value:empty::before {
            content: "Not found";
            color: #999;
            font-style: italic;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        
        .error.show {
            display: block;
        }
        
        .file-name {
            margin-top: 15px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 SmartParcha</h1>
        <p class="subtitle">Upload a prescription to extract patient and medication information</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📄</div>
            <div class="upload-text">Click to upload or drag and drop</div>
            <div class="upload-hint">Supports JPG, JPEG, PNG, or PDF files</div>
            <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.pdf">
        </div>
        
        <button class="btn" id="submitBtn" disabled>Extract Information</button>
        
        <div class="file-name" id="fileName"></div>
        
        <div class="preview" id="preview"></div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing prescription with AI... This may take a moment.</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="results" id="results">
            <h2>📋 Extracted Information</h2>
            <div class="result-item">
                <div class="result-label">Patient Name</div>
                <div class="result-value" id="name"></div>
            </div>
            <div class="result-item">
                <div class="result-label">Age</div>
                <div class="result-value" id="age"></div>
            </div>
            <div class="result-item">
                <div class="result-label">Gender</div>
                <div class="result-value" id="gender"></div>
            </div>
            <div class="result-item">
                <div class="result-label">Phone Number</div>
                <div class="result-value" id="phone"></div>
            </div>
            <div class="result-item">
                <div class="result-label">Medicines Prescribed</div>
                <div class="result-value" id="medicines"></div>
            </div>
            <div class="result-item">
                <div class="result-label">Tests Written</div>
                <div class="result-value" id="tests"></div>
            </div>
            <div class="result-item">
                <div class="result-label">Raw Text</div>
                <div class="result-value" id="rawText" style="white-space: pre-wrap; font-size: 0.9em;"></div>
            </div>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const submitBtn = document.getElementById('submitBtn');
        const preview = document.getElementById('preview');
        const fileName = document.getElementById('fileName');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const error = document.getElementById('error');
        
        let selectedFile = null;
        
        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });
        
        function handleFile(file) {
            if (!file) return;
            
            const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/pdf'];
            if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|pdf)$/i)) {
                showError('Please upload a JPG, JPEG, PNG, or PDF file.');
                return;
            }
            
            selectedFile = file;
            fileName.textContent = `Selected: ${file.name}`;
            submitBtn.disabled = false;
            
            // Show preview for images
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                };
                reader.readAsDataURL(file);
            } else {
                preview.innerHTML = `<div style="padding: 20px; color: #667eea; font-size: 1.2em;">📄 PDF File: ${file.name}</div>`;
            }
            
            // Hide previous results
            results.classList.remove('show');
            error.classList.remove('show');
        }
        
        submitBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            // Show loading, hide results and error
            loading.classList.add('show');
            results.classList.remove('show');
            error.classList.remove('show');
            submitBtn.disabled = true;
            
            try {
                const response = await fetch('/ocr/prescription', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to process prescription');
                }
                
                const data = await response.json();
                
                // Display results
                document.getElementById('name').textContent = data.name || '';
                document.getElementById('age').textContent = data.age || '';
                document.getElementById('gender').textContent = data.gender || '';
                document.getElementById('phone').textContent = data.phone_number || '';
                document.getElementById('medicines').textContent = data.medicines_prescribed || '';
                document.getElementById('tests').textContent = data.tests_written || '';
                document.getElementById('rawText').textContent = data.raw_text || '';
                
                results.classList.add('show');
                loading.classList.remove('show');
                submitBtn.disabled = false;
                
            } catch (err) {
                showError(err.message || 'An error occurred while processing the prescription.');
                loading.classList.remove('show');
                submitBtn.disabled = false;
            }
        });
        
        function showError(message) {
            error.textContent = `❌ Error: ${message}`;
            error.classList.add('show');
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


