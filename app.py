import io
import os
import re
import base64
import logging
from typing import Optional, List, Tuple, Any
from urllib.parse import urlparse

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Body, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler to avoid UnicodeDecodeError when validation errors contain bytes.
    We replace any bytes objects in the error details with a safe placeholder.
    """

    def safe_encode(obj: Any):
        if isinstance(obj, bytes):
            # Don't try to decode arbitrary binary; just mark it
            return "<binary data>"
        if isinstance(obj, dict):
            return {k: safe_encode(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [safe_encode(v) for v in obj]
        return obj

    safe_errors = safe_encode(exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": safe_errors},
    )


class ParchaContent(BaseModel):
    pageNumber: int
    content: str  # base64 or URL
    createdAt: str

class PrescriptionRequest(BaseModel):
    doctorId: str
    patientId: str
    visitId: str
    parchaContent: List[ParchaContent]
    history: List = []
    tests: List[str]
    status: str

class PrescriptionResponse(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    phone_number: Optional[str] = None
    medicines_prescribed: Optional[str] = None
    tests_written: Optional[List[str]] = None
    raw_text: Optional[str] = None
    matched_tests: Optional[List[str]] = None


def build_system_prompt(available_tests: List[str]) -> str:
    """
    Build system prompt with available tests list for Gemini to match against.
    """
    tests_list_str = "\n".join([f"- {test}" for test in available_tests])
    
    return f"""
You are a senior medical document expert specializing in reading and understanding handwritten prescriptions.

Task:
- Carefully read the attached medical prescription image or PDF.
- Extract the following fields as accurately as possible:
  - name
  - age
  - gender
  - phone_number
  - medicines_prescribed
  - tests_written (extract ALL test names mentioned in the prescription as written)
- If some information is missing or unclear, set that field to null and do NOT hallucinate.

IMPORTANT - Test Matching:
You will be provided with a list of available tests. Your task is to identify which tests from this list are mentioned in the prescription.

CRITICAL RULES:
1. Return ONLY the EXACT test names from the available tests list below - copy them exactly as written
2. Use your medical knowledge to match abbreviations and variations:
   - "KFT" or "Kidney Function" → return "KIDNEY FUNCTION TEST" (exact name from list)
   - "LFT" or "Liver Function" → return "LIVER FUNCTION TEST" (exact name from list)
   - "Glucose Fasting" or "FBS" → return "Glucose fasting" (exact name from list)
   - "PPBS" or "Post Prandial Glucose" → return "Glucose post prandial" (exact name from list)
   - "Urea" or "BUN" → return "UREA" (exact name from list)
3. If you find tests in the prescription but they do NOT match any test in the available list, return an EMPTY array [] for matched_tests
4. Do NOT force matches or guess
5. Do NOT return variations or your own interpretations - ONLY return exact names from the list below

Available Tests List (return EXACT names only):
{tests_list_str}

Examples:
- Prescription says "KFT" → matched_tests: ["KIDNEY FUNCTION TEST"]
- Prescription says "LFT, Glucose" → matched_tests: ["LIVER FUNCTION TEST", "Glucose fasting"]
- Prescription says "CBC" but CBC is not in the list → matched_tests: []
- Prescription says "KFT, LFT, Urea" → matched_tests: ["KIDNEY FUNCTION TEST", "LIVER FUNCTION TEST", "UREA"]

Output:
- Respond strictly as a compact JSON object with keys:
  {{
    "name": string or null,
    "age": string or null,
    "gender": string or null,
    "phone_number": string or null,
    "medicines_prescribed": string or null,
    "tests_written": string or null,  // comma-separated list of all tests found as written in prescription
    "matched_tests": array or [],  // array of EXACT test names from the available tests list above. Return [] if no matches found.
    "raw_text": string or null  // your best-effort full transcription of the prescription
  }}
- Do NOT add explanations, markdown, comments, or any text outside of the JSON.
- For matched_tests, return ONLY the exact test names from the available tests list (copy them exactly as shown above).
- If no matches are found, return an empty array [] - do NOT try to force matches.
- Use your medical expertise to match abbreviations and variations correctly, but return ONLY the exact names from the list.
"""


def normalize_test_name(test: str) -> str:
    """
    Normalize test name for comparison (case-insensitive, trim whitespace).
    """
    return test.strip().upper()


def validate_and_filter_matched_tests(
    gemini_matched_tests: List[str], 
    available_tests: List[str]
) -> List[str]:
    """
    Validate that matched tests from Gemini exist in the available tests list.
    Returns only tests that exactly match (case-insensitive) the available tests.
    """
    if not gemini_matched_tests or not available_tests:
        return []
    
    # Create a normalized lookup map: normalized_name -> original_name
    available_tests_map = {}
    for test in available_tests:
        normalized = normalize_test_name(test)
        # Store the original name for exact matching
        if normalized not in available_tests_map:
            available_tests_map[normalized] = test
    
    validated_tests = []
    for gemini_test in gemini_matched_tests:
        if not gemini_test or not gemini_test.strip():
            continue
        
        normalized_gemini = normalize_test_name(gemini_test)
        
        # Check if normalized version exists in available tests
        if normalized_gemini in available_tests_map:
            # Return the exact original name from available_tests
            original_name = available_tests_map[normalized_gemini]
            if original_name not in validated_tests:
                validated_tests.append(original_name)
                logger.info(f"Validated test match: '{gemini_test}' → '{original_name}'")
        else:
            logger.warning(f"Gemini returned test '{gemini_test}' which is not in available tests list. Filtering out.")
    
    return validated_tests


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


def decode_base64_image(base64_string: str) -> Tuple[bytes, str]:
    """Decode base64 image string and return (data, mime_type)."""
    # Remove data URL prefix if present
    if base64_string.startswith("data:image/"):
        header, data = base64_string.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
    else:
        data = base64_string
        mime_type = "image/png"  # Default
    
    image_data = base64.b64decode(data)
    return image_data, mime_type


async def download_image_from_url(url: str) -> Tuple[bytes, str]:
    """Download image from URL and return (data, mime_type)."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        
        # Determine mime type from content-type or URL
        content_type = response.headers.get("content-type", "image/png")
        if not content_type.startswith("image/"):
            # Try to infer from URL
            if url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"):
                content_type = "image/jpeg"
            elif url.lower().endswith(".png"):
                content_type = "image/png"
            else:
                content_type = "image/png"  # Default
        
        return response.content, content_type




@app.post("/ocr/prescription", response_model=PrescriptionResponse)
async def ocr_prescription(request: PrescriptionRequest = Body(...)):
    """
    Process medical prescription from parchaContent and extract key fields.
    Supports multiple pages with base64 or URL images.
    """
    if not request.parchaContent:
        raise HTTPException(status_code=400, detail="No parchaContent provided")

    try:
        import json
        
        # Process all pages and combine results
        all_results = []
        combined_raw_text = []
        
        for page in request.parchaContent:
            # Determine if content is base64 or URL
            if page.content.startswith("data:image/") or page.content.startswith("http://") or page.content.startswith("https://"):
                if page.content.startswith("http://") or page.content.startswith("https://"):
                    # Download from URL
                    data, mime_type = await download_image_from_url(page.content)
                else:
                    # Decode base64
                    data, mime_type = decode_base64_image(page.content)
            else:
                # Assume base64 without prefix
                try:
                    data = base64.b64decode(page.content)
                    mime_type = "image/png"
                except Exception:
                    raise HTTPException(status_code=400, detail=f"Invalid image format for page {page.pageNumber}")

            uploaded_file = {
                "mime_type": mime_type,
                "data": data,
            }

            # Build prompt with available tests list for Gemini to match
            prompt = build_system_prompt(request.tests).strip()

            # Use Gemini 2.5 models
            models_to_try = [
                "models/gemini-2.5-flash",
                "models/gemini-2.5-pro",
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
                    break
                except Exception as e:
                    last_error = e
                    logger.error(f"Error with model {model_name} for page {page.pageNumber}: {str(e)}")
                    continue
            
            if response is None:
                error_msg = str(last_error) if last_error else "Unknown error"
                logger.error(f"All models failed for page {page.pageNumber}. Last error: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process page {page.pageNumber}. Last error: {error_msg}"
                )

            # Check if response has error
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                if hasattr(response.prompt_feedback, "block_reason") and response.prompt_feedback.block_reason:
                    logger.error(f"Response blocked for page {page.pageNumber}: {response.prompt_feedback}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Gemini blocked the response for page {page.pageNumber}. Reason: {response.prompt_feedback.block_reason}"
                    )

            # Extract text from response
            try:
                if hasattr(response, "text"):
                    text = response.text
                elif hasattr(response, "candidates") and response.candidates:
                    if hasattr(response.candidates[0], "content"):
                        text = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts else str(response)
                    else:
                        text = str(response)
                else:
                    text = str(response)
            except Exception as e:
                logger.error(f"Error extracting text from response for page {page.pageNumber}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to extract text from Gemini response for page {page.pageNumber}: {str(e)}"
                )
            if not text:
                logger.warning(f"Empty response from Gemini for page {page.pageNumber}")
                continue  # Skip empty responses

            # Log the raw response for debugging
            logger.info(f"Gemini response for page {page.pageNumber} (first 500 chars): {text[:500]}")

            # Check if response is an error message
            if text.strip().startswith("Error") or text.strip().startswith("Internal") or "error" in text.lower()[:100]:
                logger.error(f"Gemini returned error message for page {page.pageNumber}: {text[:200]}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini API error for page {page.pageNumber}: {text[:200]}"
                )

            # Parse JSON from response
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.strip()

            try:
                page_data = json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for page {page.pageNumber}. Response: {text[:500]}")
                # Try to extract JSON from the response
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        page_data = json.loads(cleaned[start : end + 1])
                    except json.JSONDecodeError:
                        logger.error(f"Could not parse JSON from page {page.pageNumber}. Error: {str(e)}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to parse Gemini response for page {page.pageNumber}. Response was not valid JSON: {text[:200]}"
                        )
                else:
                    logger.error(f"No JSON found in response for page {page.pageNumber}. Response: {text[:200]}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Gemini response for page {page.pageNumber} was not valid JSON: {text[:200]}"
                    )
            
            all_results.append(page_data)
            if page_data.get("raw_text"):
                combined_raw_text.append(f"Page {page.pageNumber}:\n{page_data.get('raw_text')}")

        if not all_results:
            raise HTTPException(status_code=500, detail="No valid data extracted from any page")

        # Combine results from all pages (take first non-null value for each field)
        combined_result = {
            "name": None,
            "age": None,
            "gender": None,
            "phone_number": None,
            "medicines_prescribed": None,
            "tests_written": None,
            "matched_tests": None,  # Start with None, only set if Gemini finds matches
            "raw_text": "\n\n".join(combined_raw_text) if combined_raw_text else None,
        }

        matched_tests_from_gemini = []  # Collect all matched tests from Gemini
        
        for result in all_results:
            for key in ["name", "age", "gender", "phone_number", "medicines_prescribed", "tests_written", "raw_text"]:
                if combined_result[key] is None and result.get(key):
                    combined_result[key] = result.get(key)
                elif combined_result[key] and result.get(key) and key in ["medicines_prescribed", "tests_written"]:
                    # Combine medicines and tests
                    combined_result[key] = f"{combined_result[key]}, {result.get(key)}"
            
            # Collect matched_tests from all pages - only use what Gemini explicitly matched
            if result.get("matched_tests"):
                if isinstance(result["matched_tests"], list):
                    for test in result["matched_tests"]:
                        if test and test.strip():  # Only add non-empty tests
                            matched_tests_from_gemini.append(test.strip())
                elif isinstance(result["matched_tests"], str):
                    # Handle case where Gemini returns string instead of array
                    tests = [t.strip() for t in result["matched_tests"].split(",") if t.strip()]
                    matched_tests_from_gemini.extend(tests)
        
        # Validate and filter matched tests against the request's available tests list
        # This ensures we only return exact names from the provided test list
        if matched_tests_from_gemini and request.tests:
            logger.info(f"Gemini matched {len(matched_tests_from_gemini)} tests: {matched_tests_from_gemini}")
            final_matched_tests = validate_and_filter_matched_tests(
                matched_tests_from_gemini, 
                request.tests
            )
            logger.info(f"After validation, {len(final_matched_tests)} tests match: {final_matched_tests}")
        else:
            final_matched_tests = None
        
        # Only set matched_tests if we have validated matches, otherwise return None
        if not final_matched_tests:
            final_matched_tests = None

        # Convert tests_written to a list of strings for the response model
        tests_written_value = combined_result.get("tests_written")
        tests_written_list: Optional[List[str]] = None
        if isinstance(tests_written_value, str):
            # Split on common delimiters: comma, semicolon, newline
            tests_written_list = [
                t.strip() for t in re.split(r"[,;\n]", tests_written_value) if t.strip()
            ]
        elif isinstance(tests_written_value, list):
            tests_written_list = [str(t).strip() for t in tests_written_value if str(t).strip()]
        else:
            tests_written_list = None

        return PrescriptionResponse(
            name=combined_result.get("name"),
            age=combined_result.get("age"),
            gender=combined_result.get("gender"),
            phone_number=combined_result.get("phone_number"),
            medicines_prescribed=combined_result.get("medicines_prescribed"),
            tests_written=tests_written_list,
            raw_text=combined_result.get("raw_text"),
            matched_tests=final_matched_tests,  # Will be None if no matches found by Gemini
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
                <div class="result-label">Tests Written (Detected)</div>
                <div class="result-value" id="tests"></div>
            </div>
            <div class="result-item">
                <div class="result-label">Matched Tests (From Database)</div>
                <div class="result-value" id="matchedTests"></div>
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
        let selectedDataUrl = null; // data:image/...;base64,... for backend JSON API
        
        // Example tests list for demo usage
        const defaultTests = [
            'LIVER FUNCTION TEST',
            'Glucose fasting',
            'UREA',
            'Glucose post prandial',
            'KIDNEY FUNCTION TEST'
        ];
        
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
            selectedDataUrl = null;
            fileName.textContent = `Selected: ${file.name}`;
            submitBtn.disabled = false;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                selectedDataUrl = e.target.result; // data URL for backend
                
                if (file.type.startsWith('image/')) {
                    preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                } else {
                    preview.innerHTML = `<div style="padding: 20px; color: #667eea; font-size: 1.2em;">📄 PDF File: ${file.name}</div>`;
                }
            };
            reader.readAsDataURL(file);
            
            // Hide previous results
            results.classList.remove('show');
            error.classList.remove('show');
        }
        
        submitBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            if (!selectedDataUrl) {
                showError('File is not ready yet. Please wait a moment and try again.');
                return;
            }
            
            // Build JSON body as expected by the backend
            const body = {
                doctorId: 'demo-doctor',
                patientId: 'demo-patient',
                visitId: 'demo-visit',
                parchaContent: [
                    {
                        pageNumber: 1,
                        content: selectedDataUrl,
                        createdAt: new Date().toISOString()
                    }
                ],
                history: [],
                tests: defaultTests,
                status: 'saved'
            };
            
            // Show loading, hide results and error
            loading.classList.add('show');
            results.classList.remove('show');
            error.classList.remove('show');
            submitBtn.disabled = true;
            
            try {
                const response = await fetch('/ocr/prescription', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(body)
                });
                
                if (!response.ok) {
                    let message = 'Failed to process prescription';
                    try {
                        const errorData = await response.json();
                        if (errorData && errorData.detail) {
                            if (typeof errorData.detail === 'string') {
                                message = errorData.detail;
                            } else if (Array.isArray(errorData.detail) && errorData.detail.length > 0) {
                                // FastAPI validation errors: take first message
                                const first = errorData.detail[0];
                                if (first && (first.msg || first.message)) {
                                    message = first.msg || first.message;
                                } else {
                                    message = JSON.stringify(errorData.detail);
                                }
                            } else if (typeof errorData.detail === 'object') {
                                message = JSON.stringify(errorData.detail);
                            }
                        }
                    } catch (e) {
                        // Ignore JSON parse errors and use default message
                    }
                    throw new Error(message);
                }
                
                const data = await response.json();
                
                // Display results
                document.getElementById('name').textContent = data.name || '';
                document.getElementById('age').textContent = data.age || '';
                document.getElementById('gender').textContent = data.gender || '';
                document.getElementById('phone').textContent = data.phone_number || '';
                document.getElementById('medicines').textContent = data.medicines_prescribed || '';
                
                // Display detected tests (as written in prescription)
                const testsWritten = Array.isArray(data.tests_written) 
                    ? data.tests_written.join(', ') 
                    : (data.tests_written || '');
                document.getElementById('tests').textContent = testsWritten;
                
                // Display matched tests (from database list)
                const matchedTests = Array.isArray(data.matched_tests) 
                    ? data.matched_tests.join(', ') 
                    : (data.matched_tests ? 'No matches found' : 'No tests detected');
                document.getElementById('matchedTests').textContent = matchedTests || 'No matches found';
                
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


