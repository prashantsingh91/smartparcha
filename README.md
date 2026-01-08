SmartParcha
===========

OCR + Gemini API for medical prescriptions.

Setup
-----
1. Create and activate virtual environment (already created as `.venv` by the assistant):

   ```bash
   cd /Users/prashantsingh/Documents/smartparcha
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Environment variables:

   - Set in `.env` (already created):
     - `GEMINI_API_KEY`

Running the API
---------------

Start the FastAPI server with Uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open the interactive docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

API
---

- **Health check**
  - **GET** `/health`

- **OCR prescription**
  - **POST** `/ocr/prescription`
  - Form-data: `file` (JPG, JPEG, PNG, or PDF)
  - Returns JSON:
    - `name`
    - `age`
    - `gender`
    - `phone_number`
    - `medicines_prescribed`
    - `tests_written`
    - `raw_text`

