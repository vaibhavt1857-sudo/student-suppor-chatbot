# Student Support Chatbot: End-to-End AI-Driven Academic & Campus Assistance System
## 📌 Main Project File
- **run.py**  
- To run:  
  ```bash
  python run.py

## Executive Summary & System Overview

The **Student Support Chatbot** is a production-grade, multi-modal artificial intelligence platform architected specifically to streamline academic support, student administrative queries, and campus navigation. Built on a modular Python Flask architecture, the system integrates fine-tuned Transformer-based Question Answering (DistilBERT), state-of-the-art Generative AI (Google Gemini API), Computer Vision (OCR and document analysis), and Speech Processing (Speech-to-Text and Text-to-Speech) into a cohesive enterprise-ready platform.

By combining deterministic local intent matching, deep contextual Question Answering via locally trained weights, and generative fallback capabilities, the system achieves near-zero hallucination rates for critical campus data while retaining high conversational flexibility.

---

## Complete Directory & Project Architecture

The project enforces strict separation of concerns through an Application Factory pattern, modular routing, isolated model instances, and decoupled configuration management. Below is the full structural layout of the application repository.

```text
student-support-chatbot/
├── app/                        # Primary Application Package
│   ├── __init__.py             # Flask application factory and extension initialization
│   ├── database.py             # Database connector, session lifecycle, and query handlers
│   ├── image_processing.py     # OCR extraction, preprocessing pipeline, and image analysis
│   ├── modules.py              # Auxiliary domain logic and business helper functions
│   ├── routes.py               # Blueprint endpoints, request routing, and response serialization
│   └── speech.py               # Audio processing, STT (Speech-to-Text) & TTS engine integration
│
├── chatbot_project/            # Isolated Sub-domain Data & Extensions
│   ├── data/                   # Project-specific supplemental datasets
│   └── modules/                # Contextual helper scripts and custom domain logic
│
├── config/                     # Centralized Configuration Management
│   ├── app.txt                 # Application runtime notes and deployment logs
│   ├── constants.py            # System-wide static variables and magic number declarations
│   ├── jb_config.json          # Environment-specific JSON properties
│   ├── logging_config.py       # Rotating file logger setup and formatters
│   └── settings.py             # Main Flask configuration and environment variables
│
├── data/                       # Operational Intent Sets & Rule Repositories
│   └── intents.json            # Tagged intent schemas, response patterns, and triggers
│
├── models/                     # Deep Learning Model Artifacts & Analytics
│   ├── distilbert_qa/          # Saved HuggingFace DistilBERT model directory
│   │   ├── config.json         # Transformer architecture hyperparameters
│   │   ├── pytorch_model.bin   # Fine-tuned PyTorch tensor weights
│   │   └── tokenizer.json      # BPE/WordPiece Tokenizer vocabulary mapping
│   └── training_report.txt     # Loss curves, accuracy metrics, and evaluation reports
│
├── static/                     # Web Presentation Layer (Static Assets)
│   ├── css/
│   │   └── style.css           # Modular UI stylesheets and responsive grid system
│   ├── js/
│   │   └── filechat.js         # AJAX engine, WebSocket events, and audio stream handling
│   ├── DSS/                    # Dynamic decision support system resources
│   ├── images/                 # App assets, icons, and UI illustration banners
│   └── JS/                     # Extended client-side utility libraries
│
├── templates/                  # Jinja2 Frontend View Templates
│   ├── about.html              # Project credits, team info, and architecture overview
│   ├── dashboard.html          # Student analytics, query logs, and account metrics
│   ├── features.html           # Interactive capability showcase panel
│   ├── index.html              # Primary conversational chat interface
│   ├── join.html               # Registration, authentication, and onboarding portal
│   └── news.html               # Real-time campus announcements and bulletin board
│
├── tests/                      # Automated Quality Assurance Test Suite
│   ├── test_app.py             # Route integration and response code tests
│   ├── test_models.py          # PyTorch inference, tensor output, and accuracy validation
│   └── test_preprocess.py      # Data cleaning, normalization, and tokenization tests
│
├── gemini_integration.py       # Google Gemini API wrapper, prompt templates, and streaming
├── train_distilbert_qa.py      # PyTorch fine-tuning loop for DistilBERT Question-Answering
├── verify_distilbert_qa.py     # Post-training model validation and benchmark suite
└── run.py                      # Main application execution entry point
```

---

## Detailed Component & Module Deep Dive

### 1. `app/` Directory — Core Backend Engine

#### `app/__init__.py` (Application Factory)
*   **Purpose**: Implements the Flask Application Factory pattern (`create_app()`).
*   **Key Operations**: Initializes application instances, registers operational Blueprints (`routes.py`), configures global CORS settings, binds database instances, and attaches custom error handlers (404, 500, 413 File Too Large).
*   **Design Rationale**: Decouples application creation from global state, allowing isolated testing fixtures in `tests/test_app.py` without mutating production state.

#### `app/database.py` (Persistence & Session Layer)
*   **Purpose**: Abstracts all interaction with the persistence layer (SQLite/PostgreSQL/MySQL).
*   **Key Operations**: Manages database connection lifecycles, user authentication states, query log archiving, intent analytics tracking, and conversation thread persistence.
*   **Security Features**: Enforces strict parameterized query execution to eliminate SQL Injection (SQLi) vulnerabilities.

#### `app/image_processing.py` (Computer Vision & OCR Engine)
*   **Purpose**: Provides multi-modal visual document parsing capabilities.
*   **Key Operations**: Receives uploaded image streams (PNG, JPEG), applies preprocessing pipelines including grayscale conversion, adaptive Gaussian thresholding, and deskewing, and then executes Optical Character Recognition (OCR) using Tesseract / OpenCV drivers.
*   **Use Cases**: Allows students to upload photographs of physical class schedules, library notices, or handwritten exam circulars and receive immediate text parsing and conversational Q&A on the extracted content.

#### `app/speech.py` (Voice Processing & Audio I/O)
*   **Purpose**: Manages bi-directional audio communications.
*   **Key Operations**: Performs Speech-to-Text (STT) conversion on WebM/WAV audio blobs captured via client browser WebRTC drivers, normalizing incoming speech into text tokens. Synthesizes outgoing text responses into natural vocal audio streams using Text-to-Speech (TTS) drivers.

#### `app/routes.py` (HTTP Routing & Endpoint Controllers)
*   **Purpose**: Contains the Flask route controllers and API request handler logic.
*   **Primary Endpoints**:
    *   `GET /`: Serves `templates/index.html` (the primary conversational interface).
    *   `POST /api/chat`: Primary conversational endpoint receiving JSON payloads containing user input, session IDs, and context parameters.
    *   `POST /api/upload`: Handles file uploads for OCR and image analysis.
    *   `POST /api/speech`: Handles raw voice payloads and returns synthesized response audio.
    *   `GET /dashboard`: Renders user analytics and query metrics.

#### `app/modules.py` (Core Business Helpers)
*   **Purpose**: Contains independent domain utilities and text transformation functions.
*   **Key Operations**: Handles string normalization, stop-word filtering, regex extraction of student IDs or course codes, response template substitution, and fuzzy string matching algorithms (Levenshtein Distance) for typo tolerance.

---

### 2. `config/` Directory — Systems, Security & Environment Settings

*   **`settings.py`**: Central configuration hub loading environment variables using `python-dotenv`. Controls environment profiles (`DevelopmentConfig`, `TestingConfig`, `ProductionConfig`), secret key definitions, database URI strings, upload directory limits, and token limits.
*   **`logging_config.py`**: Establishes enterprise logging using Python's `logging.handlers.RotatingFileHandler`. Configures log formatters to record timestamps, log levels (`INFO`, `WARNING`, `ERROR`), route execution times, and unhandled stack traces into `app.log` without exhausting disk space.
*   **`constants.py`**: Defines immutable application constants including max file size limits (e.g., 16 MB), allowed file extensions (`.png`, `.jpg`, `.pdf`), model confidence thresholds (e.g., 0.75 cutoff), and HTTP response status constants.
*   **`jb_config.json` & `app.txt`**: Store environment specific parameters, server deployment notes, and platform runtime settings used during local debugging or deployment.

---

### 3. `data/` & `chatbot_project/` — Domain Knowledge Base

*   **`data/intents.json`**: The primary operational rule book containing structured JSON schemas for pattern matching. Each schema object contains:
    *   `tag`: Unique string identifier for the query category (e.g., `"exam_schedule"`, `"fee_structure"`, `"library_hours"`).
    *   `patterns`: List of common user utterance expressions and training phrases.
    *   `responses`: Array of deterministic canned responses or context templates.
*   **`chatbot_project/data/` & `chatbot_project/modules/`**: Modular space housing supplemental domain datasets, tabular course roadmaps, academic syllabus guides, and specialized helper scripts for processing specialized departmental data.

---

## Deep Learning & Natural Language Processing Engine

### 1. Locally Fine-Tuned DistilBERT Question-Answering Model (`models/distilbert_qa/`)
To achieve fast, deterministic, and accurate answers to specific campus regulations without relying on cloud service latency or incurring API costs, the system incorporates a fine-tuned **DistilBERT** (Distilled BERT) Question-Answering transformer model.

*   **`pytorch_model.bin`**: Binary file storing fine-tuned PyTorch tensor weights trained specifically on SQuAD-formatted campus Q&A pairs.
*   **`config.json`**: Defines model hyperparameters including 6 hidden layers, 12 attention heads, 768 hidden dimension sizes, and 66M parameters (providing 60% faster inference than standard BERT-Base).
*   **`tokenizer.json`**: Vocabulary mappings and WordPiece tokenization rules optimized for technical and academic terms.
*   **`train_distilbert_qa.py`**: The training pipeline execution script. Loads raw training corpora, tokenizes text into start and end position labels for answer spans, calculates Cross-Entropy Loss over target tokens, executes AdamW optimization with linear learning rate decay, and outputs model checkpoints.
*   **`verify_distilbert_qa.py`**: Post-training evaluation utility. Runs benchmark evaluation suites across holdout test queries, calculating Exact Match (EM) and F1-score performance metrics while outputting performance logs into `models/training_report.txt`.

### 2. Generative AI Fallback Engine (`gemini_integration.py`)
For open-ended conversational requests, creative drafting, or queries extending beyond local datasets, the system integrates Google's **Gemini API**.

*   **Architecture**: `gemini_integration.py` wraps the Google Generative AI SDK, encapsulating API key authentication, request timeout handling, and exception recovery.
*   **Prompt Engineering & Guardrails**: Enforces structured system instruction prompts that constrain Gemini's responses to remain helpful, respectful, and focused on student support context while preventing prompt injection or off-topic hallucinations.

---

## Multi-Modal Pipeline Execution Flow

```text
                                ## Multi-Modal Pipeline Execution Flow

### Flowchart Diagram
```text
+-------------------+
|  User Query Input |
+---------+---------+
          |
     [Input Type?]
   /     |      \
(Audio) (Image) (Text)
   v       v       v
Speech.py OCR.py  Sanitization
   \       |       /
    +-------+-------+
            |
     Normalized Text
            v
+-------------------+
| Intent Matcher    |
| (intents.json)    |
+---------+---------+
          |
 [High Confidence?]
   /              \
 (Yes)            (No)
   v               v
+----------------+ +--------------------+
| Instant Response| | DistilBERT QA      |
| (Deterministic) | | Inference Engine   |
+----------------+ +---------+----------+
                             |
                 [Context Found / High F1?]
                   /                     \
                 (Yes)                  (No)
                   v                     v
        +-------------------+   +--------------------+
        | DistilBERT Answer |   | Google Gemini API  |
        +-------------------+   | Generative Fallback|
                                +--------------------+

```

### Detailed Pipeline Stages:

1.  **Ingestion & Normalization**:
    *   Audio input from the Web interface is converted to text via `app/speech.py`.
    *   Image files are parsed into raw text strings via `app/image_processing.py`.
    *   Direct text inputs undergo token sanitization, HTML escaping, and lowercasing in `app/modules.py`.
2.  **Tier 1: Intent Matching Engine (`data/intents.json`)**:
    *   Matches normalized input tokens against pre-compiled regex patterns and intent tags.
    *   If a high-confidence match ($Score \ge 0.85$) is returned, the system immediately returns a canned response, ensuring zero latency ($<10	ext{ms}$).
3.  **Tier 2: DistilBERT Local Context Extraction (`models/distilbert_qa/`)**:
    *   If Tier 1 yields no direct match, the query is passed into the PyTorch inference loop in `train_distilbert_qa.py`/`verify_distilbert_qa.py`.
    *   The model evaluates campus context blocks to compute start and end token logits. If the predicted probability exceeds the threshold ($Score \ge 0.70$), the extracted text span is returned.
4.  **Tier 3: Generative AI Fallback (`gemini_integration.py`)**:
    *   If neither local intent nor local QA models yield high confidence, the request is dispatched to Google Gemini with context guardrails to generate an answer.

---

## Web Frontend & Presentation Layer (`static/` & `templates/`)

### Jinja2 Views (`templates/`)
*   **`index.html`**: Main workspace featuring a responsive chat window, real-time speech recording buttons, file drag-and-drop zones, and message history.
*   **`dashboard.html`**: Provides analytical charts showing recent queries, intent classification distributions, and usage metrics.
*   **`features.html`**: Interactive walkthrough explaining multi-modal features (OCR, Speech, DistilBERT Q&A).
*   **`news.html`**: Dynamic bulletin board pulling institutional announcements and academic updates.
*   **`about.html` & `join.html`**: Information pages for project background, team credits, user registration, and system onboarding.

### Client Logic & Styling (`static/`)
*   **`static/js/filechat.js`**: Asynchronous JavaScript client controlling:
    *   Fetch/AJAX requests to `/api/chat`, `/api/upload`, and `/api/speech`.
    *   MediaRecorder API integration for live browser voice capture.
    *   Dynamic DOM creation for user and bot message bubbles.
    *   Typing indicator animations and smooth scrolling effects.
*   **`static/css/style.css`**: Professional CSS stylesheet implementing CSS custom properties (variables), flexible box layout (flexbox), responsive grid structures, dark/light theme options, and styled UI elements.

---

## Automated Quality Assurance Suite (`tests/`)

The application includes an automated unit and integration test suite implemented via `pytest`:

*   **`tests/test_app.py`**: Validates Flask endpoints. Verifies HTTP status codes (200 OK, 400 Bad Request, 404 Not Found), checks JSON response payloads, and verifies file upload validation rules.
*   **`tests/test_models.py`**: Executes model inference tests. Verifies PyTorch tensor shapes, tokenizer outputs, context truncation behaviors, and inference execution bounds.
*   **`tests/test_preprocess.py`**: Evaluates utility functions in `app/modules.py` and `app/image_processing.py`. Ensures string cleaning routines, regex filters, and image transformation pipelines operate predictably across edge cases.

---

## Step-by-Step Installation & Setup Guide

### System Prerequisites
*   **Operating System**: Linux (Ubuntu 20.04/22.04 LTS recommended), macOS, or Windows 10/11
*   **Python**: Version 3.9, 3.10, or 3.11
*   **Hardware**: Minimum 8 GB RAM (16 GB recommended for model fine-tuning)
*   **Dependencies**: Tesseract OCR engine (for visual processing)

### 1. System Dependencies Installation

#### On Ubuntu / Debian:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv tesseract-ocr ffmpeg
```

#### On macOS (using Homebrew):
```bash
brew update
brew install python tesseract ffmpeg
```

#### On Windows:
1. Download and install Python 3.10 from [python.org](https://www.python.org/).
2. Download and install Tesseract OCR for Windows and add it to your System PATH environment variable.

---

### 2. Project Repository Setup

```bash
# Clone the repository
git clone https://github.com/vaibhavt1857-sudo/student-suppor-chatbot.git
cd student-support-chatbot

# Create virtual environment
python3 -m venv env

# Activate virtual environment
# On Linux / macOS:
source env/bin/activate

# On Windows (PowerShell):
.\env\Scripts\Activate.ps1
```

---

### 3. Install Python Package Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Example `requirements.txt` dependencies included:*
`Flask`, `torch`, `transformers`, `google-generativeai`, `pytesseract`, `opencv-python`, `SpeechRecognition`, `gTTS`, `pytest`, `python-dotenv`.

---

### 4. Configuration & Environment Variables

Create a `.env` file in the root directory:

```env
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=e83a9c71d6f4b2a8e103f5921049b78a
GEMINI_API_KEY=your_actual_google_gemini_api_key
DATABASE_URL=sqlite:///app.db
TESSERACT_PATH=/usr/bin/tesseract
```

---

## Operational Guide & Command Execution

### 1. Running the Flask Web Server
To start the development application server:
```bash
python run.py
```
Access the application interface by navigating to `http://127.0.0.1:5000` in your browser.

### 2. Fine-Tuning the DistilBERT QA Model
To retrain the Transformer Question-Answering model on new datasets:
```bash
python train_distilbert_qa.py
```
This process reads context files from `chatbot_project/data/`, performs gradient updates over PyTorch epochs, and updates the saved model artifacts inside `models/distilbert_qa/`.

### 3. Verifying the DistilBERT QA Model
To evaluate the accuracy and response quality of the trained model:
```bash
python verify_distilbert_qa.py
```
Check `models/training_report.txt` for generated performance summaries and F1 metrics.

### 4. Executing Automated Tests
To run the full unit test suite:
```bash
pytest tests/ -v
```

---

## Future Scope & Planned Enhancements

1.  **Vector Database & RAG Upgrade**: Transition static context searches to a Dense Retrieval-Augmented Generation (RAG) architecture using ChromaDB or FAISS vector databases.
2.  **Role-Based Access Control (RBAC)**: Extend `app/database.py` and `templates/dashboard.html` to support granular permission models for Students, Faculty, and Admin Users.
3.  **LMS & Calendar Integration**: Integrate direct OAuth synchronization with Google Classroom, Canvas, or Moodle APIs for automated schedule reminders.
4.  **Native Mobile Application**: Build React Native / Flutter cross-platform mobile apps consuming the backend REST API endpoints.

---

## Author & Academic Profile

**Vaibhav Kumar Tiwari**
*   **Program**: Bachelor of Technology (B.Tech) in Computer Science and Engineering
*   **Specialization**: Artificial Intelligence & Machine Learning (AI & ML)
*   **Institution**: GNIOT (Affiliated with AKTU)
*   **GitHub**: [@vaibhavt1857-sudo](https://github.com/vaibhavt1857-sudo)
*   **Certifications**: NPTEL Programming in Java (2025)
