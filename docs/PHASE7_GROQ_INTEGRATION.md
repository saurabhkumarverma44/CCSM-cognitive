# CCSM Phase 7 — GroqCloud scene reasoning

## What this phase adds

Phase 7 adds a question-answering adapter around the Phase 6 `ContextBuilder`. It does **not** replace YOLO12l, ByteTrack, identity recovery, `MemoryManager`, or trajectory compression.

Flow:

`Camera -> YOLO12l/ByteTrack -> MemoryManager -> ContextBuilder JSON -> GroqCloud -> Answer`

Press **`g`** in the CCSM window to ask a question about the current memory snapshot. Press Enter on a blank question to cancel. Press **`c`** to print the local context without making an API call. Groq is called only after an explicit `g` keypress; the application does not send every frame to the LLM.

## 1. Install dependencies (Windows PowerShell)

Activate the existing environment from the project root:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell reports that script execution is disabled, you can use the interpreter directly instead of activating:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Do not reinstall PyTorch if your current CUDA/PyTorch installation is working. The added requirements are the Groq SDK and python-dotenv; the rest of the requirements file remains in place.

## 2. Configure GroqCloud key securely

1. Create an API key in your GroqCloud account.
2. In the project root, create `.env` from the example:

```powershell
Copy-Item .env.example .env
notepad .env
```

3. Replace `replace_with_your_groq_api_key` with your key. Do not include quotes unless your key itself requires them. Never commit `.env` or paste the key into source code/chat.
4. Optional model and request settings are already documented in `.env.example`.

`.env` is ignored by Git. `.env.example` contains placeholders only and can be committed.

## 3. Run the tests

```powershell
python -m pytest tests/ -v
```

The LLM unit tests use a fake client and do not need a real API key or internet access. Existing vision tests need the project's existing dependencies installed (including Ultralytics and OpenCV).

## 4. Start CCSM

```powershell
python main.py
```

Controls:

- `q`: quit
- `m`: toggle HUD
- `d`: toggle debug logging
- `c`: print current structured semantic context locally
- `g`: ask GroqCloud about the current memory snapshot
- `p`: memory compression statistics and manual purge

Example questions:

- `How many objects are currently visible?`
- `Which tracked objects are occluded, and what are their memory IDs?`
- `What movement direction is recorded for object ID 2?`
- `Can the current memory determine what happened while an object was missing?`

The model receives structured object memory, not raw camera frames. It cannot recognize details absent from that memory. IDs are tracking IDs, not real-world identities. Scene coordinates/velocity are pixel-based. The system prompt instructs the model to state uncertainty instead of inventing missing evidence.

## 5. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | required | Secret API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model ID; change if unavailable in your account |
| `GROQ_TIMEOUT_SECONDS` | `30` | Request timeout (1–120 seconds) |
| `GROQ_MAX_TOKENS` | `500` | Maximum generated response tokens (64–4096) |
| `GROQ_TEMPERATURE` | `0.2` | Response randomness (0–2) |
| `CCSM_LLM_MAX_CONTEXT_CHARS` | `18000` | Upper bound for context size (1000–100000) |

## Design and safety notes

- The client is initialized lazily when `g` is pressed. CCSM can start without an API key; LLM use reports a configuration message if the key is missing.
- The adapter sends one snapshot and one question per explicit user request. It does not maintain chat history or persist LLM conversations.
- Request exceptions are surfaced as sanitized error messages; request content and API key are not written to logs.
- The scene context is treated as untrusted observation data, not instructions. This is a prompt-level safeguard, not a guarantee against all model errors.
- API availability, quotas, model names, and prices are controlled by GroqCloud and can change. Check your GroqCloud console if the provider rejects a model or request.
