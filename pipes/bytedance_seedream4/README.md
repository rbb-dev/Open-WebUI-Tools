# ByteDance Seedream 4.0 Pipe

Open WebUI pipe for image generation and editing using ByteDance Seedream 4.0 via API, with a lightweight task model to infer intent, watermark, and size.

## ID and name
- Pipe ID: `seedream-4-0`
- Display name: `ByteDance: Seedream 4.0`
- File: `bytedance_seedream4_pipe.py`

## Features
- Image generation and editing (preserves original dimensions on edits unless resize is requested).
- Task‑model analysis to infer intent (generate vs edit), watermark, and size.
- Size extraction from natural language or explicit dimensions, with validation and closest‑size mapping.
- Accepts base64 data URIs and Open WebUI file references, with multi‑image gateway fallback.
- Streams OpenAI‑compatible responses and emits rich status updates.
- Uploads generated assets to the WebUI file store and returns Markdown image links.
- Configurable valves and defensive error handling.

Supported sizes include: 1024x1024, 2048x2048, 2304x1728, 1728x2304, 2560x1440, 1440x2560, 2496x1664, 1664x2496, 3024x1296, 4096x4096, 6240x2656, plus 1K/2K/4K shorthands.

## Installation
1. Place this folder at `pipes/bytedance_seedream4/` inside your Open WebUI custom pipes location (for Docker, mount or copy it into the backend’s pipes directory).
2. Ensure Python dependencies used by Open WebUI cover `httpx`, `Pillow`, `pydantic`, `fastapi`, and `starlette` (these are already part of Open WebUI’s backend).
3. Restart Open WebUI.

## Configuration (valves)
Set valves via the Open WebUI plugin UI or by editing defaults in the file.
- `API_KEY` (string, required): API bearer token for your image backend.
- `API_BASE_URL` (string, default: `https://api.cometapi.com/v1`).
- `ENABLE_LOGGING` (bool, default: `False`): when `True`, logs at INFO level; otherwise, only errors.
- `MODEL` (string, default: `bytedance-seedream-4-0-250828`).
- `GUIDANCE_SCALE` (int, default: `3`): guidance strength passed to CometAPI /images/generations (1–20, higher values hug the prompt more closely).
- `TASK_MODEL_ENABLED` (bool, default: `True`): enable internal prompt analysis to set intent/size/watermark.
- `WATERMARK` (bool, default: `True`): fallback watermark when task model is disabled or fails.
- `DEFAULT_SIZE` (string, default: `2048x2048`).
- `REQUEST_TIMEOUT` (int seconds, default: `600`).

Notes:
- The task model uses Open WebUI’s internal `generate_chat_completions` call (model `gpt-4.1-mini` by default). Ensure your Open WebUI is configured to serve this model or equivalent; otherwise disable `TASK_MODEL_ENABLED` and the pipe will fall back to heuristics.
- For edits, the original image size is preserved unless the prompt explicitly requests a resize (e.g. “resize to 1024x1024”, “make it smaller”).

## Usage
- In the UI, select the tool named `ByteDance: Seedream 4.0` (ID `seedream-4-0`).
- Provide a text prompt. Optionally include an image as a base64 data URI or an Open WebUI file reference to perform an edit.
- The pipe returns Markdown image links to the generated results.

## Troubleshooting
- `Error: API_KEY not set in valves.`: provide a valid API key.
- Gateway rejects image array: the pipe automatically retries with a single image.
- Oversized input images: images estimated over ~10 MB after decode are skipped.

## Credits
Authored by @rbb-dev. Contributions welcome.
