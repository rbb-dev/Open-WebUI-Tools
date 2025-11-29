"""
title: ByteDance Seedream 4.0 Image Generation & Editing Pipe
description: ByteDance Seedream 4.0 plugin with task model integration
id: seedream-4-0
author: rbb-dev
author_url: https://github.com/rbb-dev/
version: 0.9.1
features:
  - Image generation and editing using ByteDance Seedream 4.0 via API.
  - Task model analysis to infer intent (generate vs edit), image size, and watermark.
  - For edits, preserves original image dimensions unless a resize is explicitly requested.
  - Natural‑language and dimension parsing with size validation and closest‑size mapping.
  - Supported sizes: 1024x1024, 2048x2048, 2304x1728, 1728x2304, 2560x1440, 1440x2560,
    2496x1664, 1664x2496, 3024x1296, 4096x4096, 6240x2656, plus 1K/2K/4K shorthands.
  - Accepts base64 data URIs and Open WebUI file references; handles multiple images with gateway fallback.
  - Streams OpenAI‑compatible responses and emits rich status updates during processing.
  - Uploads generated assets to the WebUI file store and returns Markdown image links.
  - Configurable via valves (API key, base URL, model, default size, watermark, timeout).
  - Defensive error handling, logging, and approximate 10 MB per‑image size checks.
"""

import base64
import io
import json
import logging
import re
import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Awaitable, Tuple
import httpx
from PIL import Image
from fastapi import Request, UploadFile, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from open_webui.routers.files import upload_file
from open_webui.models.users import UserModel, Users
from open_webui.main import generate_chat_completions
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from starlette.datastructures import Headers

logger = logging.getLogger(__name__)
# Avoid 'No handler could be found' warnings; rely on host/root handlers.
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class Pipe:
    class Valves(BaseModel):
        # Auth and endpoint
        API_KEY: str = Field(default="", description="API key (Bearer)")
        API_BASE_URL: str = Field(default="https://api.cometapi.com/v1", description="API base URL")
        # Logging
        ENABLE_LOGGING: bool = Field(default=False, description="Enable info/debug logs for this plugin. When False, only errors are logged.")
        # Model and basic generation options
        MODEL: str = Field(default="bytedance-seedream-4-0-250828", description="Doubao Seedream 4.0 model id")
        GUIDANCE_SCALE: int = Field(
            default=3,
            description="Default guidance scale (1-20). Larger values stay closer to the prompt per CometAPI docs.",
        )
        # Task model settings
        TASK_MODEL: str = Field(default="gpt-4.1-mini", description="Task model id used when Open WebUI does not supply one.")
        # Defaults when task model output is missing optional fields
        WATERMARK: bool = Field(default=False, description="Default watermark preference when task model omits the field.")
        DEFAULT_SIZE: str = Field(default="2048x2048", description="Default image size when not specified or unsupported size requested.")
        # HTTP client
        REQUEST_TIMEOUT: int = Field(default=600, description="Request timeout in seconds")

    def __init__(self):
        """Configure valves, logging, and cached size metadata on instantiation."""
        self.valves = self.Valves()
        # Apply logging policy on startup
        self._apply_logging_valve()
        # Supported size parameters from documentation
        self.supported_sizes = {
            # Exact pixel dimensions
            "1024x1024",
            "2048x2048",
            "2304x1728",
            "1728x2304",
            "2560x1440",
            "1440x2560",
            "2496x1664",
            "1664x2496",
            "3024x1296",
            "4096x4096",
            "6240x2656",
            # Resolution shorthand
            "1K",
            "2K",
            "4K",
        }
        self._supported_size_lookup = {size.lower(): size for size in self.supported_sizes}

    def _apply_logging_valve(self) -> None:
        """Set logger level based on ENABLE_LOGGING valve.
        OFF  -> ERROR only
        ON   -> INFO and above
        """
        enabled = bool(getattr(self.valves, "ENABLE_LOGGING", False))
        logger.setLevel(logging.INFO if enabled else logging.ERROR)
        # Rely on host/root handlers; do not attach our own to avoid duplicates.
        logger.propagate = True

    async def emit_status(
        self,
        message: str,
        done: bool = False,
        show_in_chat: bool = False,
        emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ):
        """Emit status updates to the client."""
        if emitter:
            await emitter({"type": "status", "data": {"description": message, "done": done}})
        if show_in_chat:
            return f"**✅ {message}**\n\n" if done else f"**⏳ {message}**\n\n"
        return ""

    def _get_image_dimensions(
        self, image_data: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """Extract dimensions from base64 image data."""
        try:
            image_bytes = base64.b64decode(image_data)
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            logger.error(f"Failed to extract image dimensions: {e}")
            return None, None

    @staticmethod
    def _estimate_base64_size_bytes(image_data: str) -> Optional[int]:
        """Approximate decoded byte size of a base64 string without allocating memory."""
        if not image_data:
            return None
        try:
            length = len(image_data.strip())
        except Exception:
            return None
        if not length:
            return None
        return (length * 3) // 4

    @staticmethod
    def _strip_image_placeholders(text: str) -> str:
        """Remove placeholder tokens like [image:2] from prompts before sending to API."""
        if not text:
            return ""
        return re.sub(r"\[image:\d+\]", "", text).strip()

    def _build_reference_image_context(
        self,
        images: List[Dict[str, Any]],
        conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Compile detailed metadata for every provided reference image."""
        if not images:
            return None
        mention_lookup: Dict[int, List[int]] = {}
        if conversation:
            for entry in conversation:
                for idx in entry.get("image_refs", []):
                    mention_lookup.setdefault(idx, []).append(entry.get("message_index", -1))
        details: List[Dict[str, Any]] = []
        for idx, image in enumerate(images):
            width, height = self._get_image_dimensions(image.get("data", ""))
            approx_bytes = self._estimate_base64_size_bytes(image.get("data", ""))
            detail = {
                "index": idx,
                "mime_type": (image.get("mimeType") or "").lower(),
                "origin": image.get("origin"),
                "width": width,
                "height": height,
                "size_label": f"{width}x{height}" if width and height else None,
                "approx_bytes": approx_bytes,
                "approx_megapixels": round((width * height) / 1_000_000, 3)
                if width and height
                else None,
                "mentioned_in": mention_lookup.get(idx, []),
            }
            details.append(detail)
        return {"count": len(images), "details": details}

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        """Extract the first balanced JSON object found inside arbitrary text.

        Returns only the first detected object; assumes the text contains valid JSON
        and does not attempt to parse multiple distinct objects.
        """
        if not text:
            return None
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            start = text.find("{", start + 1)
        return None

    @staticmethod
    def _normalise_model_content(value: Any) -> str:
        """Best-effort conversion of model content fragments to string."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "content" in item:
                        parts.append(str(item.get("content", "")))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if isinstance(value, dict):
            for key in ("text", "content"):
                if key in value and value[key]:
                    return str(value[key])
        return str(value) if value is not None else ""

    def _consume_sse_line(self, raw_line: str, content_parts: List[str]) -> None:
        """Parse a single SSE data line and append its content if valid."""
        line = (raw_line or "").strip()
        if not line or line == "data: [DONE]":
            return
        payload = line[5:].strip() if line.startswith("data:") else line
        if not payload:
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return
        for choice in data.get("choices", []):
            delta = choice.get("delta") or choice.get("message")
            if not delta:
                continue
            content_value = delta.get("content") if isinstance(delta, dict) else delta
            piece = self._normalise_model_content(content_value)
            if piece:
                content_parts.append(piece)

    async def _read_model_response_content(self, response: Any) -> str:
        """Normalise streaming or non-streaming task model responses to text."""
        if hasattr(response, "body_iterator"):
            content_parts: List[str] = []
            buffer = ""
            async for chunk in response.body_iterator:
                if not chunk:
                    continue
                try:
                    chunk_str = chunk.decode("utf-8")
                except Exception:
                    chunk_str = chunk.decode("utf-8", errors="ignore")
                buffer += chunk_str
                while "\n" in buffer:
                    raw_line, buffer = buffer.split("\n", 1)
                    self._consume_sse_line(raw_line, content_parts)
            if buffer:
                self._consume_sse_line(buffer, content_parts)
            return "".join(content_parts)
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                first_choice = choices[0]
                message = first_choice.get("message") or first_choice.get("delta")
                if message:
                    content_value = message.get("content") if isinstance(message, dict) else message
                    return self._normalise_model_content(content_value)
        return str(response or "")

    async def _analyse_prompt_with_task_model(
        self,
        conversation: List[Dict[str, Any]],
        image_context: Optional[Dict[str, Any]],
        raw_user_prompt: str,
        __user__: dict,
        body: dict,
        user_obj: Optional[UserModel],
        __request__: Request,
        emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """Delegate all decision making to the configured task model."""
        supported_sizes = sorted(self.supported_sizes)
        sizes_list = ", ".join(supported_sizes)
        system_prompt = f"""
You are the orchestration task model for ByteDance Seedream 4.0 image generation and editing. The payload you receive has:
- `conversation`: ordered chat history entries (`message_index`, `role`, `text`, `image_refs`). `text` already strips inline image markdown and uses placeholders like `[image:2]` when an attachment is referenced.
- `raw_user_prompt`: the latest user-written text extracted before post-processing (may still contain noise; prefer `conversation` for context).
- `defaults`: fallback values (size, watermark) when the user does not specify them.
- `reference_images.details`: zero or more reference images with `index`, `origin`, `mime_type`, `width`, `height`, `size_label`, `approx_bytes`, and `approx_megapixels`.

Processing checklist BEFORE you answer:
1. Read the full prompt and infer the real goal (new render vs tweak/edit vs variation).
2. Inspect every reference image entry. Note which ones are usable, their sizes, and whether they already match the allowed Seedream sizes.
3. Decide how each image contributes (base, style, mask, ignore) and whether any per-image resizing is needed.
4. Determine the final output size, making sure it is one of the supported sizes ({sizes_list}).
5. Decide watermark, resize behaviour, and craft short explanations for the UI.

Always respond with ONLY JSON (no markdown). First, interpret everything the user asked for (including synonyms such as "remix", "tweak", "draw", "make another", etc.). Then map that understanding onto the canonical fields below. The response MUST match this schema and explain every decision:
{{
  "prompt": string,                        // clean natural-language prompt to send to the API (no placeholders like [image:2])
  "intent": "generate" | "edit",          // is the user asking for a new render or editing references?
  "watermark": boolean,                     // true = add watermark, false = no watermark
  "size": string,                           // one of: {sizes_list}
  "resize_target": {{"width": int, "height": int}} | null,  // desired final width/height when user explicitly asks to resize
  "use_reference_images": boolean,          // true only if at least one image is actively used
  "image_plan": [
    {{
      "index": int,                         // index from reference_images.details
      "action": "use" | "ignore",         // whether the downstream pipeline should include this image
      "role": "base" | "reference" | "style" | "mask", // describe how the image is used
      "target_size": string | null          // per-image resize from {sizes_list} when that image is off-spec
    }}
  ],
  "status_message": string,                // <=120 chars, short sentence describing what will happen (shown to user)
  "notes": string                          // <=120 chars, internal reasoning or highlights (not shown to model output)
}}

- `prompt`: REQUIRED. Write the exact text we should send to the downstream API. Use clear natural language, mention the desired scene/edit explicitly, and NEVER include placeholder tokens like `[image:2]` or other markup—describe the images instead (e.g., “Use the second uploaded photo…”). Keep it concise but complete.
- `intent`: Analyse verbs and nouns first, without assuming specific keywords. Requests that derive a new artwork, variation, or fresh concept should map to `generate`. Requests that modify, fix, repaint, or otherwise transform provided references should map to `edit` (provided at least one usable image exists). When wording is ambiguous, choose the option that best matches the overall goal and explain the decision in `notes`.
- `watermark`: Professional/commercial/portfolio/headshot/logo work → false. Casual, meme, playful, or social media content → true. When unsure, apply the defaults and justify in `notes`.
- `size` (FINAL OUTPUT):
    1. The downstream API accepts ONLY the supported sizes listed above. Every final request must use one of them.
    2. Parse any dimension/ratio/"k" wording from the prompt. If the user specifies something unsupported, map it to the closest allowed size by balancing aspect ratio and pixel count.
    3. When editing, prefer the primary reference image’s size if it already matches a supported option; otherwise map it to the nearest supported size and mention the mapping in `notes`.
    4. Document any trade-offs (e.g., “user asked for 2500×1500 so mapped to 2560×1440”).
- `resize_target` (FINAL OUTPUT DIMENSIONS): Populate only when the user explicitly requests a different final size/resolution. Use integer values that correspond to the chosen `size`. Leave null when editing without resize instructions and the chosen `size` already matches the reference.
- `image_plan` (PER-IMAGE ACTIONS):
    • Provide one entry for every image in `reference_images.details`, even if ignored. Use `conversation[].image_refs` (which mirror placeholders like `[image:3]`) to understand how each image was mentioned.
    • `action`: `use` when the image participates in the edit/generation, otherwise `ignore`.
    • `role` guidelines: `base` = primary image being edited, `reference` = additional visual guidance, `style` = style/lighting reference, `mask` = mask/cutout.
    • `target_size`: REQUIRED whenever that image’s native width/height does not exactly match an allowed size. Choose the closest supported size (same metric as `size`) so the pipeline knows to resize before sending to the API. Omit/leave null only when the image already matches an allowed size or when the image is ignored.
    • If multiple images conflict (very different ratios), select a best compromise: use the most important image (usually the base) to decide the final `size`, then resize the others via their `target_size`. Explain trade-offs in `notes`.
- `use_reference_images`: Set to true if and only if at least one `image_plan` entry has `action":"use"`. Otherwise set to false.
- `status_message`: A concise sentence (<=120 chars) summarising the plan for the UI, e.g., “Editing base image #0, resize to 4K, no watermark”. Mention intent, selected images, and final size when possible.
- `notes`: Mention crucial reasoning—size mappings, ignored images, ambiguity resolutions, safety concerns. Keep it <=120 chars and combine with `status_message` to stay under 200 chars total.

General rules:
- When multiple reference images are supplied, ensure the plan is internally consistent (e.g., only one base image, masks only when user implied masking, etc.).
- Respect safety hints (skip unsupported formats or missing metadata, note it in `notes`).
- Never invent metadata that is not present. Base your decisions solely on the provided prompt, defaults, and reference image details.
- The output MUST be pure JSON with the exact keys described. No code fences, no backticks, no prose outside the JSON.
"""

        has_images = bool((image_context or {}).get("count", 0))

        task_payload = {
            "raw_user_prompt": raw_user_prompt,
            "conversation": conversation,
            "has_reference_images": has_images,
            "reference_images": image_context or {"count": 0, "details": []},
            "supported_sizes": supported_sizes,
            "defaults": {
                "size": self.valves.DEFAULT_SIZE,
                "watermark": self.valves.WATERMARK,
            },
        }

        try:
            task_model = self._resolve_task_model(body, user_obj, __request__)
            await self.emit_status("Analysing prompt...", emitter=emitter)
            resolved_user = user_obj or await self._get_user_by_id(__user__["id"])
            if not resolved_user:
                logger.error("Unable to resolve user for task model call")
                raise RuntimeError("user_lookup_failed")

            form_data = {
                "model": task_model,
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": json.dumps(task_payload, ensure_ascii=True)},
                ],
                "max_tokens": 200,
                "temperature": 0.1,
                "stream": False,
            }
            logger.info("Task model request payload: %s", json.dumps(form_data, ensure_ascii=False))

            response = await generate_chat_completions(
                form_data=form_data,
                user=resolved_user,
                request=__request__,
            )
            content = await self._read_model_response_content(response)
            logger.info("Task model raw response: %s", content)
            if not content:
                raise RuntimeError("task_model_empty_response")

            json_blob = self._extract_first_json_object(content)
            if not json_blob:
                logger.debug(f"Task model raw response: {content[:1000]}")
                raise RuntimeError("task_model_missing_json")

            params = json.loads(json_blob)
            if not isinstance(params, dict):
                raise RuntimeError("task_model_invalid_schema")

            image_count = (image_context or {}).get("count", 0)
            validated_params = self._validate_task_model_params(
                params,
                image_count,
                fallback_prompt=raw_user_prompt or "",
            )
            logger.info(f"Task model determined parameters: {validated_params}")
            return validated_params
        except Exception as exc:
            logger.error(f"Task model call failed: {exc}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise

    def _validate_size_choice(self, value: Any) -> str:
        """Ensure the supplied size string matches supported entries."""
        normalized = str(value or "").replace("×", "x").strip().lower()
        if not normalized:
            raise ValueError("Empty size value from task model")
        if normalized not in self._supported_size_lookup:
            raise ValueError(f"Unsupported size requested by task model: {value}")
        return self._supported_size_lookup[normalized]

    def _coerce_guidance_scale(self, value: Any) -> int:
        """Parse guidance scale from request/body and clamp to sane defaults."""
        default = getattr(self.valves, "GUIDANCE_SCALE", 3)
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(1, min(parsed, 20))

    @staticmethod
    def _coerce_image_count(value: Any) -> int:
        """Parse desired output count (`n`) and keep it within API limits."""
        if value is None:
            return 1
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 1
        # CometAPI constraint: references + generated images <= 15.
        return max(1, min(parsed, 10))

    def _validate_task_model_params(
        self,
        params: Dict[str, Any],
        image_count: int,
        fallback_prompt: str,
    ) -> Dict[str, Any]:
        """Validate and sanitise parameters coming back from the task model."""
        prompt_value = params.get("prompt")
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            prompt_text = self._strip_image_placeholders(fallback_prompt)
            logger.warning("Task model response missing 'prompt'; falling back to user prompt")
        else:
            prompt_text = self._strip_image_placeholders(prompt_value)

        intent_raw = str(params.get("intent", "generate")).lower()
        intent = intent_raw if intent_raw in {"generate", "edit"} else "generate"

        size = self._validate_size_choice(params.get("size", self.valves.DEFAULT_SIZE))

        watermark = bool(params.get("watermark", self.valves.WATERMARK))
        use_reference_images = bool(params.get("use_reference_images", False))
        if image_count <= 0:
            use_reference_images = False

        resize_target = params.get("resize_target")
        validated_resize: Optional[Dict[str, int]] = None
        if isinstance(resize_target, dict):
            width = resize_target.get("width")
            height = resize_target.get("height")
            if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
                validated_resize = {"width": width, "height": height}

        notes = params.get("notes")
        notes_str = notes.strip() if isinstance(notes, str) else ""

        image_plan_entries: List[Dict[str, Any]] = []
        raw_plan = params.get("image_plan") or []
        if isinstance(raw_plan, list) and image_count > 0:
            for entry in raw_plan:
                if not isinstance(entry, dict):
                    continue
                index = entry.get("index")
                if not isinstance(index, int) or index < 0 or index >= image_count:
                    continue
                action = str(entry.get("action", "ignore")).lower()
                if action not in {"use", "ignore"}:
                    action = "ignore"
                role = str(entry.get("role", "reference")).lower()
                if role not in {"base", "reference", "style", "mask"}:
                    role = "reference"
                target_size = entry.get("target_size")
                validated_target_size = None
                if target_size:
                    try:
                        validated_target_size = self._validate_size_choice(target_size)
                    except ValueError:
                        validated_target_size = None
                image_plan_entries.append(
                    {
                        "index": index,
                        "action": action,
                        "role": role,
                        "target_size": validated_target_size,
                    }
                )

        use_reference_images = use_reference_images or any(
            entry.get("action") == "use" for entry in image_plan_entries
        )

        if intent == "edit" and not use_reference_images:
            logger.info("Task model intent was 'edit' without usable reference images; switching to 'generate'.")
            intent = "generate"

        status_message = params.get("status_message")
        status_message = status_message.strip() if isinstance(status_message, str) else ""

        return {
            "prompt": prompt_text,
            "intent": intent,
            "watermark": watermark,
            "size": size,
            "use_reference_images": use_reference_images,
            "resize_target": validated_resize,
            "notes": notes_str,
            "image_plan": image_plan_entries,
            "status_message": status_message,
        }

    def _resolve_task_model(
        self,
        body: Dict[str, Any],
        user_obj: Optional[UserModel],
        __request__: Request,
    ) -> str:
        """Resolve which task model to call using Open WebUI preferences."""

        def _extract(source: Any) -> Optional[str]:
            """Normalise candidate objects and pull out the first task-model identifier."""
            mapping = self._object_to_mapping(source)
            for key in (
                "task_model",
                "task_model_id",
                "task_model_name",
                "taskModel",
            ):
                value = mapping.get(key)
                if isinstance(value, str):
                    value = value.strip()
                    if value:
                        return value
            return None

        for candidate_source in (
            body,
            body.get("metadata") if isinstance(body, dict) else None,
        ):
            candidate = _extract(candidate_source)
            if candidate:
                return candidate

        if user_obj:
            for candidate_source in (user_obj, getattr(user_obj, "settings", None)):
                candidate = _extract(candidate_source)
                if candidate:
                    return candidate

        app = getattr(__request__, "app", None)
        state = getattr(app, "state", None) if app else None
        settings = getattr(state, "settings", None) if state else None
        candidate = _extract(settings)
        if candidate:
            return candidate

        return self.valves.TASK_MODEL

    @staticmethod
    def _object_to_mapping(source: Any) -> Dict[str, Any]:
        """Convert arbitrary objects (dicts, Pydantic models, namespaces) into a mapping."""
        if not source:
            return {}
        if isinstance(source, dict):
            return source
        for attr in ("model_dump", "dict"):
            method = getattr(source, attr, None)
            if callable(method):
                try:
                    return method()  # type: ignore[return-value]
                except Exception:
                    continue
        return getattr(source, "__dict__", {}) or {}

    async def _get_user_by_id(self, user_id: str) -> Optional[UserModel]:
        """Fetch a user record without blocking the async loop."""
        try:
            return await run_in_threadpool(Users.get_user_by_id, user_id)
        except Exception as exc:
            logger.error(f"Failed to load user {user_id}: {exc}")
            return None

    async def _get_file_by_id(self, file_id: str):
        """Look up file metadata via the ORM in a threadpool."""
        try:
            from open_webui.models.files import Files

            return await run_in_threadpool(Files.get_file_by_id, file_id)
        except Exception as exc:
            logger.error(f"Failed to load file {file_id}: {exc}")
            return None

    async def _read_file_bytes(self, path: str) -> bytes:
        """Read file contents without blocking the event loop."""

        def _read() -> bytes:
            """Blocking helper invoked inside run_in_threadpool."""
            with open(path, "rb") as f:
                return f.read()

        return await run_in_threadpool(_read)

    async def pipes(self) -> List[dict]:
        """Return the manifest entry consumed by Open WebUI."""
        return [{"id": "seedream-4-0", "name": "ByteDance: Seedream 4.0"}]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> StreamingResponse:
        """Main entrypoint invoked by Open WebUI for each generation/edit request."""
        # Re-apply in case valves changed at runtime
        self._apply_logging_valve()
        user = await self._get_user_by_id(__user__["id"])

        async def stream_response():
            """Yield OpenAI-compatible response chunks (streaming or single payload)."""
            try:
                model = self.valves.MODEL
                messages = body.get("messages", [])
                is_stream = bool(body.get("stream", False))
                if not user:
                    yield self._format_data(
                        is_stream=is_stream,
                        content="Error: Unable to load user context.",
                        finish_reason="stop",
                    )
                    if is_stream:
                        yield "data: [DONE]\n\n"
                    return
                # Extract structured conversation and images
                conversation_log, images, last_user_text = await self._collect_conversation_and_images(messages)
                if not self.valves.API_KEY:
                    yield self._format_data(
                        is_stream=is_stream,
                        content="Error: API_KEY not set in valves.",
                        finish_reason="stop",
                    )
                    if is_stream:
                        yield "data: [DONE]\n\n"
                    return
                # Build reference metadata for task model
                image_context: Optional[Dict[str, Any]] = None
                if images:
                    await self.emit_status("Analyzing reference image metadata...", emitter=__event_emitter__)
                    image_context = self._build_reference_image_context(images, conversation_log)
                    if image_context:
                        summary = ", ".join(
                            f"#{detail['index']}:{detail.get('size_label') or 'unknown'}"
                            for detail in image_context.get("details", [])
                        )
                        logger.info(
                            "Reference images detected (%s total): %s",
                            image_context.get("count", 0),
                            summary,
                        )
                # Get dynamic parameters from task model
                dynamic_params = await self._analyse_prompt_with_task_model(
                    conversation_log,
                    image_context,
                    last_user_text,
                    __user__,
                    body,
                    user,
                    __request__,
                    emitter=__event_emitter__,
                )
                resolved_prompt = dynamic_params.get("prompt") or last_user_text or ""
                resolved_prompt = self._strip_image_placeholders(resolved_prompt)
                if not resolved_prompt:
                    raise RuntimeError("Task model did not return a prompt")
                # Build request JSON using dynamic parameters
                endpoint = "/images/generations"
                json_data: Dict[str, Any] = {
                    "model": model,
                    "prompt": resolved_prompt,
                    "response_format": "b64_json",
                    "watermark": dynamic_params["watermark"],
                    "size": dynamic_params["size"],
                    "guidance_scale": self._coerce_guidance_scale(body.get("guidance_scale")),
                    "n": self._coerce_image_count(body.get("n") or body.get("num_images")),
                }
                task_notes = dynamic_params.get("notes") or ""
                if task_notes:
                    logger.info(f"Task model notes: {task_notes}")

                image_plan = dynamic_params.get("image_plan", [])
                selected_images: List[Dict[str, str]] = []
                selected_indices: List[int] = []
                if image_plan:
                    logger.info("Task model image plan: %s", image_plan)
                    for directive in image_plan:
                        if directive.get("action") != "use":
                            continue
                        idx = directive.get("index")
                        if isinstance(idx, int) and 0 <= idx < len(images):
                            selected_indices.append(idx)
                            selected_images.append(images[idx])
                elif dynamic_params.get("use_reference_images"):
                    selected_indices = list(range(len(images)))
                    selected_images = images[:]

                if dynamic_params["intent"] == "edit" and not selected_images:
                    logger.info(
                        "Task model intent 'edit' but no usable reference images selected; falling back to generation mode."
                    )

                is_edit_request = dynamic_params["intent"] == "edit" and bool(selected_images)

                # Prepare images for editing
                image_uris: List[str] = []
                if is_edit_request:
                    await self.emit_status("Preparing image editing...", emitter=__event_emitter__)
                    for im in selected_images:
                        mime = (im.get("mimeType") or "").lower().strip()
                        if mime not in ("image/png", "image/jpeg", "image/jpg"):
                            continue
                        if mime == "image/jpg":
                            mime = "image/jpeg"
                        b64 = im.get("data", "")
                        if not b64:
                            continue
                        approx_size = self._estimate_base64_size_bytes(b64)
                        if approx_size and approx_size > 10 * 1024 * 1024:
                            logger.info("Skipping image >10MB after decode estimate")
                            continue
                        image_uris.append(f"data:{mime};base64,{b64}")
                    if image_uris:
                        if len(image_uris) == 1:
                            json_data["image"] = image_uris[0]
                        else:
                            json_data["image"] = image_uris[:10]

                resize_target = dynamic_params.get("resize_target")
                resize_info = ""
                if resize_target:
                    resize_info = f", resize_to: {resize_target.get('width')}x{resize_target.get('height')}"
                params_info = (
                    f"(intent: {dynamic_params['intent']}, size: {dynamic_params['size']}, "
                    f"images used: {len(selected_images)}/{len(images)}, watermark: {dynamic_params['watermark']}{resize_info})"
                )
                status_message = dynamic_params.get("status_message")
                if status_message:
                    status_message = status_message.strip()
                if not status_message:
                    status_message = (
                        f"Editing image {params_info}..." if is_edit_request else f"Generating image {params_info}..."
                    )
                await self.emit_status(status_message, emitter=__event_emitter__)
                # Make API request
                try:
                    async with httpx.AsyncClient(
                        base_url=self.valves.API_BASE_URL,
                        headers={
                            "Authorization": f"Bearer {self.valves.API_KEY}",
                            "Content-Type": "application/json",
                        },
                        timeout=self.valves.REQUEST_TIMEOUT,
                    ) as client:
                        redacted_payload = {
                            "model": json_data.get("model"),
                            "size": json_data.get("size"),
                            "watermark": json_data.get("watermark"),
                            "prompt": json_data.get("prompt"),
                            "guidance_scale": json_data.get("guidance_scale"),
                            "n": json_data.get("n"),
                        }
                        if "image" in json_data:
                            redacted_payload["image_count"] = (
                                len(json_data["image"])
                                if isinstance(json_data["image"], list)
                                else 1
                            )
                            redacted_payload["image"] = "<omitted base64>"
                        logger.info(
                            "Request payload summary: prompt=%s | size=%s | watermark=%s | images=%s",
                            redacted_payload.get("prompt"),
                            redacted_payload.get("size"),
                            redacted_payload.get("watermark"),
                            redacted_payload.get("image_count", 0),
                        )
                        logger.info("Request payload detail: %s", redacted_payload)
                        response = await client.post(endpoint, json=json_data)
                        # Handle array fallback for older gateways
                        if response.status_code == 400 and isinstance(json_data.get("image"), list):
                            try:
                                msg = response.text
                                logger.debug(f"API Error Response: {json.dumps(response.json(), indent=2)}")
                            except Exception:
                                msg = ""
                            if ("image of type string" in msg or "unmarshal array into" in msg):
                                logger.error("API gateway rejected image array, retrying with first image string.")
                                retry_payload = dict(json_data)
                                retry_payload["image"] = json_data["image"][0]
                                response = await client.post(endpoint, json=retry_payload)
                        response.raise_for_status()
                except httpx.HTTPError as e:
                    logger.error(f"API request failed: {str(e)}")
                    error_status = await self.emit_status("An error occurred while calling API", True, True, emitter=__event_emitter__)
                    yield self._format_data(
                        is_stream=is_stream,
                        content=f"{error_status}Error from API: {str(e)}",
                        finish_reason="stop",
                    )
                    if is_stream:
                        yield "data: [DONE]\n\n"
                    return
                response_data = response.json()
                # Process images
                await self.emit_status("Image processing complete!", True, emitter=__event_emitter__)
                image_markdown = []
                for i, item in enumerate(response_data.get("data", []), start=1):
                    if "b64_json" in item:
                        try:
                            item_mime = (item.get("mime_type") or item.get("mimeType") or "image/jpeg").lower()
                            if item_mime == "image/jpg":
                                item_mime = "image/jpeg"
                            image_url = await self._upload_image(
                                __request__=__request__,
                                user=user,
                                image_data=item["b64_json"],
                                mime_type=item_mime,
                            )
                            image_markdown.append(f"![image_{i}]({image_url})")
                        except Exception as e:
                            logger.error(f"Image upload failed: {str(e)}")
                            error_status = await self.emit_status("An error occurred while uploading image", True, True, emitter=__event_emitter__)
                            yield self._format_data(
                                is_stream=is_stream,
                                content=f"{error_status}Error uploading image: {str(e)}",
                                finish_reason="stop",
                            )
                            if is_stream:
                                yield "data: [DONE]\n\n"
                            return
                    elif "url" in item:
                        image_markdown.append(f"![image_{i}]({item['url']})")
                content = "\n\n".join(image_markdown) if image_markdown else "No images returned."
                # Return response
                if is_stream:
                    yield self._format_data(
                        is_stream=True,
                        model=model,
                        content=content,
                        finish_reason=None,
                    )
                    yield self._format_data(
                        is_stream=True,
                        model=model,
                        content=None,
                        usage=response_data.get("usage"),
                        finish_reason="stop",
                    )
                    yield "data: [DONE]\n\n"
                else:
                    yield self._format_data(
                        is_stream=False,
                        model=model,
                        content=content,
                        usage=response_data.get("usage"),
                        finish_reason="stop",
                    )
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                error_status = await self.emit_status("An error occurred while processing request", True, True, emitter=__event_emitter__)
                yield self._format_data(
                    is_stream=body.get("stream", False),
                    content=f"{error_status}Error processing request: {str(e)}",
                    finish_reason="stop",
                )
                if body.get("stream", False):
                    yield "data: [DONE]\n\n"

        media_type = "text/event-stream" if body.get("stream", False) else "application/json"
        return StreamingResponse(stream_response(), media_type=media_type)

    async def _collect_conversation_and_images(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, str]], str]:
        """Capture the conversation text plus every referenced image."""

        conversation: List[Dict[str, Any]] = []
        images: List[Dict[str, str]] = []
        last_user_text = ""

        data_uri_pattern = re.compile(r"!\[[^\]]*\]\(data:([^;]+);base64,([^)]+)\)")
        file_pattern = re.compile(r"!\[[^\]]*\]\((/api/v1/files/[^)]+|/files/[^)]+)\)")
        remote_pattern = re.compile(r"!\[[^\]]*\]\((https?://[^)]+)\)")

        def _append_image_record(mime_type: str, data: str, origin: Dict[str, Any]) -> int:
            idx = len(images)
            images.append({"mimeType": mime_type, "data": data, "origin": origin})
            return idx

        async def _append_file_image(file_id: str) -> Optional[int]:
            file_item = await self._get_file_by_id(file_id)
            if file_item and file_item.path:
                try:
                    file_data = await self._read_file_bytes(file_item.path)
                except Exception as exc:
                    logger.error(f"Failed to read file {file_id} from disk: {exc}")
                    return None
                data = base64.b64encode(file_data).decode("utf-8")
                meta = (file_item.meta or {})
                mime_type = meta.get("content_type", "image/png")
                return _append_image_record(mime_type, data, {"type": "file", "id": file_id})
            logger.error(f"Failed to fetch file {file_id}: not found")
            return None

        async def _append_remote_image(url: str) -> Optional[int]:
            remote_image = await self._fetch_remote_image(url)
            if not remote_image:
                return None
            idx = len(images)
            images.append(remote_image)
            return idx

        async def _ingest_url_source(url: str) -> Optional[int]:
            if not url:
                return None
            url = url.strip()
            if url.startswith("data:"):
                parts = url.split(";base64,", 1)
                if len(parts) == 2:
                    mime_type = parts[0].replace("data:", "", 1).lower()
                    data = parts[1]
                    return _append_image_record(mime_type, data, {"type": "data_uri"})
                return None
            if "/api/v1/files/" in url or "/files/" in url:
                file_id = (
                    url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0]
                    if "/api/v1/files/" in url
                    else url.split("/files/")[-1].split("/")[0].split("?")[0]
                )
                return await _append_file_image(file_id)
            return await _append_remote_image(url)

        for message_index, message in enumerate(messages):
            role = message.get("role", "user")
            content = message.get("content", "")
            text_segments: List[str] = []
            image_refs: List[int] = []

            def _register_placeholder(idx: Optional[int]) -> str:
                if idx is None:
                    return ""
                image_refs.append(idx)
                return f"[image:{idx}]"

            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        text_segments.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        idx = await _ingest_url_source(item.get("image_url", {}).get("url", ""))
                        placeholder = _register_placeholder(idx)
                        if placeholder:
                            text_segments.append(placeholder)
            elif isinstance(content, str):
                text_value = content

                while True:
                    match = data_uri_pattern.search(text_value)
                    if not match:
                        break
                    mime_type = match.group(1).lower()
                    data = match.group(2)
                    idx = _append_image_record(mime_type, data, {"type": "data_uri"})
                    placeholder = _register_placeholder(idx)
                    text_value = text_value[: match.start()] + placeholder + text_value[match.end():]

                while True:
                    match = file_pattern.search(text_value)
                    if not match:
                        break
                    file_url = match.group(1)
                    file_id = (
                        file_url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0]
                        if "/api/v1/files/" in file_url
                        else file_url.split("/files/")[-1].split("/")[0].split("?")[0]
                    )
                    idx = await _append_file_image(file_id)
                    placeholder = _register_placeholder(idx)
                    text_value = text_value[: match.start()] + placeholder + text_value[match.end():]

                while True:
                    match = remote_pattern.search(text_value)
                    if not match:
                        break
                    remote_url = match.group(1)
                    idx = await _append_remote_image(remote_url)
                    placeholder = _register_placeholder(idx)
                    replacement = placeholder or ""
                    text_value = text_value[: match.start()] + replacement + text_value[match.end():]

                text_segments.append(text_value.strip())
            else:
                text_segments.append("")

            cleaned_text = " ".join(segment for segment in text_segments if segment).strip()
            conversation.append(
                {
                    "message_index": message_index,
                    "role": role,
                    "text": cleaned_text,
                    "image_refs": image_refs,
                }
            )
            if role == "user" and cleaned_text:
                last_user_text = cleaned_text

        return conversation, images, last_user_text

    async def _fetch_remote_image(self, url: str) -> Optional[Dict[str, str]]:
        """Download remote image URLs when provided by the client."""
        url = (url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return None
        try:
            async with httpx.AsyncClient(timeout=min(self.valves.REQUEST_TIMEOUT, 60)) as client:
                response = await client.get(url)
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch remote image {url}: {e}")
            return None
        mime_type = response.headers.get("content-type", "").split(";")[0].lower()
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        if mime_type not in {"image/png", "image/jpeg"}:
            logger.error(f"Unsupported remote image type '{mime_type}' from {url}. Skipping.")
            return None
        if len(response.content) > 10 * 1024 * 1024:
            logger.error(f"Remote image {url} exceeds 10MB decoded size. Skipping.")
            return None
        data = base64.b64encode(response.content).decode("utf-8")
        return {"mimeType": mime_type, "data": data, "origin": {"type": "url", "value": url}}

    async def _upload_image(
        self, __request__: Request, user: UserModel, image_data: str, mime_type: str
    ) -> str:
        """Upload generated image bytes to the WebUI file store and return its URL."""
        try:
            file_item = await run_in_threadpool(
                upload_file,
                request=__request__,
                background_tasks=BackgroundTasks(),
                file=UploadFile(
                    file=io.BytesIO(base64.b64decode(image_data)),
                    filename=f"generated-image-{uuid.uuid4().hex}.jpg",
                    headers=Headers({"content-type": mime_type}),
                ),
                process=False,
                user=user,
                metadata={"mime_type": mime_type},
            )
            image_url = __request__.app.url_path_for("get_file_content_by_id", id=file_item.id)
            return image_url
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            raise

    def _format_data(
        self,
        is_stream: bool,
        model: str = "",
        content: Optional[str] = "",
        usage: Optional[dict] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        """Format the response data in the expected OpenAI-compatible format."""
        data = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk" if is_stream else "chat.completion",
            "created": int(time.time()),
            "model": model,
        }
        if is_stream:
            is_stop_chunk = finish_reason == "stop" and content is None
            delta: Dict[str, Any] = {}
            if not is_stop_chunk:
                delta["role"] = "assistant"
                if content is not None:
                    delta["content"] = content
            data["choices"] = [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "delta": delta,
                }
            ]
        else:
            message_content = content or ""
            data["choices"] = [
                {
                    "finish_reason": finish_reason or "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message_content,
                    },
                }
            ]
        if usage:
            data["usage"] = usage
        return f"data: {json.dumps(data)}\n\n" if is_stream else json.dumps(data)
