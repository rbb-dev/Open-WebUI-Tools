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
        # Task model settings
        TASK_MODEL_ENABLED: bool = Field(default=True, description="Enable task model for dynamic parameter detection")
        # Fallback defaults when task model is disabled/unavailable
        WATERMARK: bool = Field(default=False, description="Fallback: Add watermark when task model fails or is disabled.")
        DEFAULT_SIZE: str = Field(default="2048x2048", description="Default image size when not specified or unsupported size requested.")
        # HTTP client
        REQUEST_TIMEOUT: int = Field(default=600, description="Request timeout in seconds")
        TASK_MODEL_FALLBACK: str = Field(default="gpt-4.1-mini", description="Fallback task model id when Open WebUI does not supply one.")

    def __init__(self):
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
        self._supported_dimension_pairs = [
            (1024, 1024),
            (2048, 2048),
            (2304, 1728),
            (1728, 2304),
            (2560, 1440),
            (1440, 2560),
            (2496, 1664),
            (1664, 2496),
            (3024, 1296),
            (4096, 4096),
            (6240, 2656),
        ]

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

    def _dimensions_to_size_param(self, width: int, height: int) -> str:
        """Convert image dimensions to supported size parameter."""
        size_str = f"{width}x{height}"
        # Check if exact match exists
        if size_str in self.supported_sizes:
            return size_str
        # Find closest supported size
        min_diff = float("inf")
        closest_size = self.valves.DEFAULT_SIZE
        for sup_w, sup_h in self._supported_dimension_pairs:
            # Calculate difference based on total pixel count and aspect ratio similarity
            pixel_diff = abs((width * height) - (sup_w * sup_h))
            aspect_diff = abs((width / height) - (sup_w / sup_h)) * 1000  # Weight aspect ratio
            total_diff = pixel_diff + aspect_diff
            if total_diff < min_diff:
                min_diff = total_diff
                closest_size = f"{sup_w}x{sup_h}"
        logger.info(f"Mapped {width}x{height} to closest supported size: {closest_size}")
        return closest_size

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

    def _extract_size_from_prompt(self, prompt: str) -> str:
        """Extract size information from natural language input."""
        prompt_lower = prompt.lower()
        # Check for exact dimension patterns
        dimension_patterns = [
            r"(\d{3,4})\s*[x×]\s*(\d{3,4})",  # 1024x1024, 2048×2048
            r"(\d{3,4})\s*by\s*(\d{3,4})",  # 1024 by 1024
            r"(\d{3,4})\s*×\s*(\d{3,4})",  # 1024 × 1024
        ]
        for pattern in dimension_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
                size_str = f"{width}x{height}"
                if size_str in self.supported_sizes:
                    return size_str
        # Check for resolution shorthand
        if "4k" in prompt_lower or "4096" in prompt_lower:
            return "4K"
        elif "2k" in prompt_lower or "2048" in prompt_lower:
            return "2K"
        elif "1k" in prompt_lower or "1024" in prompt_lower:
            return "1K"
        # Check for aspect ratio descriptions
        aspect_ratio_map = {
            ("square", "1:1"): "2048x2048",
            ("4:3", "4 by 3"): "2304x1728",
            ("3:4", "3 by 4", "portrait"): "1728x2304",
            ("16:9", "widescreen", "landscape"): "2560x1440",
            ("9:16", "vertical", "phone screen"): "1440x2560",
            ("3:2",): "2496x1664",
            ("2:3",): "1664x2496",
            ("21:9", "ultrawide"): "3024x1296",
        }
        for keywords, size in aspect_ratio_map.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return size
        # Default size
        return self.valves.DEFAULT_SIZE

    def _validate_and_get_size(self, requested_size: str) -> str:
        """Validate requested size and return supported size or default."""
        candidate = (requested_size or "").strip()
        if not candidate:
            return self.valves.DEFAULT_SIZE
        dimension_normalized = candidate.replace("×", "x")
        lookup_key = dimension_normalized.lower()
        if lookup_key in self._supported_size_lookup:
            return self._supported_size_lookup[lookup_key]
        # If not directly supported, try to find closest match
        normalized = dimension_normalized
        if "x" in normalized:
            try:
                width_str, height_str = normalized.split("x", 1)
                width, height = int(width_str.strip()), int(height_str.strip())
                if width > 0 and height > 0:
                    return self._dimensions_to_size_param(width, height)
            except ValueError:
                pass
        return self.valves.DEFAULT_SIZE

    async def _analyse_prompt_with_task_model(
        self,
        prompt: str,
        has_images: bool,
        original_image_size: Optional[str],
        __user__: dict,
        body: dict,
        user_obj: Optional[UserModel],
        __request__: Request,
        emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """Use task model to determine optimal generation parameters."""
        if not self.valves.TASK_MODEL_ENABLED:
            return self._get_fallback_parameters(prompt, original_image_size, has_images)
        # Build analysis prompt based on whether we're editing or generating
        if has_images and original_image_size:
            analysis_prompt = f"""Analyse this image editing prompt and determine optimal parameters:
Prompt: "{prompt}"
Has reference images: {has_images}
Original image size: {original_image_size}
Rules for parameter determination:
1. **intent**: Determine if user wants to generate new image or edit existing one
   - "generate" for: new image requests, create, make, draw, produce, show me, another
   - "edit" for: modify, change, alter, adjust, fix, recolor, add, remove, replace
   - If has_images=True and user wants modifications: "edit"
   - If has_images=True but user wants new/different image: "generate"
2. **watermark**: 
   - false for professional/artistic/commercial use (headshots, logos, artwork, professional photos)
   - true for casual/social media/meme content
3. **size**: For IMAGE EDITING, preserve original size UNLESS user explicitly requests resize
   - If editing and NO resize mentioned: return original_image_size ({original_image_size})
   - If editing and resize requested: extract new size from prompt
   - Supported sizes: 1024x1024, 2048x2048, 2304x1728, 1728x2304, 2560x1440, 1440x2560, 2496x1664, 1664x2496, 3024x1296, 4096x4096, 6240x2656, 1K, 2K, 4K
   - Look for resize keywords: "resize", "make it", "change size", "bigger", "smaller", specific dimensions
Examples for editing:
- "remove the tyre on the bottom left" → {{"intent": "edit", "watermark": true, "size": "{original_image_size}"}}
- "change the bird to red" → {{"intent": "edit", "watermark": true, "size": "{original_image_size}"}}
- "resize this to 1024x1024 and remove the car" → {{"intent": "edit", "watermark": true, "size": "1024x1024"}}
- "make this image smaller and change the color" → {{"intent": "edit", "watermark": true, "size": "1024x1024"}}
Respond ONLY with valid JSON:"""
        else:
            analysis_prompt = f"""Analyse this image generation prompt and determine optimal parameters:
Prompt: "{prompt}"
Has reference images: {has_images}
Rules for parameter determination:
1. **intent**: Determine if user wants to generate new image or edit existing one
   - "generate" for: new image requests, create, make, draw, produce, show me, another
   - "edit" for: modify, change, alter, adjust, fix, recolor, add, remove, replace
2. **watermark**: 
   - false for professional/artistic/commercial use (headshots, logos, artwork, professional photos)
   - true for casual/social media/meme content
3. **size**: Extract size from natural language, choose from supported values:
   - Exact dimensions: 1024x1024, 2048x2048, 2304x1728, 1728x2304, 2560x1440, 1440x2560, 2496x1664, 1664x2496, 3024x1296, 4096x4096, 6240x2656
   - Resolution shorthand: 1K, 2K, 4K
   - Common mappings: square/1:1→2048x2048, 4:3→2304x1728, 3:4/portrait→1728x2304, 16:9/widescreen→2560x1440, 9:16/vertical→1440x2560, 21:9/ultrawide→3024x1296
   - Default: 2048x2048 if no size specified
Examples:
- "make a Melbourne skyline in 4K" → {{"intent": "generate", "watermark": true, "size": "4K"}}
- "professional headshot 16:9" → {{"intent": "generate", "watermark": false, "size": "2560x1440"}}
- "create square logo 1024x1024" → {{"intent": "generate", "watermark": false, "size": "1024x1024"}}
Respond ONLY with valid JSON:"""
        try:
            task_model = self._resolve_task_model(body, user_obj, __request__)
            await self.emit_status("Analysing prompt...", emitter=emitter)
            resolved_user = user_obj or await self._get_user_by_id(__user__["id"])
            if not resolved_user:
                logger.error("Unable to resolve user for task model call")
                raise RuntimeError("user_lookup_failed")
            # Structure the request body properly
            form_data = {
                "model": task_model,
                "messages": [{"role": "user", "content": analysis_prompt}],
                "max_tokens": 150,
                "temperature": 0.1,
                "stream": False,
            }
            # Make the internal call with proper objects
            response = await generate_chat_completions(
                form_data=form_data,
                user=resolved_user,
                request=__request__,
            )
            content = await self._read_model_response_content(response)
            if content:
                # Extract JSON from response
                json_blob = self._extract_first_json_object(content)
                if json_blob:
                    try:
                        params = json.loads(json_blob)
                        if not isinstance(params, dict):
                            logger.error("Task model JSON root is not an object")
                        else:
                            validated_params = self._validate_task_model_params(params, prompt, original_image_size)
                            logger.info(f"Task model determined parameters: {validated_params}")
                            return validated_params
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from task model: {e}")
                else:
                    logger.error("Task model response did not contain valid JSON")
                    logger.debug(f"Task model response snippet: {content[:1000]}")
            else:
                logger.error("Task model returned empty response")
        except Exception as e:
            logger.error(f"Task model call failed: {e}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")
        # Fallback to valve defaults
        logger.info("Task model failed, using fallback analysis")
        return self._get_fallback_parameters(prompt, original_image_size, has_images)

    def _validate_task_model_params(
        self, params: Dict[str, Any], prompt: str, original_image_size: Optional[str]
    ) -> Dict[str, Any]:
        """Validate and sanitise parameters from task model."""
        validated = {}
        # intent
        intent = params.get("intent", "generate")
        validated["intent"] = intent if intent in ["generate", "edit"] else "generate"
        # watermark
        watermark = params.get("watermark", True)
        validated["watermark"] = bool(watermark)
        # size - validate against supported sizes
        requested_size = params.get("size", original_image_size or self.valves.DEFAULT_SIZE)
        validated["size"] = self._validate_and_get_size(str(requested_size))
        return validated

    def _resolve_task_model(
        self,
        body: Dict[str, Any],
        user_obj: Optional[UserModel],
        __request__: Request,
    ) -> str:
        """Resolve which task model to call using Open WebUI preferences."""

        def _extract(source: Any) -> Optional[str]:
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

        return self.valves.TASK_MODEL_FALLBACK

    @staticmethod
    def _object_to_mapping(source: Any) -> Dict[str, Any]:
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
        try:
            return await run_in_threadpool(Users.get_user_by_id, user_id)
        except Exception as exc:
            logger.error(f"Failed to load user {user_id}: {exc}")
            return None

    async def _get_file_by_id(self, file_id: str):
        try:
            from open_webui.models.files import Files

            return await run_in_threadpool(Files.get_file_by_id, file_id)
        except Exception as exc:
            logger.error(f"Failed to load file {file_id}: {exc}")
            return None

    async def _read_file_bytes(self, path: str) -> bytes:
        """Read file contents without blocking the event loop."""
        def _read() -> bytes:
            with open(path, "rb") as f:
                return f.read()

        return await run_in_threadpool(_read)

    def _get_fallback_parameters(
        self,
        prompt: str = "",
        original_image_size: Optional[str] = None,
        has_images: bool = False,
    ) -> Dict[str, Any]:
        """Get fallback parameters from valves and prompt analysis."""
        # For editing, preserve original size unless explicitly requesting resize
        if original_image_size and not self._prompt_requests_resize(prompt):
            size = original_image_size
        else:
            # Extract size from prompt for generation or explicit resize
            extracted_size = self._extract_size_from_prompt(prompt) if prompt else self.valves.DEFAULT_SIZE
            size = self._validate_and_get_size(extracted_size)
        intent = "edit" if (original_image_size or has_images) else "generate"
        return {
            "intent": intent,
            "watermark": self.valves.WATERMARK,
            "size": size,
        }

    def _prompt_requests_resize(self, prompt: str) -> bool:
        """Check if prompt explicitly requests resizing."""
        prompt_lower = prompt.lower()
        dimension_patterns = [
            r"\d{3,4}\s*[x×]\s*\d{3,4}",
            r"\d{3,4}\s*by\s*\d{3,4}",
        ]
        for pattern in dimension_patterns:
            if re.search(pattern, prompt_lower):
                return True
        intent_keywords = [
            "resize",
            "rescale",
            "scale",
            "change size",
            "adjust size",
            "make it",
            "make larger",
            "make smaller",
            "set",
            "increase",
            "decrease",
            "bigger",
            "smaller",
            "larger",
            "reduce",
        ]
        size_indicators = [
            "resolution",
            "dimensions",
            "dimension",
            "aspect ratio",
            "ratio",
            "size",
        ]
        numeric_indicators = ["1k", "2k", "4k", "1024", "2048", "4096"]
        has_intent = any(keyword in prompt_lower for keyword in intent_keywords)
        has_size_keyword = any(keyword in prompt_lower for keyword in size_indicators)
        has_numeric_hint = any(token in prompt_lower for token in numeric_indicators)
        return has_intent and (has_size_keyword or has_numeric_hint)

    async def pipes(self) -> List[dict]:
        return [{"id": "seedream-4-0", "name": "ByteDance: Seedream 4.0"}]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> StreamingResponse:
        # Re-apply in case valves changed at runtime
        self._apply_logging_valve()
        user = await self._get_user_by_id(__user__["id"])

        async def stream_response():
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
                # Extract prompt and images
                prompt, images = await self._extract_prompt_and_images(messages)
                if not self.valves.API_KEY:
                    yield self._format_data(
                        is_stream=is_stream,
                        content="Error: API_KEY not set in valves.",
                        finish_reason="stop",
                    )
                    if is_stream:
                        yield "data: [DONE]\n\n"
                    return
                # Get original image size for editing
                original_image_size = None
                if images:
                    await self.emit_status("Detecting original image dimensions...", emitter=__event_emitter__)
                    first_image = images[0]
                    width, height = self._get_image_dimensions(
                        first_image.get("data", "")
                    )
                    if width and height:
                        original_image_size = self._dimensions_to_size_param(width, height)
                        logger.info(f"Original image dimensions: {width}x{height} -> {original_image_size}")
                # Get dynamic parameters from task model
                dynamic_params = await self._analyse_prompt_with_task_model(
                    prompt,
                    bool(images),
                    original_image_size,
                    __user__,
                    body,
                    user,
                    __request__,
                    emitter=__event_emitter__,
                )
                # Build request JSON using dynamic parameters
                endpoint = "/images/generations"
                json_data: Dict[str, Any] = {
                    "model": model,
                    "prompt": prompt,
                    "response_format": "b64_json",
                    "watermark": dynamic_params["watermark"],
                    "size": dynamic_params["size"],
                }
                # Handle intent and images
                if dynamic_params["intent"] == "edit" and not images:
                    logger.info("Task model suggested edit but no images provided, treating as generate")
                elif dynamic_params["intent"] == "generate" and images:
                    logger.info("Task model suggested generate despite images present, ignoring images for new generation")
                    images = []  # Clear images for new generation
                # Prepare images for editing
                image_uris: List[str] = []
                if images and dynamic_params["intent"] == "edit":
                    await self.emit_status("Preparing image editing...", emitter=__event_emitter__)
                    for im in images:
                        mime = (im.get("mimeType") or "").lower().strip()
                        if mime not in ("image/png", "image/jpeg", "image/jpg"):
                            continue
                        if mime == "image/jpg":
                            mime = "image/jpeg"
                        b64 = im.get("data", "")
                        if not b64:
                            continue
                        # Size check (optional)
                        try:
                            if (len(b64) * 3) // 4 > 10 * 1024 * 1024:
                                logger.info("Skipping image >10MB after decode estimate")
                                continue
                        except Exception:
                            pass
                        image_uris.append(f"data:{mime};base64,{b64}")
                    if image_uris:
                        if len(image_uris) == 1:
                            json_data["image"] = image_uris[0]
                        else:
                            json_data["image"] = image_uris[:10]
                # Show size preservation info in status
                size_info = f"size: {dynamic_params['size']}"
                if original_image_size and dynamic_params["size"] == original_image_size:
                    size_info += " (preserved)"
                elif original_image_size and dynamic_params["size"] != original_image_size:
                    size_info += f" (resized from {original_image_size})"
                params_info = f"({dynamic_params['intent']}, {size_info}, watermark: {dynamic_params['watermark']})"
                status_message = f"Editing image {params_info}..." if image_uris else f"Generating image {params_info}..."
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
                            "prompt_chars": len(json_data.get("prompt", "")),
                        }
                        if "image" in json_data:
                            redacted_payload["image_count"] = (
                                len(json_data["image"])
                                if isinstance(json_data["image"], list)
                                else 1
                            )
                        logger.info("Request payload summary: %s", redacted_payload)
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

    async def _extract_prompt_and_images(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, str]]]:
        """Extract prompt and image data from messages."""
        prompt = ""
        images: List[Dict[str, str]] = []
        # Take prompt from the most recent user message
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        prompt += item.get("text", "") + " "
            elif isinstance(content, str):
                prompt = content
            break
        prompt = prompt.strip()
        # Gather images from the last message (user or assistant) that contains images
        for message in reversed(messages):
            role = message.get("role", "")
            if role not in ["user", "assistant"]:
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if not url:
                            continue
                        if url.startswith("data:"):
                            parts = url.split(";base64,", 1)
                            if len(parts) == 2:
                                mime_type = parts[0].replace("data:", "", 1).lower()
                                data = parts[1]
                                images.append({"mimeType": mime_type, "data": data})
                        elif "/api/v1/files/" in url or "/files/" in url:
                            file_id = url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0] if "/api/v1/files/" in url else url.split("/files/")[-1].split("/")[0].split("?")[0]
                            file_item = await self._get_file_by_id(file_id)
                            if file_item and file_item.path:
                                try:
                                    file_data = await self._read_file_bytes(file_item.path)
                                except Exception as exc:
                                    logger.error(f"Failed to read file {file_id} from disk: {exc}")
                                    continue
                                data = base64.b64encode(file_data).decode("utf-8")
                                meta = (file_item.meta or {})
                                mime_type = meta.get("content_type", "image/png")
                                images.append({"mimeType": mime_type, "data": data})
                            else:
                                logger.error(f"Failed to fetch file {file_id}: not found")
                        else:
                            remote_image = await self._fetch_remote_image(url)
                            if remote_image:
                                images.append(remote_image)
            elif isinstance(content, str):
                # Inline data URI markdown
                for mime_type, data in re.findall(
                    r"!\[[^\]]*\]\(data:([^;]+);base64,([^)]+)\)", content
                ):
                    images.append({"mimeType": mime_type.lower(), "data": data})
                # File reference markdown
                for file_url in re.findall(
                    r"!\[[^\]]*\]\((/api/v1/files/[^)]+|/files/[^)]+)\)", content
                ):
                    file_id = file_url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0] if "/api/v1/files/" in file_url else file_url.split("/files/")[-1].split("/")[0].split("?")[0]
                    file_item = await self._get_file_by_id(file_id)
                    if file_item and file_item.path:
                        try:
                            file_data = await self._read_file_bytes(file_item.path)
                        except Exception as exc:
                            logger.error(f"Failed to read file {file_id} from disk: {exc}")
                            continue
                        data = base64.b64encode(file_data).decode("utf-8")
                        meta = (file_item.meta or {})
                        mime_type = meta.get("content_type", "image/png")
                        images.append({"mimeType": mime_type, "data": data})
                    else:
                        logger.error(f"Failed to fetch file {file_id}: not found")
                # Remote image markdown
                for remote_url in re.findall(
                    r"!\[[^\]]*\]\((https?://[^)]+)\)", content
                ):
                    remote_image = await self._fetch_remote_image(remote_url)
                    if remote_image:
                        images.append(remote_image)
            if images:
                break
        # Clean prompt of image markdown
        if images and isinstance(prompt, str):
            prompt = re.sub(
                r"!\[[^\]]*\]\(data:[^;]+;base64,[^)]+\)", "", prompt
            ).strip()
            prompt = re.sub(
                r"!\[[^\]]*\]\((/api/v1/files/[^)]+|/files/[^)]+)\)", "", prompt
            ).strip()
        logger.info(f"Extracted prompt: '{prompt[:100]}...', found {len(images)} image(s)")
        return prompt, images

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
        return {"mimeType": mime_type, "data": data}

    async def _upload_image(
        self, __request__: Request, user: UserModel, image_data: str, mime_type: str
    ) -> str:
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
