"""
title: ByteDance Seedream 4.0 Image Generation & Editing Pipe
description: ByteDance Seedream 4.0 plugin with task model integration
id: seedream-4-0
author: rbb-dev
author_url: https://github.com/rbb-dev/
version: 0.9.0
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
        API_BASE_URL: str = Field(
            default="https://api.cometapi.com/v1", description="API base URL"
        )
        # Logging
        ENABLE_LOGGING: bool = Field(
            default=False,
            description="Enable info/debug logs for this plugin. When False, only errors are logged.",
        )
        # Model and basic generation options
        MODEL: str = Field(
            default="bytedance-seedream-4-0-250828",
            description="Doubao Seedream 4.0 model id",
        )
        # Task model settings
        TASK_MODEL_ENABLED: bool = Field(
            default=True, description="Enable task model for dynamic parameter detection"
        )
        # Fallback defaults when task model is disabled/unavailable
        WATERMARK: bool = Field(
            default=True,
            description="Fallback: Add watermark when task model fails or is disabled.",
        )
        DEFAULT_SIZE: str = Field(
            default="2048x2048",
            description="Default image size when not specified or unsupported size requested.",
        )
        # HTTP client
        REQUEST_TIMEOUT: int = Field(
            default=600, description="Request timeout in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.emitter: Optional[Callable[[dict], Awaitable[None]]] = None
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
        self, message: str, done: bool = False, show_in_chat: bool = False
    ):
        """Emit status updates to the client."""
        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )
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
        supported_dimensions = [
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
        min_diff = float("inf")
        closest_size = self.valves.DEFAULT_SIZE
        for sup_w, sup_h in supported_dimensions:
            # Calculate difference based on total pixel count and aspect ratio similarity
            pixel_diff = abs((width * height) - (sup_w * sup_h))
            aspect_diff = (
                abs((width / height) - (sup_w / sup_h)) * 1000
            )  # Weight aspect ratio
            total_diff = pixel_diff + aspect_diff
            if total_diff < min_diff:
                min_diff = total_diff
                closest_size = f"{sup_w}x{sup_h}"
        logger.info(
            f"Mapped {width}x{height} to closest supported size: {closest_size}"
        )
        return closest_size

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
            match = re.search(pattern, prompt)
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
        if requested_size in self.supported_sizes:
            return requested_size
        # If not directly supported, try to find closest match
        if "x" in requested_size:
            try:
                width, height = map(int, requested_size.split("x"))
                # Check if dimensions are within valid range
                total_pixels = width * height
                aspect_ratio = max(width, height) / min(width, height)
                if (1024 * 1024 <= total_pixels <= 4096 * 4096) and (
                    1 / 16 <= aspect_ratio <= 16
                ):
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
        __request__: Request,
    ) -> Dict[str, Any]:
        """Use task model to determine optimal generation parameters."""
        if not self.valves.TASK_MODEL_ENABLED:
            return self._get_fallback_parameters(prompt, original_image_size)
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
            await self.emit_status("🤖 Analysing prompt...")
            # Get the actual user object (not just the dict)
            user_obj = Users.get_user_by_id(__user__["id"])
            # Structure the request body properly
            form_data = {
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": analysis_prompt}],
                "max_tokens": 150,
                "temperature": 0.1,
                "stream": False,
            }
            # Make the internal call with proper objects
            response = await generate_chat_completions(
                form_data=form_data,
                user=user_obj,  # Pass the actual User object, not dict
                request=__request__,
            )
            content = ""
            # Handle the response properly
            if hasattr(response, "body_iterator"):
                # Streaming response
                content_parts = []
                async for chunk in response.body_iterator:
                    if chunk:
                        try:
                            chunk_str = chunk.decode("utf-8")
                            for line in chunk_str.split("\n"):
                                if line.startswith("data: ") and not line.startswith(
                                    "data: [DONE]"
                                ):
                                    try:
                                        data = json.loads(line[6:])
                                        if "choices" in data and data["choices"]:
                                            delta_content = (
                                                data["choices"][0]
                                                .get("delta", {})
                                                .get("content", "")
                                            )
                                            if delta_content:
                                                content_parts.append(delta_content)
                                    except json.JSONDecodeError:
                                        continue
                        except Exception:
                            continue
                content = "".join(content_parts)
            elif isinstance(response, dict):
                # Direct response
                if "choices" in response and response["choices"]:
                    content = (
                        response["choices"][0].get("message", {}).get("content", "")
                    )
            else:
                # Other response types
                content = str(response)
            if content:
                # Extract JSON from response
                json_match = re.search(r"\{[^{}]*\}", content)
                if json_match:
                    try:
                        params = json.loads(json_match.group())
                        validated_params = self._validate_task_model_params(
                            params, prompt, original_image_size
                        )
                        logger.info(
                            f"Task model determined parameters: {validated_params}"
                        )
                        return validated_params
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from task model: {e}")
                else:
                    logger.error("Task model response did not contain valid JSON")
                    logger.debug(f"Task model response: {content[:200]}")
            else:
                logger.error("Task model returned empty response")
        except Exception as e:
            logger.error(f"Task model call failed: {e}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")
        # Fallback to valve defaults
        logger.info("Task model failed, using fallback analysis")
        return self._get_fallback_parameters(prompt, original_image_size)

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
        requested_size = params.get(
            "size", original_image_size or self.valves.DEFAULT_SIZE
        )
        validated["size"] = self._validate_and_get_size(str(requested_size))
        return validated

    def _get_fallback_parameters(
        self, prompt: str = "", original_image_size: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get fallback parameters from valves and prompt analysis."""
        # For editing, preserve original size unless explicitly requesting resize
        if original_image_size and not self._prompt_requests_resize(prompt):
            size = original_image_size
        else:
            # Extract size from prompt for generation or explicit resize
            extracted_size = (
                self._extract_size_from_prompt(prompt)
                if prompt
                else self.valves.DEFAULT_SIZE
            )
            size = self._validate_and_get_size(extracted_size)
        return {
            "intent": "edit" if original_image_size else "generate",
            "watermark": self.valves.WATERMARK,
            "size": size,
        }

    def _prompt_requests_resize(self, prompt: str) -> bool:
        """Check if prompt explicitly requests resizing."""
        prompt_lower = prompt.lower()
        resize_keywords = [
            "resize",
            "make it",
            "change size",
            "bigger",
            "smaller",
            "larger",
            "reduce size",
            "increase size",
            "scale",
            "dimensions",
            "resolution",
            "1k",
            "2k",
            "4k",
            "1024",
            "2048",
            "4096",
        ]
        # Check for dimension patterns
        dimension_patterns = [
            r"\d{3,4}\s*[x×]\s*\d{3,4}",
            r"\d{3,4}\s*by\s*\d{3,4}",
        ]
        for pattern in dimension_patterns:
            if re.search(pattern, prompt):
                return True
        return any(keyword in prompt_lower for keyword in resize_keywords)

    async def pipes(self) -> List[dict]:
        return [{"id": "seedream-4-0", "name": "ByteDance: Seedream 4.0"}]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> StreamingResponse:
        self.emitter = __event_emitter__
        # Re-apply in case valves changed at runtime
        self._apply_logging_valve()
        user = Users.get_user_by_id(__user__["id"])

        async def stream_response():
            try:
                model = self.valves.MODEL
                messages = body.get("messages", [])
                is_stream = bool(body.get("stream", False))
                # Extract prompt and images
                prompt, images = await self._extract_prompt_and_images(messages)
                if not self.valves.API_KEY:
                    yield self._format_data(
                        is_stream=is_stream,
                        content="Error: API_KEY not set in valves.",
                    )
                    return
                # Get original image size for editing
                original_image_size = None
                if images:
                    await self.emit_status("📏 Detecting original image dimensions...")
                    first_image = images[0]
                    width, height = self._get_image_dimensions(
                        first_image.get("data", "")
                    )
                    if width and height:
                        original_image_size = self._dimensions_to_size_param(
                            width, height
                        )
                        logger.info(
                            f"Original image dimensions: {width}x{height} -> {original_image_size}"
                        )
                # Get dynamic parameters from task model
                dynamic_params = await self._analyse_prompt_with_task_model(
                    prompt, bool(images), original_image_size, __user__, __request__
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
                    logger.info(
                        "Task model suggested edit but no images provided, treating as generate"
                    )
                elif dynamic_params["intent"] == "generate" and images:
                    logger.info(
                        "Task model suggested generate despite images present, ignoring images for new generation"
                    )
                    images = []  # Clear images for new generation
                # Prepare images for editing
                image_uris: List[str] = []
                if images and dynamic_params["intent"] == "edit":
                    await self.emit_status("✂️ Preparing image editing...")
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
                                logger.info(
                                    "Skipping image >10MB after decode estimate"
                                )
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
                if (
                    original_image_size
                    and dynamic_params["size"] == original_image_size
                ):
                    size_info += " (preserved)"
                elif (
                    original_image_size
                    and dynamic_params["size"] != original_image_size
                ):
                    size_info += f" (resized from {original_image_size})"
                params_info = f"({dynamic_params['intent']}, {size_info}, watermark: {dynamic_params['watermark']})"
                await self.emit_status(
                    f"🔄 Editing image {params_info}..."
                    if image_uris
                    else f"🖼️ Generating image {params_info}..."
                )
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
                        logger.info(
                            f"Request payload: {json.dumps(json_data, indent=2)}"
                        )
                        response = await client.post(endpoint, json=json_data)
                        # Handle array fallback for older gateways
                        if response.status_code == 400 and isinstance(
                            json_data.get("image"), list
                        ):
                            try:
                                msg = response.text
                                logger.debug(
                                    f"API Error Response: {json.dumps(response.json(), indent=2)}"
                                )
                            except Exception:
                                msg = ""
                            if (
                                "image of type string" in msg
                                or "unmarshal array into" in msg
                            ):
                                logger.error(
                                    "API gateway rejected image array, retrying with first image string."
                                )
                                retry_payload = dict(json_data)
                                retry_payload["image"] = json_data["image"][0]
                                response = await client.post(
                                    endpoint, json=retry_payload
                                )
                        response.raise_for_status()
                except httpx.HTTPError as e:
                    logger.error(f"API request failed: {e}")
                    error_status = await self.emit_status(
                        "❌ An error occurred while calling API", True, True
                    )
                    yield self._format_data(
                        is_stream=is_stream,
                        content=f"{error_status}Error from API: {str(e)}",
                    )
                    return
                response_data = response.json()
                # Process images
                await self.emit_status("✅ Image processing complete!", True)
                image_markdown = []
                for i, item in enumerate(response_data.get("data", []), start=1):
                    if "b64_json" in item:
                        try:
                            image_url = self._upload_image(
                                __request__=__request__,
                                user=user,
                                image_data=item["b64_json"],
                                mime_type="image/jpeg",
                            )
                            image_markdown.append(f"![image_{i}]({image_url})")
                        except Exception as e:
                            logger.error(f"Image upload failed: {e}")
                            error_status = await self.emit_status(
                                "❌ An error occurred while uploading image", True, True
                            )
                            yield self._format_data(
                                is_stream=is_stream,
                                content=f"{error_status}Error uploading image: {str(e)}",
                            )
                            return
                    elif "url" in item:
                        image_markdown.append(f"![image_{i}]({item['url']})")
                content = (
                    "\n\n".join(image_markdown)
                    if image_markdown
                    else "No images returned."
                )
                # Return response
                if is_stream:
                    yield self._format_data(
                        is_stream=True, model=model, content=content
                    )
                    yield self._format_data(
                        is_stream=True,
                        model=model,
                        content=None,
                        usage=response_data.get("usage"),
                    )
                else:
                    yield self._format_data(
                        is_stream=False,
                        model=model,
                        content=content,
                        usage=response_data.get("usage"),
                    )
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                error_status = await self.emit_status(
                    "❌ An error occurred while processing request", True, True
                )
                yield self._format_data(
                    is_stream=body.get("stream", False),
                    content=f"{error_status}Error processing request: {str(e)}",
                )

        return StreamingResponse(stream_response())

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
                            file_id = (
                                url.split("/api/v1/files/")[-1]
                                .split("/")[0]
                                .split("?")[0]
                                if "/api/v1/files/" in url
                                else url.split("/files/")[-1]
                                .split("/")[0]
                                .split("?")[0]
                            )
                            try:
                                from open_webui.models.files import Files

                                file_item = Files.get_file_by_id(file_id)
                                if file_item and file_item.path:
                                    with open(file_item.path, "rb") as f:
                                        file_data = f.read()
                                    data = base64.b64encode(file_data).decode("utf-8")
                                    mime_type = file_item.meta.get(
                                        "content_type", "image/png"
                                    )
                                    images.append({"mimeType": mime_type, "data": data})
                            except Exception as e:
                                logger.error(f"Failed to fetch file {file_id}: {e}")
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
                    file_id = (
                        file_url.split("/api/v1/files/")[-1].split("/")[0].split("?")[0]
                        if "/api/v1/files/" in file_url
                        else file_url.split("/files/")[-1].split("/")[0].split("?")[0]
                    )
                    try:
                        from open_webui.models.files import Files

                        file_item = Files.get_file_by_id(file_id)
                        if file_item and file_item.path:
                            with open(file_item.path, "rb") as f:
                                file_data = f.read()
                            data = base64.b64encode(file_data).decode("utf-8")
                            mime_type = file_item.meta.get("content_type", "image/png")
                            images.append({"mimeType": mime_type, "data": data})
                    except Exception as e:
                        logger.error(f"Failed to fetch file {file_id}: {e}")
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
        logger.info(
            f"Extracted prompt: '{prompt[:100]}...', found {len(images)} image(s)"
        )
        return prompt, images

    def _upload_image(
        self, __request__: Request, user: UserModel, image_data: str, mime_type: str
    ) -> str:
        try:
            file_item = upload_file(
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
            image_url = __request__.app.url_path_for(
                "get_file_content_by_id", id=file_item.id
            )
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
    ) -> str:
        """Format the response data in the expected OpenAI-compatible format."""
        data = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk" if is_stream else "chat.completion",
            "created": int(time.time()),
            "model": model,
        }
        if content is not None:
            data["choices"] = [
                {
                    "finish_reason": "stop" if not is_stream else None,
                    "index": 0,
                    "delta" if is_stream else "message": {
                        "role": "assistant",
                        "content": content,
                    },
                }
            ]
        if usage:
            data["usage"] = usage
        return f"data: {json.dumps(data)}\n\n" if is_stream else json.dumps(data)
