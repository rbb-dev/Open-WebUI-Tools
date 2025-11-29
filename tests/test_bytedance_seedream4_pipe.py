import json
import sys
import types
from types import SimpleNamespace

import pytest


def _install_stub_modules():
    if "open_webui" in sys.modules:
        return

    open_webui = types.ModuleType("open_webui")
    sys.modules["open_webui"] = open_webui

    routers = types.ModuleType("open_webui.routers")
    sys.modules["open_webui.routers"] = routers

    files_mod = types.ModuleType("open_webui.routers.files")

    def _stub_upload_file(*args, **kwargs):  # pragma: no cover - should never run in tests
        raise RuntimeError("upload_file stub invoked")

    files_mod.upload_file = _stub_upload_file
    routers.files = files_mod
    sys.modules["open_webui.routers.files"] = files_mod
    open_webui.routers = routers

    models_pkg = types.ModuleType("open_webui.models")
    sys.modules["open_webui.models"] = models_pkg

    users_mod = types.ModuleType("open_webui.models.users")

    class UserModel:  # pragma: no cover - placeholder for type hints
        pass

    class Users:  # pragma: no cover - placeholder
        @staticmethod
        def get_user_by_id(_):
            return None

    users_mod.UserModel = UserModel
    users_mod.Users = Users
    models_pkg.users = users_mod
    sys.modules["open_webui.models.users"] = users_mod

    main_mod = types.ModuleType("open_webui.main")

    async def _stub_generate(*_, **__):  # pragma: no cover - patched in tests
        raise RuntimeError("generate_chat_completions stub invoked")

    main_mod.generate_chat_completions = _stub_generate
    sys.modules["open_webui.main"] = main_mod
    open_webui.main = main_mod


_install_stub_modules()

from pipes.bytedance_seedream4 import bytedance_seedream4_pipe as seedream_pipe
from pipes.bytedance_seedream4.bytedance_seedream4_pipe import Pipe


@pytest.fixture
def pipe_instance():
    return Pipe()


@pytest.mark.asyncio
async def test_analyse_prompt_with_task_model_returns_valid_payload(monkeypatch, pipe_instance):
    async def fake_get_user(user_id):
        return SimpleNamespace(id=user_id, settings={})

    monkeypatch.setattr(pipe_instance, "_get_user_by_id", fake_get_user)

    expected = {
        "prompt": "Resize this image to 2048x2048 and remove watermark [image:0]",
        "intent": "edit",
        "use_reference_images": True,
        "watermark": False,
        "size": "2048x2048",
        "resize_target": {"width": 2048, "height": 2048},
        "notes": "resize and edit",
        "status_message": "Editing with resize",
        "image_plan": [
            {"index": 0, "action": "use", "role": "base", "target_size": "2048x2048"}
        ],
    }

    async def fake_generate_chat_completions(*_, **__):
        return {"choices": [{"message": {"content": json.dumps(expected)}}]}

    monkeypatch.setattr(seedream_pipe, "generate_chat_completions", fake_generate_chat_completions)

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings={})))
    conversation = [
        {"message_index": 0, "role": "user", "text": "resize this image", "image_refs": [0]}
    ]

    result = await pipe_instance._analyse_prompt_with_task_model(
        conversation=conversation,
        image_context={
            "count": 1,
            "details": [
                {
                    "index": 0,
                    "mime_type": "image/png",
                    "width": 1024,
                    "height": 1024,
                    "size_label": "1024x1024",
                }
            ],
        },
        raw_user_prompt="resize this image",
        __user__={"id": "user-1"},
        body={},
        user_obj=None,
        __request__=request,
        emitter=None,
    )

    assert result["prompt"] == "Resize this image to 2048x2048 and remove watermark"
    assert result["intent"] == "edit"
    assert result["use_reference_images"] is True
    assert result["size"] == "2048x2048"
    assert result["resize_target"] == {"width": 2048, "height": 2048}
    assert result["notes"] == "resize and edit"
    assert result["image_plan"] == expected["image_plan"]
    assert result["status_message"] == "Editing with resize"


def test_validate_task_model_params_rejects_unsupported_size(pipe_instance):
    with pytest.raises(ValueError):
        pipe_instance._validate_task_model_params(
            {
                "intent": "generate",
                "size": "999x999",
                "use_reference_images": False,
                "watermark": True,
            },
            image_count=0,
            fallback_prompt="fallback",
        )


def test_validate_task_model_params_strips_placeholders(pipe_instance):
    params = {
        "prompt": "Paint the dog [image:1] orange",
        "intent": "edit",
        "size": "2048x2048",
        "use_reference_images": True,
        "watermark": False,
        "image_plan": [],
    }

    result = pipe_instance._validate_task_model_params(
        params,
        image_count=1,
        fallback_prompt="ignored",
    )

    assert result["prompt"] == "Paint the dog orange"


@pytest.mark.asyncio
async def test_collect_conversation_and_images_handles_inline_data_uri(pipe_instance):
    base64_data = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "Xw8AAkMBgXkGLdIAAAAASUVORK5CYII="
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Use this ![alt](data:image/png;base64,{base64_data}) for the edit",
                }
            ],
        }
    ]

    conversation, images, last_user_text = await pipe_instance._collect_conversation_and_images(messages)

    assert conversation[0]["text"].startswith("Use this")
    assert len(images) == 1
    assert images[0]["mimeType"] == "image/png"
    assert images[0]["data"] == base64_data
    assert last_user_text.startswith("Use this")
    assert images[0]["origin"] == {"type": "data_uri"}
