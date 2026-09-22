"""
ComfyUI on Modal
================
Minimal, production-grade ComfyUI server deployed on Modal.

Features:
- Instant cold starts via GPU and memory snapshotting (~1–2 seconds)
- Reverse-proxy fix for seamless workflow saving behind Modal edge proxies
- Persistent storage for models and generated outputs
- Built-in ComfyUI-Manager for easy extension and node management
- Pre-loaded starter checkpoint (DreamShaper 8) for immediate generation
- Fast multi-connection model downloads from Hugging Face & Civitai

Commands:
    # 1. Start interactive dev server:
    uv run modal serve main.py

    # 2. Deploy permanent autoscaling service:
    uv run modal deploy main.py

    # 3. Download a model from Hugging Face:
    uv run modal run main.py::download_model --repo-id "Lykon/DreamShaper" --filename "DreamShaper_8_pruned.safetensors"

    # 4. Download a model from Civitai or direct URL:
    CIVITAI_TOKEN="optional_token" uv run modal run main.py::download_url \
        --url "https://civitai.com/api/download/models/128713" \
        --filename "dreamshaper_xl.safetensors"
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "comfyui-server"

# GPU type: A10G (default, 24GB VRAM), L4 (budget), A100-80GB (heavy workflows)
GPU_TYPE = os.environ.get("MODAL_GPU", "A10G")

# Toggle GPU & Memory Snapshotting (enabled by default for sub-2s cold boots)
ENABLE_GPU_SNAPSHOT = os.environ.get("MODAL_GPU_SNAPSHOT", "1").lower() in ("1", "true")

app = modal.App(name=APP_NAME)

# ---------------------------------------------------------------------------
# Persistent Storage Volumes
# ---------------------------------------------------------------------------
vol_models = modal.Volume.from_name("comfyui-models", create_if_missing=True)
vol_output = modal.Volume.from_name("comfyui-output", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container Image Definition
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "git-lfs",
        "wget",
        "curl",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libgl1",
        "libglib2.0-0",
        "axel",  # Accelerated multi-threaded downloader
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        # Clone ComfyUI core
        "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI",
        "cd /root/ComfyUI && pip install -r requirements.txt",
        # Clone ComfyUI-Manager
        "git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git /root/ComfyUI/custom_nodes/ComfyUI-Manager",
        "cd /root/ComfyUI/custom_nodes/ComfyUI-Manager && pip install -r requirements.txt",
    )
    .pip_install("huggingface_hub[hf_transfer]>=0.25.0")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
    .add_local_dir("workflows", remote_path="/root/workflows")
)

# ---------------------------------------------------------------------------
# ComfyUI Extra Model Paths Configuration
# ---------------------------------------------------------------------------
EXTRA_MODEL_PATHS_YAML = """
modal_storage:
    base_path: /root/models/
    checkpoints: checkpoints/
    clip: clip/
    clip_vision: clip_vision/
    configs: configs/
    controlnet: controlnet/
    diffusion_models: diffusion_models/
    embeddings: embeddings/
    loras: loras/
    text_encoders: text_encoders/
    upscale_models: upscale_models/
    vae: vae/
    vae_approx: vae_approx/
"""

# Patch for Modal edge reverse proxies:
# Modal decodes %2F to / in URL paths. ComfyUI frontend saves workflows to /userdata/workflows%2Ffoo.json.
# After decoding, ComfyUI's default single-segment route fails with HTTP 405.
# This patch registers a slash-tolerant {file:.+} route on ComfyUI's router.
REVERSE_PROXY_FIX_CODE = '''"""
Reverse proxy fix for Modal edge proxies.
Allows ComfyUI to save workflows when Modal decodes %2F into / in /userdata/{file}.
"""
from server import PromptServer

try:
    server = PromptServer.instance
    for route in list(server.app.router.routes()):
        resource = getattr(route, "resource", None)
        if resource:
            info = resource.get_info()
            path = info.get("path") or info.get("formatter") or str(resource)
            if "/userdata/{file}" in str(path):
                server.app.router.add_route(route.method, "/userdata/{file:.+}", route.handler)
    print("[ReverseProxyFix] Successfully registered slash-tolerant /userdata/{file:.+} routes.")
except Exception as e:
    print(f"[ReverseProxyFix] Warning: could not apply reverse proxy fix: {e}")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
'''

MODEL_SUBDIRS = [
    "checkpoints",
    "clip",
    "clip_vision",
    "configs",
    "controlnet",
    "diffusion_models",
    "embeddings",
    "loras",
    "text_encoders",
    "upscale_models",
    "vae",
    "vae_approx",
]


def ensure_environment():
    """Ensure directory structure, configuration files, and custom patches exist."""
    # Ensure volume model subdirectories exist
    for subdir in MODEL_SUBDIRS:
        os.makedirs(f"/root/models/{subdir}", exist_ok=True)
    os.makedirs("/root/output", exist_ok=True)
    os.makedirs("/root/ComfyUI/input", exist_ok=True)
    os.makedirs("/root/ComfyUI/user/default/workflows", exist_ok=True)

    # Configure extra model paths for ComfyUI
    config_path = Path("/root/ComfyUI/extra_model_paths.yaml")
    config_path.write_text(EXTRA_MODEL_PATHS_YAML.strip())

    # Install the reverse-proxy fix custom node
    proxy_fix_dir = Path("/root/ComfyUI/custom_nodes/reverse_proxy_fix")
    proxy_fix_dir.mkdir(parents=True, exist_ok=True)
    (proxy_fix_dir / "__init__.py").write_text(REVERSE_PROXY_FIX_CODE.strip())

    # Sync any pre-packaged workflows from workflows directory
    workflows_src = Path("/root/workflows")
    if workflows_src.exists():
        for wf in workflows_src.glob("*.json"):
            shutil.copy(wf, f"/root/ComfyUI/user/default/workflows/{wf.name}")


def ensure_starter_checkpoint():
    """Ensure at least one model exists so the default workflow works immediately."""
    ckpt_dir = Path("/root/models/checkpoints")
    existing_checkpoints = list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.ckpt"))
    diff_dir = Path("/root/models/diffusion_models")
    existing_diff = list(diff_dir.glob("*.safetensors")) if diff_dir.exists() else []

    if not existing_checkpoints and not existing_diff:
        print("\n[ComfyUI] No models found in persistent volume.")
        print("[ComfyUI] Downloading starter model 'DreamShaper 8' (~2 GB)...")
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="Lykon/DreamShaper",
            filename="DreamShaper_8_pruned.safetensors",
            local_dir="/root/models/checkpoints",
            token=os.environ.get("HF_TOKEN"),
        )
        vol_models.commit()
        print("[ComfyUI] Starter model downloaded successfully!\n")


def wait_for_port(port: int, timeout: int = 180):
    """Block until ComfyUI socket is accepting connections."""
    import socket
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"ComfyUI failed to start listening on port {port} within {timeout}s")


# ---------------------------------------------------------------------------
# ComfyUI Modal Service
# ---------------------------------------------------------------------------
COMFY_CMD = [
    "python",
    "/root/ComfyUI/main.py",
    "--listen", "0.0.0.0",
    "--port", "8188",
    "--output-directory", "/root/output",
    "--extra-model-paths-config", "/root/ComfyUI/extra_model_paths.yaml",
    "--preview-method", "auto",
]

if ENABLE_GPU_SNAPSHOT:
    @app.cls(
        image=image,
        gpu=GPU_TYPE,
        max_containers=1,
        timeout=3600 * 3,
        scaledown_window=180,  # Shut down after 3 minutes of inactivity
        enable_memory_snapshot=True,
        experimental_options={"enable_gpu_snapshot": True},
        volumes={
            "/root/models": vol_models,
            "/root/output": vol_output,
        },
    )
    @modal.concurrent(max_inputs=100)
    class ComfyUI:
        @modal.enter(snap=True)
        def start_checkpoint(self):
            """Runs during snapshot creation: boots ComfyUI and captures running state."""
            ensure_environment()
            ensure_starter_checkpoint()
            print(f"[ComfyUI] Booting on GPU [{GPU_TYPE}] for initial memory snapshot...")
            self.proc = subprocess.Popen(COMFY_CMD)
            wait_for_port(8188, timeout=180)
            print("[ComfyUI] Server ready. Modal is capturing memory snapshot...")

        @modal.enter(snap=False)
        def start_restore(self):
            """Runs on fast resume from snapshot (< 2 seconds)."""
            wait_for_port(8188, timeout=30)
            print("[ComfyUI] Restored from memory snapshot in ~1 second!")

        @modal.web_server(port=8188, startup_timeout=180)
        def ui(self):
            print("[ComfyUI] Serving web traffic on port 8188.")

        @modal.exit()
        def cleanup(self):
            proc = getattr(self, "proc", None)
            if proc is not None:
                try:
                    proc.terminate()
                except (ProcessLookupError, OSError):
                    pass
            print("[ComfyUI] Container shutdown cleanly.")

else:
    @app.cls(
        image=image,
        gpu=GPU_TYPE,
        max_containers=1,
        timeout=3600 * 3,
        scaledown_window=600,  # Shut down after 10 minutes of inactivity
        volumes={
            "/root/models": vol_models,
            "/root/output": vol_output,
        },
    )
    @modal.concurrent(max_inputs=100)
    class ComfyUI:
        @modal.enter()
        def start(self):
            ensure_environment()
            ensure_starter_checkpoint()
            print(f"[ComfyUI] Booting on GPU [{GPU_TYPE}]...")
            self.proc = subprocess.Popen(COMFY_CMD)
            wait_for_port(8188, timeout=180)
            print("[ComfyUI] Server listening on 0.0.0.0:8188.")

        @modal.web_server(port=8188, startup_timeout=180)
        def ui(self):
            print("[ComfyUI] Serving web traffic on port 8188.")

        @modal.exit()
        def cleanup(self):
            proc = getattr(self, "proc", None)
            if proc is not None:
                try:
                    proc.terminate()
                except (ProcessLookupError, OSError):
                    pass
            print("[ComfyUI] Container shutdown cleanly.")


# ---------------------------------------------------------------------------
# High-Speed Model Download Helpers
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/root/models": vol_models},
    timeout=3600,
)
def download_model(repo_id: str, filename: str, subfolder: str = "checkpoints"):
    """Download any safetensors or checkpoint file from Hugging Face into persistent storage."""
    from huggingface_hub import hf_hub_download

    dest_dir = f"/root/models/{subfolder}"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading '{filename}' from Hugging Face repo '{repo_id}' to '{dest_dir}'...")

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    vol_models.commit()
    print(f"Successfully saved to: {path}")


@app.function(
    image=image,
    volumes={"/root/models": vol_models},
    timeout=3600,
)
def download_url(url: str, filename: str, subfolder: str = "checkpoints"):
    """Download a file from an HTTP(S) URL (Civitai, etc.) using axel multi-connection acceleration."""
    dest_dir = Path(f"/root/models/{subfolder}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_file = dest_dir / filename

    download_url_str = url
    # CivitAI presigned R2 redirect bugfix: pass token as query param, not Authorization header
    if url.startswith(("https://civitai.com/", "https://civitai.red/")):
        token = os.environ.get("CIVITAI_TOKEN", "")
        if token:
            parts = urlsplit(url)
            query = dict(parse_qsl(parts.query))
            query.setdefault("token", token)
            download_url_str = urlunsplit(parts._replace(query=urlencode(query)))

    print(f"Downloading '{filename}' from '{url}' using 16 parallel connections (axel)...")
    try:
        subprocess.run(
            ["axel", "-n", "16", "-o", str(target_file), download_url_str],
            check=True,
        )
    except Exception as e:
        print(f"Axel failed ({e}), falling back to urllib...")
        import urllib.request
        urllib.request.urlretrieve(download_url_str, target_file)

    vol_models.commit()
    print(f"Successfully saved to: {target_file}")


@app.function(
    image=image,
    volumes={"/root/models": vol_models},
    timeout=3600,
)
def download_sdxl_turbo():
    """Preset: Download SDXL Turbo (~6.9 GB) for ultra-fast, single-step generations."""
    from huggingface_hub import hf_hub_download

    dest_dir = "/root/models/checkpoints"
    os.makedirs(dest_dir, exist_ok=True)
    print("Downloading SDXL Turbo (sd_xl_turbo_1.0_fp16.safetensors)...")
    hf_hub_download(
        repo_id="stabilityai/sdxl-turbo",
        filename="sd_xl_turbo_1.0_fp16.safetensors",
        local_dir=dest_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    vol_models.commit()
    print("SDXL Turbo download complete!")
