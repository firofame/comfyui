"""ComfyUI on Modal — Qwen-Image 2.1.

    uv run modal serve main.py
"""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import modal

app = modal.App("comfyui")

models = modal.Volume.from_name("comfyui-models", create_if_missing=True)

MODEL_REPO_ID = "Comfy-Org/Qwen-Image-2.1"
# On the models volume so later runs reuse the files instead of downloading them again.
HUB_CACHE_DIR = "/root/ComfyUI/models/.cache/huggingface/hub"
# (path inside the Hugging Face repo, ComfyUI models subfolder)
QWEN_MODELS = (
    ("diffusion_models/qwen_image_2.1_bf16.safetensors", "diffusion_models"),
    ("diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "diffusion_models"),
    ("text_encoders/qwen3vl_8b_bf16.safetensors", "text_encoders"),
    ("text_encoders/qwen3vl_8b_int8_convrot.safetensors", "text_encoders"),
    ("vae/qwen_image_2.1_vae_bf16.safetensors", "vae"),
)
MODELS_DIR = "/root/ComfyUI/models"
SOURCE_DIR = "/training"
WORKFLOWS_DIR = "/opt/workflows"
INPUTS_DIR = "/opt/input"
download_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "huggingface_hub"
)


def download_models(repo_id: str, files: list[tuple[str, str]], cache_dir: str):
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)
    volume = modal.Volume.from_name("comfyui-models")
    for repo_path, subfolder in files:
        filename = os.path.basename(repo_path)
        dest_dir = f"/root/ComfyUI/models/{subfolder}"
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, filename)
        if os.path.lexists(dest):
            print(f"model already present: {dest}")
            continue

        cached = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            cache_dir=cache_dir,
        )
        os.symlink(cached, dest)
        volume.commit()
        print(dest)


def install_workflows():
    if not os.path.isdir(WORKFLOWS_DIR):
        return
    dst = "/root/ComfyUI/user/default/workflows"
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(WORKFLOWS_DIR)):
        if not name.endswith(".json"):
            continue
        shutil.copy2(os.path.join(WORKFLOWS_DIR, name), os.path.join(dst, name))
        print(f"workflow installed: {name}")


def install_inputs():
    if not os.path.isdir(INPUTS_DIR):
        return
    dst = "/root/ComfyUI/input"
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(INPUTS_DIR)):
        src = os.path.join(INPUTS_DIR, name)
        if not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(dst, name))
        print(f"input installed: {name}")


image = (
    # https://github.com/comfy-org/comfyui#manual-install-windows-linux
    # Python 3.13 is the supported version. NVIDIA stable install is:
    # pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install("huggingface_hub")
    .run_commands(
        "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI",
        "cd /root/ComfyUI && pip install -r requirements.txt",
        # Volume mounts fail if the target path already has files from the clone.
        "rm -rf /root/ComfyUI/models && mkdir -p /root/ComfyUI/models",
    )
    .run_function(
        download_models,
        volumes={"/root/ComfyUI/models": models},
        timeout=7200,
        kwargs={
            "repo_id": MODEL_REPO_ID,
            "files": list(QWEN_MODELS),
            "cache_dir": HUB_CACHE_DIR,
        },
    )
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), "workflows"),
        remote_path=WORKFLOWS_DIR,
    )
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), "input"),
        remote_path=INPUTS_DIR,
    )
)


@app.function(
    image=download_image,
    volumes={MODELS_DIR: models},
    cpu=2,
    memory=4096,
    timeout=3600,
    scaledown_window=2,
    retries=0,
)
def download_model(repo_id: str, filename: str, subfolder: str = "checkpoints"):
    if subfolder not in {
        "checkpoints", "loras", "vae", "diffusion_models", "text_encoders",
        "controlnet", "clip", "upscale_models",
    }:
        raise ValueError(f"Unsupported model subfolder: {subfolder}")
    download_models(repo_id, [(filename, subfolder)], HUB_CACHE_DIR)


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@app.function(
    image=download_image,
    volumes={MODELS_DIR: models},
    cpu=2,
    memory=4096,
    timeout=3600,
    scaledown_window=2,
    retries=0,
)
def install_lora(source_path: str, filename: str) -> str:
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("LoRA filename must be a plain filename without directories")
    source_root = Path(SOURCE_DIR).resolve()
    source = (source_root / source_path.lstrip("/")).resolve()
    if not source.is_relative_to(source_root):
        raise ValueError("LoRA source path must stay inside the source volume")
    if not source.is_file():
        raise FileNotFoundError(f"Final trained LoRA not found: {source}")

    destination_dir = Path(MODELS_DIR) / "loras"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    source_digest = file_sha256(source)
    if destination.is_file() and file_sha256(destination) == source_digest:
        print(f"LoRA already installed and verified: {destination}")
        return str(destination)

    with tempfile.NamedTemporaryFile(dir=destination_dir, delete=False) as file:
        temporary = Path(file.name)
    try:
        shutil.copy2(source, temporary)
        if file_sha256(temporary) != source_digest:
            raise RuntimeError(f"LoRA SHA-256 verification failed: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    models.commit()
    print(f"Installed and SHA-256 verified LoRA: {destination}")
    return str(destination)


@app.local_entrypoint()
def prepare_lora(source_volume: str, source_path: str, filename: str):
    install_lora.with_options(
        volumes={
            MODELS_DIR: models,
            SOURCE_DIR: modal.Volume.from_name(source_volume),
        }
    ).remote(source_path, filename)


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    scaledown_window=60,
    max_containers=1,
    volumes={"/root/ComfyUI/models": models},
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8188, startup_timeout=300)
def ui():
    install_workflows()
    install_inputs()
    subprocess.Popen(
        ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
        cwd="/root/ComfyUI",
    )
