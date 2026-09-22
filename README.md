# ComfyUI on Modal

A minimal, production-grade project for running **ComfyUI** on [Modal](https://modal.com) with sub-2-second cold starts, persistent volumes, reverse-proxy fixes, accelerated multi-connection downloads, and built-in **ComfyUI-Manager**.

---

## 1. Quickstart: Launching ComfyUI Interactively

Install dependencies and start ComfyUI in development mode (spins up GPU, caches container, and provides a live HTTPS link in your terminal):

```bash
uv sync
uv run modal serve main.py
```

### What happens when you run this:
1. Modal provisions a GPU container (**NVIDIA A10G** by default).
2. Mounts two persistent volumes:
   - `comfyui-models`: Stores checkpoints, LoRAs, VAEs, text encoders, and diffusion models.
   - `comfyui-output`: Stores generated images and outputs.
3. Automatically applies the **Reverse Proxy Fix** so saving workflows in the UI never fails with HTTP 405.
4. If no checkpoint exists yet, automatically downloads the lightweight **DreamShaper 8** checkpoint (~2GB) so your default text-to-image workflow is ready right away.
5. Captures a **GPU memory snapshot** after initialization, enabling subsequent container wakeups in **~1–2 seconds**.
6. Terminal outputs a secure URL like:
   ```text
   https://<username>--comfyui-server-comfyui-ui-dev.modal.run
   ```
7. Click the URL to open the ComfyUI web UI and start generating!
8. When finished, press `Ctrl+C` in your terminal to shut down the container and stop billing.

---

## 2. Deploying Permanently (Autoscaling Web Service)

To deploy a permanent URL that stays active and automatically scales to zero when idle:

```bash
uv run modal deploy main.py
```

Modal will print your permanent production URL:
```text
https://<username>--comfyui-server-comfyui-ui.modal.run
```
Whenever you visit this URL, Modal wakes up your GPU container in seconds. After 3 minutes of inactivity, it automatically scales down to save costs.

---

## 3. Selecting Different GPUs & Snapshots

By default, the app runs on an **NVIDIA A10G** (~$1.10/hr) with GPU memory snapshots enabled:

```bash
# Budget option (NVIDIA L4, ~$0.80/hr)
MODAL_GPU="L4" uv run modal serve main.py

# High performance for large workflows (NVIDIA A100 80GB, ~$3.50/hr)
MODAL_GPU="A100-80GB" uv run modal serve main.py

# Disable memory snapshotting (standard cold boot)
MODAL_GPU_SNAPSHOT="0" uv run modal serve main.py
```

---

## 4. Downloading Models into Persistent Storage

Models can be downloaded directly into the persistent volume over Modal's high-speed datacenter connections:

### SDXL Turbo Preset
```bash
uv run modal run main.py::download_sdxl_turbo
```

### Custom Hugging Face Model
Download any `.safetensors` or `.ckpt` file into any subfolder (`checkpoints`, `loras`, `vae`, `controlnet`, `diffusion_models`):
```bash
uv run modal run main.py::download_model \
  --repo-id "runwayml/stable-diffusion-v1-5" \
  --filename "v1-5-pruned-emaonly.safetensors" \
  --subfolder "checkpoints"
```

### Custom URL / Civitai (16 Parallel Connections via `axel`)
CivitAI downloads automatically pass tokens in the query string to avoid presigned R2 redirect errors:
```bash
CIVITAI_TOKEN="your_token" uv run modal run main.py::download_url \
  --url "https://civitai.com/api/download/models/128713" \
  --filename "dreamshaper_xl.safetensors" \
  --subfolder "checkpoints"
```

Supported `--subfolder` targets:
- `checkpoints`
- `loras`
- `vae`
- `diffusion_models`
- `text_encoders`
- `controlnet`
- `clip`
- `upscale_models`

---

## 5. Workflows & Custom Nodes

- **Pre-loading Workflows**: Drop any workflow `.json` files into the `workflows/` directory. They will be copied into `/root/ComfyUI/user/default/workflows` at container startup.
- **Custom Nodes**: **ComfyUI-Manager** is pre-installed. Open the Manager button in the bottom-right corner of the ComfyUI interface to install nodes and models directly from the browser.

---

## Project Structure

```
comfyui/
├── .gitignore
├── .python-version    # 3.12
├── pyproject.toml     # uv project definition
├── README.md          # Guide & documentation
├── main.py            # Modal serverless ComfyUI app & download CLI
└── workflows/         # Saved ComfyUI workflow JSONs
    └── README.md
```
