# Workflows Directory

Any `.json` ComfyUI workflow files placed in this directory will be automatically synced into ComfyUI's user workflows directory (`/root/ComfyUI/user/default/workflows`) when the container launches.

### How to add workflows:
1. Export your workflow from the ComfyUI interface via **Save** or **Save (API Format)**.
2. Place the `.json` file in this folder.
3. The next time `modal serve main.py` or `modal deploy main.py` runs, the workflow will be accessible directly in the ComfyUI workflow browser.
