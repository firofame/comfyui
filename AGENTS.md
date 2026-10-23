# Generic ComfyUI Model and LoRA Setup

- Keep this project generic. Read [README.md](./README.md) and the current
  interfaces before integrating another model or LoRA. Prefer existing generic
  commands and configuration over per-person or per-run Python functions,
  hardcoded training IDs, or changes to default model selection.
- When handed a successfully trained LoRA, automatically install the final weight
  in the existing persistent model volume and prepare a compatible workflow,
  unless the user asks otherwise. Derive the model variant, trigger, and settings
  from the training metadata rather than assuming the previous identity or model.
- Prepare downloads and copy weights on CPU, reuse caches, and verify the copied
  weight against its source. Preserve unrelated models, workflows, GPU choices,
  and local changes.
- Validate the workflow and LoRA key compatibility using the installed ComfyUI
  version. Do not substitute an incompatible architecture or distilled variant.
- Do not start a paid GPU server, deploy, or generate images unless requested.
  State whether validation was CPU-only and give the workflow launch instructions.
