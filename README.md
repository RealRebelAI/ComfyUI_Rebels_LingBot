# ComfyUI_Rebels_LingBot

ComfyUI custom nodes for **LingBot-Video-Dense-1.3B** (Robbyant) — text-to-video and
text+image-to-video on consumer GPUs. Built and tested on an **RTX 3070 8GB / 16GB RAM**.

The 1.3B DiT fits entirely in 8GB VRAM. The heavy piece is the Qwen3-VL text encoder
(~8GB): these nodes load it **on CPU, encode, and free it** before the DiT ever touches
the GPU, so the two never coexist.

## Nodes

| Node | What it does |
|---|---|
| **LingBot Text Encode (lazy Qwen3-VL)** | Encodes prompt + negative on CPU with the exact template/skip-layer the checkpoint was trained with, then frees the encoder. Optional `image` input for TI2V (the image is run through the pipeline's own `preprocess_image` → `_vlm_image` chain so the vision tokens match training). Embeds are cached (clone-safe). |
| **LingBot Loader (1.3B dense)** | Builds the DiT from bundled config + your safetensors, honors the model's fp32-module list (norms/modulation), loads the Wan VAE. |
| **LingBot Sampler** | Sequential CFG (the pipeline default), denoises to latents, **evicts the DiT, then decodes** with the full card. Optional `image` input switches to the TI2V pipeline. `vae_tiling` off by default (see Known Issues). |

## Install

1. Clone into `ComfyUI/custom_nodes/`.
2. `python_embeded\python.exe -m pip install -U diffusers transformers accelerate safetensors einops`
   (needs `diffusers >= 0.37` and a `transformers` recent enough for Qwen3-VL).
3. Download the weights (see the companion HF repo) into:
   - `models/diffusion_models/LingBot_1.3b_DiT.safetensors`
   - `models/vae/LingBot_vae.safetensors`
   - `models/text_encoders/LingBot_text-encoder.safetensors`
4. Restart ComfyUI.

If you are packaging from the original HF repo yourself: `merge_qwen_encoder.py` merges the
encoder shards into one file (streaming, never holds 8GB in RAM), and
`prepare_lingbot_assets.py` copies the configs + processor into `model_assets/`.

## Settings that matter

- **Resolution: 832×480.** This is the trained resolution (defaults are set to it).
  Square or other aspects go out-of-distribution: letterbox bars, broken anatomy.
- **Frames: 81 max** (widget-capped). Beyond the trained window the video collapses
  into noise.
- **Guidance: 6.0 for t2v, ~3.0 for TI2V** (the upstream ti2v script uses 3; higher
  values cause color/saturation burn on image runs).
- **Export at 24 fps** in Video Combine (upstream scripts use FPS=24).
- steps 40, shift 3.0 are the upstream defaults.

Observed speed on a 3070: ~14 s/it at 832×480×81 with sequential CFG (~9–10 min
denoise), decode well under a minute untiled with the DiT evicted.

## Known issues

- **VAE tiling can drift color/exposure over the clip.** The Wan VAE is temporally
  causal; tiled decode maintains per-tile causal caches and long clips slowly shift
  hue/brightness. Tiling is therefore **off by default** (decode fits in ~6GB once the
  DiT is evicted). Turn it on only if decode OOMs, and expect slight drift.
- **TI2V motion is modest.** The dense 1.3B supports TI2V (upstream README), but don't
  expect large prompted actions from a 1.3B — and legible rendered text (signs, labels)
  is beyond it.
- The encoder logs `missing keys: ['lm_head.weight']` — expected; it's tied to
  `embed_tokens` and re-tied at load.

## Fixed along the way (changelog highlights)

- TI2V conditioning: image now goes through the pipeline's `preprocess_image` +
  `_vlm_image` before Qwen3-VL. Raw-PIL conditioning produced near-static videos.
- Image runs use the upstream image-tuned negative prompt automatically.
- Decode no longer runs with the DiT resident (was spilling to shared memory and
  appearing to hang at step N/N).
- OOM/cancel mid-run no longer strands the DiT/VAE on the GPU (try/finally cleanup +
  start-of-run purge). Previously the *next* run would stall at decode.
- `lm_head` re-tied after meta-init streaming load.
- Frame cap 81, resolution defaults 832×480.

## Credits

- Model & pipeline: [Robbyant / lingbot-video](https://github.com/Robbyant/lingbot-video)
- Wan VAE: diffusers `AutoencoderKLWan`
- Nodes: [RealRebelAI](https://github.com/RealRebelAI)

License of the node code: MIT. Model weights are governed by the upstream
LingBot-Video license — see the source repo before redistribution or commercial use.
