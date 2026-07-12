# ComfyUI_Rebels_LingBot (30B-3B & 1.3B)

ComfyUI custom nodes for **LingBot-Video** (Robbyant) — text-to-video and text+image-to-video
on consumer GPUs. Built and tested on an **RTX 3070 8GB / 16GB RAM**.

Two model families are supported:
- **LingBot-Video-Dense-1.3B** — fits entirely in 8GB VRAM.
- **LingBot-Video-MoE-30B-A3B** — a 128-expert MoE, run from GGUF with per-expert
  streaming dequant so only routed experts ever touch VRAM. Runs on the same 8GB card.

The heavy shared piece is the Qwen3-VL text encoder (~8GB): these nodes load it **on CPU,
encode, and free it** before any DiT touches the GPU, so the two never coexist.

## The structured caption is not optional

**LingBot-Video consumes structured JSON captions, not prose.** The DiT was trained on
captions from Robbyant's prompt rewriter, where lighting, color, framing, and camera
movement are explicit fields. Feed it plain text (or ad-hoc JSON) and exposure is
unconditioned — output drifts dark and loses motion. The **LingBot Structured Prompt**
node builds the exact schema the model expects, so this is the node you start from.

## Nodes

| Node | What it does |
|---|---|
| **LingBot Structured Prompt (JSON caption)** | Builds the exact caption schema the model was trained on. Fields for scene description, **camera movement** (a first-class field — this is how you direct the camera), subject + environment elements, and a camera_info block with lighting/color/framing dropdowns using the model's own controlled vocabulary. Wire `prompt_json` into the encoder. |
| **LingBot Text Encode (lazy Qwen3-VL)** | Encodes prompt + negative on CPU with the exact template/skip-layer the checkpoint was trained with, then frees the encoder. Detects a structured caption and passes it through in the trained format. Optional `image` input for TI2V (run through the pipeline's own `preprocess_image` → `_vlm_image` chain). Embeds cached (clone-safe). |
| **LingBot Loader (1.3B dense)** | Builds the 1.3B DiT from bundled config + your safetensors, honors the model's fp32-module list (norms/modulation), loads the Wan VAE. |
| **LingBot 30B MoE Loader (GGUF)** | Loads the 30B-A3B MoE from a GGUF. Regular linears and the 128-expert fused stacks stay as quantized bytes in RAM; each is dequantized on demand per forward, one expert matrix at a time. Outputs the same model type as the 1.3B loader, so the same sampler drives it. |
| **LingBot Sampler** | Sequential CFG (the pipeline default), denoises to latents, **evicts the DiT, then decodes** with the full card. Optional `image` input switches to TI2V. `vae_tiling` off by default (see Known Issues). Drives both the 1.3B and 30B loaders unchanged. |

## Prompting: how to direct the scene and the camera

- **Scene content** goes in `scene_description` — what's in the frame, the setting, the light.
- **Camera movement** goes in `camera_movement` — e.g. "a smooth, slow push-in" or "an
  eye-level tracking shot alongside the subject." This is a trained field, so the model
  actually follows it (within the limits of the model size).
- **Lighting matters most for exposure.** Always set `lighting_type` and `lighting`. A
  night scene wants `Artificial light`; daylight wants `Daylight` or `Sunny`. Leaving
  exposure unspecified is what makes output go dark.
- Fill the dropdowns from the provided vocabulary — they map to how the model was trained,
  so arbitrary values (e.g. "50mm prime" for lens) fall out of distribution.

## Install

1. Clone into `ComfyUI/custom_nodes/`.
2. `python_embeded\python.exe -m pip install -U diffusers transformers accelerate safetensors einops gguf`
   (needs `diffusers >= 0.37` and a `transformers` recent enough for Qwen3-VL). For the
   30B loader, install **city96's ComfyUI-GGUF** next to this pack (its dequant kernels
   are reused).
3. Download the weights (see the companion HF repos) into:
   - 1.3B: `models/diffusion_models/LingBot_1.3b_DiT.safetensors`
   - 30B: `models/diffusion_models/LingBot-Video-30B-A3B-Q3_K_M.gguf` (or another tier)
   - `models/vae/LingBot_vae.safetensors`
   - `models/text_encoders/LingBot_text-encoder.safetensors`
4. **For the 30B loader only:** copy the 30B repo's `transformer/config.json` into
   `custom_nodes/ComfyUI_Rebels_LingBot/model_assets/transformer_config_30b.json`.
5. Restart ComfyUI.

If packaging from the original HF repo yourself: `merge_qwen_encoder.py` merges the encoder
shards into one file (streaming), and `prepare_lingbot_assets.py` copies configs + processor
into `model_assets/`.

## Settings that matter

- **Resolution: 832×480** (1.3B trained resolution; defaults set to it). Square or other
  aspects go out-of-distribution: letterbox bars, broken anatomy.
- **Frames: 81 max** (widget-capped). Beyond the trained window the video collapses to noise.
- **Guidance: 6.0 for t2v, ~3.0 for TI2V** (higher burns saturation on image runs).
- **Export at 24 fps** in Video Combine.
- steps 40, shift 3.0 are the upstream defaults.

**30B notes:** it's a big model streamed from disk through per-expert dequant, so the first
chunk is slow (page cache cold) — minutes, not seconds. Q3_K_M (~13GB) fits a 16GB machine's
page cache and runs at RAM speed; Q4_K_M (~17GB) suits 32GB-RAM machines. VRAM use is the
same across tiers since experts stream.

## Known issues

- **VAE tiling can drift color/exposure.** The Wan VAE is temporally causal; tiled decode
  keeps per-tile causal caches and long clips shift hue/brightness. Off by default (decode
  fits ~6GB once the DiT is evicted). Turn on only if decode OOMs.
- **Dark / low-motion output is almost always the prompt format** — use the Structured
  Prompt node and set lighting, don't feed prose.
- **TI2V / small-model limits.** Don't expect large prompted actions or legible rendered
  text from the 1.3B. The 30B is stronger but far slower on 8GB.
- The encoder logs `missing keys: ['lm_head.weight']` — expected; tied to `embed_tokens`
  and re-tied at load.

## Credits

- Model & pipeline: [Robbyant / lingbot-video](https://github.com/Robbyant/lingbot-video)
- Wan VAE: diffusers `AutoencoderKLWan`
- GGUF dequant kernels: city96 (ComfyUI-GGUF)
- Nodes & quantizations: [RealRebelAI](https://github.com/RealRebelAI)

Node code: MIT. Model weights are governed by the upstream LingBot-Video license — see the
source repo before redistribution or commercial use.
