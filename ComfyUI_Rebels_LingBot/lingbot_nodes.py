"""
ComfyUI_Rebels_LingBot - LingBot-Video dense 1.3B on consumer hardware.
by RealRebelAI

Design (per house rules):
- Dropdown-only model selection via folder_paths. No path inputs.
- Pipeline / transformer / scheduler code + configs ship in model_assets/.
- Lazy text encoder: Qwen3-VL loads on CPU, encodes, is freed BEFORE the DiT runs.
  Encoder (~8GB bf16) and transformer (2.8GB) never coexist.
- Embed cache returns fresh CLONES (SeFi lesson: cached tensors mutated in-place
  by downstream code corrupt every later run -> progressive white-out).
- Sequential CFG is the pipeline's own default (batch_cfg=False). We keep it.

Memory profile on an 8GB card:
  encode phase : Qwen3-VL on CPU (RAM), nothing on GPU
  denoise phase: 2.79GB DiT + activations on GPU, embeds only
  decode phase : Wan VAE (offload_vae_during_denoise keeps it off-GPU until needed)
"""

import gc
import os
import sys

import torch

import folder_paths

PACK_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PACK_DIR, "model_assets")

# make the vendored lingbot_video package importable
if PACK_DIR not in sys.path:
    sys.path.insert(0, PACK_DIR)


def _assets_path(*parts):
    for base in (ASSETS_DIR, PACK_DIR):
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    # scheduler config name varies; accept any json in a local scheduler folder
    if parts and parts[0] == "scheduler":
        sdir = os.path.join(PACK_DIR, "scheduler")
        if os.path.isdir(sdir):
            js = [f for f in os.listdir(sdir) if f.endswith(".json")]
            if js:
                return os.path.join(sdir, js[0])
    raise FileNotFoundError(
        "Not found in model_assets or pack folder: {}".format(os.path.join(*parts)))


# ---------------------------------------------------------------------------
# dropdown helpers
# ---------------------------------------------------------------------------
def _diffusion_model_files():
    files = set()
    # city96 registers .gguf under unet_gguf; also collect the normal lists
    for key in ("unet_gguf", "diffusion_models", "unet"):
        try:
            files.update(folder_paths.get_filename_list(key))
        except Exception:
            pass
    # walk folders directly so .gguf shows up regardless of registered extensions
    for key in ("diffusion_models", "unet"):
        try:
            bases = folder_paths.get_folder_paths(key)
        except Exception:
            continue
        for base in bases:
            if not os.path.isdir(base):
                continue
            for root, _dirs, names in os.walk(base):
                for n in names:
                    if n.endswith((".safetensors", ".sft", ".gguf")):
                        files.add(os.path.relpath(os.path.join(root, n), base))
    return sorted(files) or ["none found"]


def _vae_files():
    try:
        files = folder_paths.get_filename_list("vae")
    except Exception:
        files = []
    return sorted(f for f in files if f.endswith(".safetensors")) or ["none found"]


def _encoder_files():
    try:
        files = folder_paths.get_filename_list("text_encoders")
    except Exception:
        files = []
    return sorted(f for f in files if f.endswith(".safetensors")) or ["none found"]


def _resolve_encoder_file(name):
    p = folder_paths.get_full_path("text_encoders", name)
    if not p:
        raise FileNotFoundError("text encoder file not found: {}".format(name))
    return p


def _resolve_model_file(name):
    for key in ("diffusion_models", "unet"):
        try:
            p = folder_paths.get_full_path(key, name)
            if p:
                return p
        except Exception:
            pass
    raise FileNotFoundError(name)


def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 1) Lazy text encoder node
# ---------------------------------------------------------------------------
class LingBotTextEncode:
    """Encodes prompt + negative with Qwen3-VL on CPU, then frees the encoder.

    Cache: keyed by (folder, prompt, negative). Values stored on CPU; outputs are
    CLONES so downstream in-place ops can never corrupt the cache (SeFi bug).
    """

    _cache = {}
    _cache_order = []
    MAX_CACHE = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "encoder_name": (_encoder_files(),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "keep_encoder_loaded": (["no (free after encode)", "yes (fast re-prompts, +8GB RAM)"],
                                        {"default": "no (free after encode)"}),
            },
            "optional": {
                "image": ("IMAGE",),  # connect for i2v: Qwen3-VL sees the image while encoding
                "i2v_width": ("INT", {"default": 832, "min": 128, "max": 1280, "step": 16}),
                "i2v_height": ("INT", {"default": 480, "min": 128, "max": 1280, "step": 16}),
            },
        }

    RETURN_TYPES = ("LINGBOT_EMBEDS",)
    FUNCTION = "encode"
    CATEGORY = "Rebels/LingBot"

    _held_encoder = None
    _held_key = None

    @staticmethod
    def _to_pil(image):
        from PIL import Image as _Image
        import numpy as _np
        arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(_np.uint8)
        return _Image.fromarray(arr)

    def encode(self, encoder_name, prompt, negative_prompt, keep_encoder_loaded, image=None,
               i2v_width=832, i2v_height=480):
        from lingbot_video.pipeline_lingbot_video import (
            LingBotVideoPipeline, DEFAULT_NEGATIVE_PROMPT, DEFAULT_NEGATIVE_PROMPT_IMAGE)

        # Structured-caption JSON -> exact trained string (caption dict, runtime keys
        # stripped, compact separators). Plain prose passes through untouched. This is
        # what carries camera_info/lighting to the DiT; without it exposure drifts dark.
        import json as _json
        _s = (prompt or "").strip()
        if _s.startswith("{") and '"comprehensive_description"' in _s and '"caption"' not in _s:
            # current schema: already the final trained string - pass through, just log
            print("[LingBot] structured caption (direct format) -> DiT ({} chars)".format(len(_s)))
            prompt = _s
        elif _s.startswith("{") and '"caption"' in _s:
            try:
                _obj = _json.loads(_s)
                _cap = _obj.get("caption", _obj)
                if isinstance(_cap, dict):
                    _rt = {"duration", "fps", "height", "width", "num_frames", "resolution", "ratio"}
                    _cap = {k: v for k, v in _cap.items() if k not in _rt}
                prompt = _json.dumps(_cap, ensure_ascii=False, separators=(",", ":"))
                print("[LingBot] structured caption normalized ({} chars) -> DiT".format(len(prompt)))
            except Exception as _e:
                print("[LingBot] caption JSON parse failed, using raw text: {}".format(_e))

        # image runs use the image-tuned negative (their default_negative_prompt_image)
        neg = negative_prompt.strip() or (
            DEFAULT_NEGATIVE_PROMPT_IMAGE if image is not None else DEFAULT_NEGATIVE_PROMPT)
        img_sig = None
        pil = None
        if image is not None:
            pil = self._to_pil(image)
            import hashlib
            img_sig = hashlib.md5(pil.tobytes()).hexdigest()[:12]
        key = (encoder_name, prompt, neg, img_sig, i2v_width, i2v_height)

        hit = LingBotTextEncode._cache.get(key)
        if hit is not None:
            pe, pm, ne, nm = hit
            print("[LingBot] embeds cache hit")
            return ({"prompt_embeds": pe.clone(), "prompt_mask": pm.clone(),
                     "negative_embeds": ne.clone(), "negative_mask": nm.clone()},)

        from transformers import AutoProcessor, AutoConfig

        weight_file = _resolve_encoder_file(encoder_name)
        keep = keep_encoder_loaded.startswith("yes")

        if LingBotTextEncode._held_encoder is not None and LingBotTextEncode._held_key == weight_file:
            text_encoder, processor = LingBotTextEncode._held_encoder
            print("[LingBot] reusing held encoder")
        else:
            if LingBotTextEncode._held_encoder is not None:
                _free(LingBotTextEncode._held_encoder)
                LingBotTextEncode._held_encoder = None
            print("[LingBot] loading Qwen3-VL encoder on CPU from {}".format(weight_file))
            # ALL configs live in model_assets; the models/text_encoders file is weights only.
            processor = AutoProcessor.from_pretrained(
                _assets_path("processor"), trust_remote_code=True)
            from transformers import Qwen3VLForConditionalGeneration
            from accelerate import init_empty_weights
            from safetensors import safe_open

            cfg = AutoConfig.from_pretrained(
                _assets_path("text_encoder"), trust_remote_code=True)
            with init_empty_weights():
                text_encoder = Qwen3VLForConditionalGeneration(cfg)
            # streaming assign from the single merged file: peak RAM ~ one copy of the
            # model (8GB), never model+state_dict (16GB) - the LongCat safe_open trick.
            sd = {}
            with safe_open(weight_file, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k).to(torch.bfloat16)
            missing, unexpected = text_encoder.load_state_dict(sd, strict=False, assign=True)
            del sd
            if missing:
                print("[LingBot] encoder missing keys ({}): {}".format(len(missing), missing[:4]))
            if unexpected:
                print("[LingBot] encoder unexpected keys ({}): {}".format(len(unexpected), unexpected[:4]))
            # lm_head is tied to embed_tokens (tie_word_embeddings) so it is absent from
            # the checkpoint; re-establish the tie after the assign-load or it stays meta.
            try:
                text_encoder.tie_weights()
            except Exception as e:
                print("[LingBot] tie_weights failed: {}".format(e))
            meta = [n for n, p in text_encoder.named_parameters() if p.device.type == "meta"]
            if meta:
                print("[LingBot] WARNING still-meta encoder params ({}): {}".format(len(meta), meta[:4]))
            text_encoder = text_encoder.eval()
            text_encoder.requires_grad_(False)

        # a pipeline with ONLY encoder+processor: __init__ tolerates None modules.
        # For image runs we need the I2V class: its preprocess_image + _vlm_image produce
        # the exact vision-token layout the checkpoint was trained with. Feeding a raw
        # PIL at native size gives a different token grid -> conditioning collapses
        # (the static-video failure).
        if pil is not None:
            from lingbot_video.pipeline_lingbot_video_i2v import LingBotVideoImageToVideoPipeline
            pipe = LingBotVideoImageToVideoPipeline(
                transformer=None, vae=None,
                text_encoder=text_encoder, processor=processor, scheduler=None)
        else:
            pipe = LingBotVideoPipeline(
                transformer=None, vae=None,
                text_encoder=text_encoder, processor=processor, scheduler=None)

        with torch.inference_mode():
            if pil is not None:
                pixel = pipe.preprocess_image(pil, i2v_height, i2v_width).to(torch.float32)
                vlm_image = pipe._vlm_image(pixel)
                pe, pm = pipe.encode_prompt(prompt, images=[vlm_image], device="cpu")
            else:
                pe, pm = pipe.encode_prompt(prompt, device="cpu")
            ne, nm = pipe.encode_prompt(neg, device="cpu")

        del pipe
        if keep:
            LingBotTextEncode._held_encoder = (text_encoder, processor)
            LingBotTextEncode._held_key = weight_file
        else:
            _free(text_encoder, processor)
            print("[LingBot] encoder freed")

        entry = (pe.to(torch.bfloat16).cpu(), pm.cpu(),
                 ne.to(torch.bfloat16).cpu(), nm.cpu())
        LingBotTextEncode._cache[key] = entry
        LingBotTextEncode._cache_order.append(key)
        while len(LingBotTextEncode._cache_order) > LingBotTextEncode.MAX_CACHE:
            old = LingBotTextEncode._cache_order.pop(0)
            LingBotTextEncode._cache.pop(old, None)

        pe, pm, ne, nm = entry
        return ({"prompt_embeds": pe.clone(), "prompt_mask": pm.clone(),
                 "negative_embeds": ne.clone(), "negative_mask": nm.clone()},)


# ---------------------------------------------------------------------------
# 2) Loader node (transformer + vae + scheduler)
# ---------------------------------------------------------------------------
class LingBotLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transformer_name": (_diffusion_model_files(),),
                "vae_name": (_vae_files(),),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["bf16", "fp16"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("LINGBOT_MODEL",)
    FUNCTION = "load"
    CATEGORY = "Rebels/LingBot"

    def load(self, transformer_name, vae_name, device, dtype):
        import json
        from safetensors.torch import load_file
        from lingbot_video.transformer_lingbot_video import (
            LingBotVideoTransformer3DModel, should_keep_in_fp32)
        from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
        from diffusers import AutoencoderKLWan

        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

        # transformer: config from model_assets, weights from dropdown
        with open(_assets_path("transformer", "config.json")) as f:
            tcfg = json.load(f)
        tcfg = {k: v for k, v in tcfg.items() if not k.startswith("_")}
        print("[LingBot] building LingBotVideoTransformer3DModel (dense 1.3B)")
        transformer = LingBotVideoTransformer3DModel(**tcfg)
        sd = load_file(_resolve_model_file(transformer_name))
        missing, unexpected = transformer.load_state_dict(sd, strict=False)
        if missing:
            print("[LingBot] transformer missing keys ({}): {}".format(len(missing), missing[:5]))
        if unexpected:
            print("[LingBot] transformer unexpected keys ({}): {}".format(len(unexpected), unexpected[:5]))
        del sd
        # mixed precision: norms/modulation stay fp32 per LINGBOT_VIDEO_FP32_MODULES
        transformer = transformer.to(torch_dtype)
        for name, p in transformer.named_parameters():
            if should_keep_in_fp32(name):
                p.data = p.data.to(torch.float32)
        transformer.eval().requires_grad_(False)

        # vae: stock diffusers AutoencoderKLWan, config from model_assets
        with open(_assets_path("vae", "config.json")) as f:
            vcfg = json.load(f)
        vcfg = {k: v for k, v in vcfg.items() if not k.startswith("_")}
        vae = AutoencoderKLWan(**vcfg)
        vsd = load_file(folder_paths.get_full_path("vae", vae_name))
        vae.load_state_dict(vsd, strict=True)
        del vsd
        vae = vae.to(torch.float32).eval()
        vae.requires_grad_(False)
        # tiling is decided per-run in the sampler (vae_tiling widget). The Wan VAE is
        # temporally causal; tiled decode keeps per-tile causal caches and produces slow
        # global color/exposure drift across frames. With the DiT evicted before decode,
        # untiled 832x480x81 fits in ~7GB, so untiled is the quality default.

        with open(_assets_path("scheduler", "scheduler_config.json")) as f:
            scfg = json.load(f)
        scfg = {k: v for k, v in scfg.items() if not k.startswith("_")}
        scheduler = FlowUniPCMultistepScheduler(**scfg)

        gc.collect()
        return ({"transformer": transformer, "vae": vae, "scheduler": scheduler,
                 "device": device, "dtype": torch_dtype},)


# ---------------------------------------------------------------------------
# 3) Sampler node
# ---------------------------------------------------------------------------
class LingBotSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lingbot_model": ("LINGBOT_MODEL",),
                "embeds": ("LINGBOT_EMBEDS",),
                # 832x480 is the checkpoint's trained resolution (repo README). Square or
                # other aspects go out-of-distribution: letterbox bars, broken anatomy.
                "width": ("INT", {"default": 832, "min": 128, "max": 1280, "step": 16}),
                "height": ("INT", {"default": 480, "min": 128, "max": 1280, "step": 16}),
                # trained window is 81 frames; beyond it the temporal RoPE collapses
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 81, "step": 4}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 12.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "offload_vae": (["yes (low VRAM)", "no"], {"default": "yes (low VRAM)"}),
                "vae_tiling": (["off (stable colors, needs ~6GB free at decode)",
                                "on (lowest VRAM, can drift color/exposure)"],
                               {"default": "off (stable colors, needs ~6GB free at decode)"}),
            },
            "optional": {
                "image": ("IMAGE",),  # connect a start frame for i2v; leave empty for t2v
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Rebels/LingBot"

    def sample(self, lingbot_model, embeds, width, height, num_frames, steps,
               guidance, shift, seed, offload_vae, vae_tiling="off", image=None):
        from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
        from lingbot_video.pipeline_lingbot_video_i2v import LingBotVideoImageToVideoPipeline

        # pipeline invariants, enforced up front with readable errors
        if num_frames != 1 and (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
            print("[LingBot] num_frames adjusted to {} (must be 1 or 4n+1)".format(num_frames))
        width -= width % 16
        height -= height % 16

        # purge residue from any previous (possibly OOM'd/cancelled) run BEFORE starting
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = lingbot_model["device"]
        dtype = lingbot_model["dtype"]
        transformer = lingbot_model["transformer"].to(device)
        vae = lingbot_model["vae"]
        tiled = vae_tiling.startswith("on")
        try:
            if tiled:
                vae.enable_tiling()
                print("[LingBot] vae tiling ON (low VRAM; slight temporal drift possible)")
            elif hasattr(vae, "disable_tiling"):
                vae.disable_tiling()
        except Exception:
            pass
        if not offload_vae.startswith("yes"):
            vae = vae.to(device)

        i2v = image is not None
        if i2v:
            print("[LingBot] WARNING: this dense-1.3B checkpoint is declared t2v "
                  "(model_index.json). i2v runs will track the start frame but produce "
                  "little/no motion. Provided for experimentation only.")
        pipe_cls = LingBotVideoImageToVideoPipeline if i2v else LingBotVideoPipeline
        pipe = pipe_cls(
            transformer=transformer, vae=vae,
            text_encoder=None, processor=None,
            scheduler=lingbot_model["scheduler"])

        gen = torch.Generator(device="cpu").manual_seed(seed & 0xFFFFFFFFFFFFFFFF)

        pe = embeds["prompt_embeds"].to(device, dtype)
        pm = embeds["prompt_mask"].to(device)
        ne = embeds["negative_embeds"].to(device, dtype)
        nm = embeds["negative_mask"].to(device)

        print("[LingBot] {} {}x{} frames={} steps={} cfg={} shift={} (sequential CFG)".format(
            "i2v" if i2v else "t2v", width, height, num_frames, steps, guidance, shift))

        try:
            kwargs = dict(
                prompt=None,
                height=height, width=width, num_frames=num_frames,
                num_inference_steps=steps, guidance_scale=guidance, shift=shift,
                generator=gen,
                prompt_embeds=pe, prompt_mask=pm,
                negative_prompt_embeds=ne, negative_prompt_mask=nm,
                batch_cfg=False,
                # get latents back, evict the DiT, THEN decode with the whole card.
                # Decoding inside pipe() keeps the 2.8GB transformer resident and the
                # Wan VAE spills to shared memory (the stuck-at-decode symptom).
                output_type="latent",
            )
            if i2v:
                from PIL import Image as _Image
                import numpy as _np
                arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(_np.uint8)
                kwargs["image"] = _Image.fromarray(arr).resize((width, height))
                # i2v needs the VAE up front to encode the start frame
                vae.to(device)

            with torch.inference_mode():
                out = pipe(**kwargs)

            latents = out.frames if hasattr(out, "frames") else out[0]

            # ---- decode phase: DiT off, VAE on, full VRAM ----
            print("[LingBot] denoise done; evicting DiT before decode")
            transformer.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Decode ladder: never lose a finished denoise to a decode OOM.
            #   1. GPU untiled (best quality, no drift)  [skipped if user forced tiling]
            #   2. GPU tiled   (low VRAM; slight temporal drift possible)
            #   3. CPU         (slow, guaranteed)
            def _try_decode(dev, tile):
                try:
                    if tile:
                        vae.enable_tiling()
                    elif hasattr(vae, "disable_tiling"):
                        vae.disable_tiling()
                except Exception:
                    pass
                vae.to(dev)
                with torch.inference_mode():
                    return pipe._decode_latents(latents.to(dev))

            attempts = ([("cuda", True)] if tiled else [("cuda", False), ("cuda", True)])
            attempts += [("cpu", False)]
            frames = None
            for dev, tile in attempts:
                if dev == "cuda" and not torch.cuda.is_available():
                    continue
                try:
                    print("[LingBot] decode attempt: {} {}".format(dev, "tiled" if tile else "untiled"))
                    frames = _try_decode(dev, tile)
                    break
                except torch.OutOfMemoryError:
                    print("[LingBot] decode OOM on {} {}; falling back".format(
                        dev, "tiled" if tile else "untiled"))
                    vae.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()
            if frames is None:
                raise RuntimeError("decode failed on every backend")
            vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import numpy as np
            arr = np.asarray(frames[0] if isinstance(frames, (list, tuple)) else frames)
            # normalize to [F,H,W,C] float32 0..1 for ComfyUI IMAGE
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            if arr.ndim == 5:  # [B,F,H,W,C]
                arr = arr[0]
            if arr.shape[-1] not in (1, 3, 4) and arr.shape[1] in (1, 3, 4):
                arr = np.moveaxis(arr, 1, -1)  # [F,C,H,W] -> [F,H,W,C]
            images = torch.from_numpy(np.ascontiguousarray(arr)).float().clamp(0, 1)

        finally:
            # OOM/cancel can fire anywhere above; without this the DiT (and
            # in i2v the VAE) stay parked on the GPU and the NEXT run stalls
            # at decode on a half-full card.
            try:
                transformer.to("cpu")
            except Exception:
                pass
            try:
                vae.to("cpu")
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _free(pipe)
        return (images,)

    # note: decode happens inside pipe(); tiling (loader) keeps its footprint small




# ---------------------------------------------------------------------------
# Structured prompt builder - emits the exact caption JSON the DiT was trained
# on. The model consumes structured captions, NOT prose; a missing camera_info
# (esp. lighting/lighting_type) leaves exposure unconditioned and output drifts
# dark. This node guarantees those fields are present.
# ---------------------------------------------------------------------------
# Structured prompt builder - emits the EXACT caption schema this checkpoint
# expects (community/rewriter reference format):
#   - comprehensive_description is an OBJECT: scene_content_description +
#     camera_movement_description  (camera moves are a first-class field!)
#   - two prominent_elements (subject + environment), full key set incl.
#     number_of_objects
#   - camera_info with the rewriter vocabulary (lighting_type incl. "Sunny")
#   - NO top-level "caption" wrapper, compact JSON
# ---------------------------------------------------------------------------
import json as _json


class LingBotStructuredPrompt:
    COLOR = ["Natural", "Warm", "Cool", "Saturated", "Blue", "Cyan", "White"]
    FRAME = ["Medium", "Medium Wide", "Wide", "Close Up", "Medium Close Up",
             "Extreme Close Up"]
    ANGLE = ["Eye level", "Low angle", "High angle"]
    LENS = ["Medium", "Wide", "Long Lens", "Telephoto", "Macro",
            "Ultra Wide / Fisheye"]
    COMP = ["Center", "Balanced", "Left heavy", "Symmetrical"]
    LIGHT = ["Soft light", "Hard light", "Bright sunlight"]
    LIGHT_TYPE = ["Sunny", "Daylight", "Studio lighting", "Artificial light"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scene_description": ("STRING", {"multiline": True, "default": "",
                    "placeholder": "What the scene looks like (subjects, setting, light, style)"}),
                "camera_movement": ("STRING", {"multiline": True, "default": "",
                    "placeholder": "Camera motion, e.g. 'A smooth eye-level tracking shot moves alongside...'"}),
                "subject_name": ("STRING", {"default": ""}),
                "subject_description": ("STRING", {"multiline": True, "default": "",
                    "placeholder": "Appearance of the main subject"}),
                "subject_action": ("STRING", {"multiline": True, "default": "",
                    "placeholder": "What the subject does over the clip"}),
                "environment_name": ("STRING", {"default": ""}),
                "environment_description": ("STRING", {"multiline": True, "default": "",
                    "placeholder": "The setting/background as its own element"}),
                "duration_s": ("INT", {"default": 5, "min": 1, "max": 20}),
                "lighting_type": (s.LIGHT_TYPE,),
                "lighting": (s.LIGHT,),
                "color": (s.COLOR,),
                "frame_size": (s.FRAME,),
                "shot_type_angle": (s.ANGLE,),
                "lens_size": (s.LENS,),
                "composition": (s.COMP,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_json",)
    FUNCTION = "build"
    CATEGORY = "Rebels/LingBot"

    @staticmethod
    def _element(name, desc, action, dur, location, size, extra=None):
        # key order transcribed from a confirmed-working caption:
        # name, description, location, relative_size, shape_and_color, texture,
        # appearance_details, relationship, orientation, pose, expression,
        # clothing, gender, skin_tone_and_texture, actions (LAST).
        # No number_of_objects. actions timestamps like "[0.0s - 5.0s]".
        el = {
            "name": name,
            "description": desc,
            "location": location,
            "relative_size": size,
            "shape_and_color": "",
            "texture": "",
            "appearance_details": "",
            "relationship": "",
            "orientation": "",
            "pose": "",
            "expression": "",
            "clothing": "",
            "gender": "",
            "skin_tone_and_texture": "",
            "actions": [{"timestamp": "[0.0s - {:.1f}s]".format(float(dur)),
                         "action": action}],
        }
        if extra:
            el.update(extra)
        return el

    def build(self, scene_description, camera_movement, subject_name,
              subject_description, subject_action, environment_name,
              environment_description, duration_s, lighting_type, lighting,
              color, frame_size, shot_type_angle, lens_size, composition):
        # EXACT working format (verified against a known-good caption):
        # {"caption": {comprehensive_description: <flat string incl. camera move>,
        #              camera_info: {...}, world_knowledge: [],
        #              prominent_elements: [...]}, "duration": N}
        scene = scene_description.strip().rstrip(".")
        cam = camera_movement.strip()
        desc = scene + "." if scene else ""
        if cam:
            desc = (desc + " " + cam).strip()
            if not desc.endswith("."):
                desc += "."
        caption = {
            "comprehensive_description": desc,
            "camera_info": {
                "color": color,
                "frame_size": frame_size,
                "shot_type_angle": shot_type_angle,
                "lens_size": lens_size,
                "composition": composition,
                "lighting": lighting,
                "lighting_type": lighting_type,
            },
            "world_knowledge": [],
            "prominent_elements": [],
        }
        if subject_name.strip():
            caption["prominent_elements"].append(self._element(
                subject_name.strip(), subject_description.strip(),
                subject_action.strip(), duration_s, "center", "medium"))
        if environment_name.strip():
            caption["prominent_elements"].append(self._element(
                environment_name.strip(), environment_description.strip(),
                "", duration_s, "background", "large"))
        payload = {"caption": caption, "duration": int(duration_s)}

        out = _json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        print("[LingBot] structured caption (caption-wrapped): {} chars, "
              "{} element(s), lighting={}/{}".format(
                  len(out), len(caption["prominent_elements"]),
                  lighting_type, lighting))
        return (out,)




# ---------------------------------------------------------------------------
# 30B-A3B MoE: GGUF loader. Produces the same LINGBOT_MODEL dict as the 1.3B
# loader, so the existing LingBotSampler drives it unchanged.
#
# Requirements:
#  - GGUF made with the expert-splitting converter (tensors like
#    blocks.N.ffn.experts.w1.<e> as 2D) in models/diffusion_models
#  - model_assets/transformer_config_30b.json  (the 30B repo's transformer
#    config.json, copied + renamed)
#  - city96 ComfyUI-GGUF installed next to this pack (dequant kernels)
#
# Expert forward uses the vendored pure-torch for-loop path
# (LINGBOT_MOE_EXPERT_BACKEND=for_loop): it indexes experts.w1[idx], which we
# serve with an on-demand per-expert dequant - only ROUTED experts are ever
# dequantized, and never more than one expert matrix resides in VRAM at once.
# ---------------------------------------------------------------------------
_DQ30 = {"mod": None}


def _dequant30(qbytes, qtype, oshape, dtype):
    if _DQ30["mod"] is None:
        import importlib.util
        for name in ("ComfyUI-GGUF", "ComfyUI-GGUF-main", "gguf"):
            cand = os.path.join(os.path.dirname(PACK_DIR), name, "dequant.py")
            if os.path.isfile(cand):
                spec = importlib.util.spec_from_file_location("rebels_lb30_dequant", cand)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _DQ30["mod"] = mod
                print("[LingBot30B] dequant kernels loaded from {}".format(cand))
                break
        if _DQ30["mod"] is None:
            raise ImportError("ComfyUI-GGUF (city96) not found next to this pack")
    return _DQ30["mod"].dequantize(qbytes, qtype, oshape, dtype=dtype)


class _GGUFLinear30(torch.nn.Module):
    """Quantized bytes stay CPU (plain attrs); dequant per forward."""

    def __init__(self, qdata, qtype, oshape, bias=None):
        super().__init__()
        object.__setattr__(self, "qdata", qdata)
        self.qtype = qtype
        self.oshape = tuple(int(x) for x in oshape)
        # register_buffer BEFORE any plain self.bias assignment - assigning
        # self.bias=None first makes it a plain attr and register_buffer then
        # errors "attribute 'bias' already exists".
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.register_buffer("bias", None, persistent=False)

    def forward(self, x):
        q = self.qdata.to(x.device, non_blocking=True) if self.qdata.device != x.device else self.qdata
        w = _dequant30(q, self.qtype, self.oshape, x.dtype)
        out = torch.nn.functional.linear(x, w, self.bias.to(x.device, x.dtype) if self.bias is not None else None)
        del w
        return out

    @property
    def weight(self):
        # The model introspects .weight.dtype/.device/.shape (e.g.
        # attn.to_q.weight.dtype). Return a 1-element strided view that reports the
        # right dtype, device, and shape without allocating the full matrix.
        dev = self.qdata.device
        base = torch.empty(1, dtype=torch.bfloat16, device=dev)
        return base.as_strided(self.oshape, [0] * len(self.oshape))


class _GGUFExpertStack:
    """Holds E per-expert quantized 2D matrices. Indexing dequants ONE expert's
    2D matrix on demand (~3MB) - matches _run_experts_for_loop's w[idx] access.
    Only routed experts are ever dequantized; one small matrix at a time."""

    def __init__(self, per_expert, device, dtype):
        self._e = per_expert  # list of (qdata_cpu, qtype, oshape2d), ordered by expert idx
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self._e)

    def __getitem__(self, idx):
        q, t, sh = self._e[int(idx)]
        q = q.to(self.device, non_blocking=True) if q.device != self.device else q
        return _dequant30(q, t, sh, self.dtype)


def _force_expert_for_loop():
    """Make _run_grouped_experts always use the per-expert for-loop sub-path.

    The model's _run_grouped_experts uses torch._grouped_mm when available, which
    calls experts.w1.bfloat16() on the whole fused stack. Our _GGUFExpertStack
    holds per-expert quantized matrices and only supports w1[idx] indexing, so we
    redirect _run_grouped_experts to the module's own _run_experts_for_loop, which
    accesses experts one at a time (matching our dequant-on-demand). Idempotent.
    """
    from lingbot_video import transformer_lingbot_video as T
    cls = None
    for name in ("LingBotVideoSparseMoeBlock", "LingBotVideoGroupedExperts"):
        c = getattr(T, name, None)
        if c is not None and hasattr(c, "_run_grouped_experts") and hasattr(c, "_run_experts_for_loop"):
            cls = c
            break
    if cls is None:
        print("[LingBot30B] WARN: could not find grouped-experts class to patch; "
              "relying on env backend only")
        return
    if getattr(cls, "_rebels_forloop_patched", False):
        return
    cls._run_grouped_experts = cls._run_experts_for_loop
    cls._rebels_forloop_patched = True
    print("[LingBot30B] expert dispatch forced to per-expert for-loop "
          "(dequant-on-demand, no full-stack materialize)")


class LingBotMoELoaderGGUF:
    # one loaded 30B at a time. Rebuilding while the previous model is alive
    # doubles RAM, evicts the GGUF from page cache, and steps crawl (~600s/it).
    _CACHE = {"key": None, "out": None}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit_gguf": ([f for f in _diffusion_model_files() if f.endswith(".gguf")] or ["none found"],),
                "vae_name": (_vae_files(),),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("LINGBOT_MODEL",)
    FUNCTION = "load"
    CATEGORY = "Rebels/LingBot"

    def load(self, dit_gguf, vae_name, device):
        # reuse the already-built model when inputs are unchanged: keeps the
        # GGUF pages hot in cache across runs instead of rebuilding
        key = (dit_gguf, vae_name, device)
        if LingBotMoELoaderGGUF._CACHE["key"] == key and LingBotMoELoaderGGUF._CACHE["out"] is not None:
            print("[LingBot30B] reusing loaded model (cache hit)")
            return (LingBotMoELoaderGGUF._CACHE["out"],)
        # different inputs: tear down the previous model BEFORE building the new
        # one, so two 30B structures never coexist in RAM
        if LingBotMoELoaderGGUF._CACHE["out"] is not None:
            print("[LingBot30B] releasing previous model before rebuild")
            LingBotMoELoaderGGUF._CACHE["out"] = None
            LingBotMoELoaderGGUF._CACHE["key"] = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            import comfy.model_management as _mm
            _mm.unload_all_models()
        except Exception:
            pass
        return self._load_impl(dit_gguf, vae_name, device, key)

    def _load_impl(self, dit_gguf, vae_name, device, cache_key):
        import json
        import gguf as ggufpkg
        from collections import defaultdict
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel
        from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
        from diffusers import AutoencoderKLWan

        # The model only accepts grouped_mm / sglang_triton / sglang_triton_fp8.
        # grouped_mm is pure-torch (no triton/sglang, works on Windows) and is the
        # right choice. BUT _run_grouped_experts takes the torch._grouped_mm fast
        # path when it exists (torch 2.10 has it), which does experts.w1.bfloat16()
        # on the WHOLE fused stack - our _GGUFExpertStack can't do that and it would
        # materialize all 128 experts. So we force the module's own for-loop
        # sub-path, which accesses experts.w1[idx] one at a time = our dequant path.
        os.environ["LINGBOT_MOE_EXPERT_BACKEND"] = "grouped_mm"
        _force_expert_for_loop()

        with open(_assets_path("transformer_config_30b.json")) as f:
            tcfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        print("[LingBot30B] building MoE skeleton ({} experts)".format(tcfg.get("num_experts")))
        with init_empty_weights():
            transformer = LingBotVideoTransformer3DModel(**tcfg)

        path = None
        for key in ("unet_gguf", "diffusion_models", "unet"):
            try:
                path = folder_paths.get_full_path(key, dit_gguf) or path
            except Exception:
                pass
        if path is None:
            for key in ("diffusion_models", "unet"):
                for base in folder_paths.get_folder_paths(key):
                    p = os.path.join(base, dit_gguf)
                    if os.path.isfile(p):
                        path = p
        if path is None:
            raise FileNotFoundError(dit_gguf)
        print("[LingBot30B] loading GGUF: {}".format(path))
        reader = ggufpkg.GGUFReader(path)

        F16 = ggufpkg.GGMLQuantizationType.F16
        F32 = ggufpkg.GGMLQuantizationType.F32
        params = dict(transformer.named_parameters())
        params.update(dict(transformer.named_buffers()))
        modules = dict(transformer.named_modules())
        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        from collections import defaultdict as _dd
        experts = _dd(dict)  # "blocks.N.ffn.experts.wX" -> {e: (qdata, qtype, oshape2d)}
        attached, unmatched = 0, []
        import re
        # split 2D experts named ...experts.w1.<e>.weight
        exp_re = re.compile(r"^(.*\.experts\.w[123])\.(\d+)\.weight$")

        for t in reader.tensors:
            name = str(t.name)
            oshape = [int(d) for d in reversed(t.shape)]
            data = torch.from_numpy(t.data)  # memmap-backed, no RAM copy
            m = exp_re.match(name)
            if m:
                experts[m.group(1)][int(m.group(2))] = (data, t.tensor_type, oshape)
                attached += 1
                continue
            if name not in params:
                unmatched.append(name)
                continue
            mod_name, _, pname = name.rpartition(".")
            parent = modules.get(mod_name)
            if t.tensor_type in (F16, F32):
                w = data.view(torch.float16 if t.tensor_type == F16 else torch.float32)
                w = w.reshape(oshape).to(torch.bfloat16 if t.tensor_type == F16 else torch.float32)
                if pname in dict(parent.named_parameters(recurse=False)):
                    setattr(parent, pname, torch.nn.Parameter(w, requires_grad=False))
                else:
                    parent._buffers[pname] = w
            elif isinstance(parent, torch.nn.Linear) and pname == "weight":
                gp_name, _, leaf = mod_name.rpartition(".")
                gp = modules.get(gp_name, transformer)
                bias = None
                if parent.bias is not None and parent.bias.device.type != "meta":
                    bias = parent.bias.detach().to(torch.bfloat16)
                setattr(gp, leaf, _GGUFLinear30(data, t.tensor_type, oshape, bias))
            else:
                w = _dequant30(data, t.tensor_type, oshape, torch.bfloat16)
                if pname in dict(parent.named_parameters(recurse=False)):
                    setattr(parent, pname, torch.nn.Parameter(w, requires_grad=False))
                else:
                    parent._buffers[pname] = w
            attached += 1

        # install expert stacks: replace fused meta Parameters (w1/w2/w3) with
        # indexable dequant-on-demand objects built from the split 2D experts
        n_stacks = 0
        for full, emap in experts.items():
            mod_name, _, wname = full.rpartition(".")
            parent = modules.get(mod_name)
            if parent is None:
                unmatched.append(full)
                continue
            ordered = [emap[i] for i in sorted(emap)]
            parent._parameters.pop(wname, None)
            object.__setattr__(parent, wname,
                               _GGUFExpertStack(ordered, dev, torch.bfloat16))
            n_stacks += 1

        meta_left = [n for n, p in transformer.named_parameters() if p.device.type == "meta"]
        print("[LingBot30B] attached {} ({} expert stacks rebuilt); unmatched {}; "
              "still-meta {}".format(attached, n_stacks, len(unmatched), len(meta_left)))
        if unmatched[:5]:
            print("[LingBot30B]   unmatched sample: {}".format(unmatched[:5]))
        if meta_left[:5]:
            print("[LingBot30B]   meta sample: {}".format(meta_left[:5]))
        transformer.eval().requires_grad_(False)

        # vae + scheduler exactly like the 1.3B loader
        with open(_assets_path("vae", "config.json")) as f:
            vcfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        vae = AutoencoderKLWan(**vcfg)
        vsd = load_file(folder_paths.get_full_path("vae", vae_name))
        vae.load_state_dict(vsd, strict=True)
        del vsd
        vae = vae.to(torch.float32).eval()
        vae.requires_grad_(False)
        with open(_assets_path("scheduler", "scheduler_config.json")) as f:
            scfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        scheduler = FlowUniPCMultistepScheduler(**scfg)

        gc.collect()
        out = {"transformer": transformer, "vae": vae, "scheduler": scheduler,
               "device": device, "dtype": torch.bfloat16}
        LingBotMoELoaderGGUF._CACHE["key"] = cache_key
        LingBotMoELoaderGGUF._CACHE["out"] = out
        return (out,)

NODE_CLASS_MAPPINGS = {
    "LingBotTextEncode": LingBotTextEncode,
    "LingBotLoader": LingBotLoader,
    "LingBotSampler": LingBotSampler,
    "LingBotStructuredPrompt": LingBotStructuredPrompt,
    "LingBotMoELoaderGGUF": LingBotMoELoaderGGUF,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LingBotTextEncode": "LingBot Text Encode (lazy Qwen3-VL)",
    "LingBotLoader": "LingBot Loader (1.3B dense)",
    "LingBotSampler": "LingBot Sampler",
    "LingBotStructuredPrompt": "LingBot Structured Prompt (JSON caption)",
    "LingBotMoELoaderGGUF": "LingBot 30B MoE Loader (GGUF)",
}
