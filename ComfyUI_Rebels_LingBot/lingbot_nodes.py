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
    files = []
    for key in ("diffusion_models", "unet"):
        try:
            files += folder_paths.get_filename_list(key)
        except Exception:
            pass
    return sorted(set(f for f in files if f.endswith((".safetensors", ".sft")))) or ["none found"]


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
        if _s.startswith("{") and '"caption"' in _s:
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
import json as _json


class LingBotStructuredPrompt:
    # controlled vocab harvested from their assets/cases/*/prompt.json
    COLOR = ["Natural", "Warm", "Cool", "Saturated", "Blue", "Cyan", "White"]
    FRAME = ["Medium", "Close Up", "Medium Close Up", "Extreme Close Up",
             "Medium Wide", "Wide"]
    ANGLE = ["Eye level", "Low angle", "High angle"]
    LENS = ["Medium", "Wide", "Long Lens", "Telephoto", "Macro",
            "Ultra Wide / Fisheye"]
    COMP = ["Balanced", "Center", "Left heavy", "Symmetrical"]
    LIGHT = ["Soft light", "Hard light", "Bright sunlight"]
    LIGHT_TYPE = ["Daylight", "Studio lighting", "Artificial light"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "description": ("STRING", {"multiline": True, "default": "",
                    "placeholder": "Plain-language scene description (becomes comprehensive_description)"}),
                # lighting first - it's the field that fixes the darkening
                "lighting_type": (s.LIGHT_TYPE,),
                "lighting": (s.LIGHT,),
                "color": (s.COLOR,),
                "frame_size": (s.FRAME,),
                "shot_type_angle": (s.ANGLE,),
                "lens_size": (s.LENS,),
                "composition": (s.COMP,),
            },
            "optional": {
                "subject_name": ("STRING", {"default": "",
                    "placeholder": "optional: main subject, e.g. 'a red fox'"}),
                "subject_action": ("STRING", {"default": "",
                    "placeholder": "optional t2v: what it does, e.g. 'trots forward, tail swishing'"}),
                "duration_s": ("INT", {"default": 0, "min": 0, "max": 20,
                    "tooltip": "0 = image/no duration; for video set the clip length in seconds"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_json",)
    FUNCTION = "build"
    CATEGORY = "Rebels/LingBot"

    def build(self, description, lighting_type, lighting, color, frame_size,
              shot_type_angle, lens_size, composition,
              subject_name="", subject_action="", duration_s=0):
        caption = {
            "comprehensive_description": description.strip(),
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
            el = {
                "name": subject_name.strip(),
                "description": description.strip()[:200],
                "location": "center",
                "relative_size": "medium",
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
            }
            if subject_action.strip():
                dur = max(duration_s, 1)
                el["actions"] = [{
                    "timestamp": "[0.0s - {:.1f}s]".format(float(dur)),
                    "action": subject_action.strip(),
                }]
            caption["prominent_elements"].append(el)

        if duration_s > 0:
            payload = {"caption": caption, "duration": duration_s}
        else:
            payload = {"caption": caption}

        out = _json.dumps(payload, ensure_ascii=False)
        print("[LingBot] structured caption: lighting={}/{}, {} element(s), duration={}".format(
            lighting_type, lighting, len(caption["prominent_elements"]), duration_s or "none"))
        return (out,)


NODE_CLASS_MAPPINGS = {
    "LingBotTextEncode": LingBotTextEncode,
    "LingBotLoader": LingBotLoader,
    "LingBotSampler": LingBotSampler,
    "LingBotStructuredPrompt": LingBotStructuredPrompt,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LingBotTextEncode": "LingBot Text Encode (lazy Qwen3-VL)",
    "LingBotLoader": "LingBot Loader (1.3B dense)",
    "LingBotSampler": "LingBot Sampler",
    "LingBotStructuredPrompt": "LingBot Structured Prompt (JSON caption)",
}
