from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from decision_calibration import (
    ATYPIA_BOUNDARY_MARGIN,
    DEFAULT_ATYPIA_THRESHOLD,
    PROB_COLS,
    REVIEW_MARGIN_THRESHOLD,
    atypia_threshold_predictions,
)


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "work"
CLASS_NAMES = ["Benign", "Atypia", "Malignant"]
DEFAULT_MULTI_SCALE_SIZES = [224, 448, 896]
EXPECTED_MODELS = 15


def select_device(preferred: str = "auto") -> str:
    preferred = (preferred or "auto").lower()
    if preferred in ("mps", "apple"):
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    if preferred in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
    if preferred == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_device_cache(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def extract_patches_from_roi(
    image: Image.Image,
    target_sizes: list[int] | None = None,
    target_mpp: float = 0.5,
    max_patches_per_scale: int = 64,
) -> dict[int, list[np.ndarray]]:
    target_sizes = target_sizes or DEFAULT_MULTI_SCALE_SIZES
    img = np.asarray(image.convert("RGB"))
    h, w = img.shape[:2]
    if w > 2000 or h > 2000:
        scale = target_mpp / 0.25
        new_w, new_h = max(1, int(w // scale)), max(1, int(h // scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tissue_mask = (gray > 20).astype(np.float32)
    has_tissue = tissue_mask.mean() > 0.1
    result: dict[int, list[np.ndarray]] = {}
    for sz in target_sizes:
        patches: list[np.ndarray] = []
        if h >= sz and w >= sz:
            coords = []
            for y in range(0, h - sz + 1, sz):
                for x in range(0, w - sz + 1, sz):
                    if tissue_mask[y : y + sz, x : x + sz].mean() > 0.1:
                        coords.append((x, y))
            if max_patches_per_scale > 0 and len(coords) > max_patches_per_scale:
                sample_idx = np.linspace(0, len(coords) - 1, max_patches_per_scale, dtype=int)
                coords = [coords[int(j)] for j in sample_idx]
            for x, y in coords:
                patch = cv2.resize(img[y : y + sz, x : x + sz], (224, 224), interpolation=cv2.INTER_AREA)
                patches.append(patch)
        if not patches and has_tissue:
            patches.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA))
        result[int(sz)] = patches
    return result


def get_uni_transform():
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    config = resolve_data_config({}, model="vit_large_patch16_224")
    return create_transform(**config)


def build_uni_model():
    import timm

    return timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )


def load_uni(device: str):
    model = build_uni_model()
    candidates = []
    env_weight = os.environ.get("BRACS_UNI_WEIGHTS") or os.environ.get("UNI_WEIGHTS_PATH")
    if env_weight:
        candidates.append(Path(env_weight).expanduser())
    candidates.extend(
        [
            ROOT / "weights" / "pytorch_model.bin",
            ROOT / "uni" / "pytorch_model.bin",
            ROOT / "uni.pth",
            ROOT / "pytorch_model.bin",
        ]
    )
    for path in candidates:
        if path.exists():
            state_dict = torch.load(path, map_location="cpu")
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
            model.to(device).eval()
            return model

    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("MahmoodLab/UNI", "pytorch_model.bin")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def encode_roi_features(
    image: Image.Image,
    uni_model,
    transform,
    device: str,
    target_sizes: list[int],
    batch_size: int = 8,
    max_patches_per_scale: int = 64,
) -> dict[int, torch.Tensor]:
    patch_dict = extract_patches_from_roi(image, target_sizes, max_patches_per_scale=max_patches_per_scale)
    features: dict[int, torch.Tensor] = {}
    for sz in sorted(patch_dict):
        patches = patch_dict[sz]
        if not patches:
            features[int(sz)] = torch.zeros((1, 1024), dtype=torch.float32)
            continue
        all_feats = []
        for start in range(0, len(patches), batch_size):
            imgs = torch.stack([transform(Image.fromarray(p).convert("RGB")) for p in patches[start : start + batch_size]])
            with torch.inference_mode():
                feats = uni_model(imgs.to(device)).float().cpu()
            all_feats.append(feats)
            clear_device_cache(device)
        features[int(sz)] = torch.cat(all_feats, dim=0)
    return features


class NystromAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, num_landmarks: int = 64, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x, mask=None):
        bsz, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, n_tokens, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if n_tokens <= self.num_landmarks:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask[:, None, None, :] == 0, -1e4)
            attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
            out = self.attn_drop(attn) @ v
        else:
            stride = max(1, n_tokens // self.num_landmarks)
            q_landmarks = q[:, :, ::stride, :]
            k_landmarks = k[:, :, ::stride, :]
            qk = (q @ k_landmarks.transpose(-2, -1)) * self.scale
            kq = (k_landmarks @ q_landmarks.transpose(-2, -1)) * self.scale
            if mask is not None:
                m_lm = mask[:, ::stride]
                qk = qk.masked_fill(m_lm[:, None, None, :] == 0, -1e4)
                kq = kq.masked_fill(m_lm[:, None, None, :] == 0, -1e4)
            qk = F.softmax(qk.float(), dim=-1).to(q.dtype)
            kq = F.softmax(kq.float(), dim=-1).to(q.dtype)
            qk = torch.nan_to_num(qk, nan=0.0, posinf=1.0, neginf=0.0)
            kq = torch.nan_to_num(kq, nan=0.0, posinf=1.0, neginf=0.0)
            kq_fp32 = torch.nan_to_num(kq.float(), nan=0.0, posinf=1.0, neginf=0.0)
            try:
                qk_pinv = torch.linalg.pinv(kq_fp32)
            except Exception:
                qk_pinv = torch.linalg.pinv(kq_fp32.cpu()).to(kq_fp32.device)
            qk_pinv = torch.nan_to_num(qk_pinv.to(kq.dtype), nan=0.0, posinf=1.0, neginf=0.0)
            v_scores = (q_landmarks @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                v_scores = v_scores.masked_fill(mask[:, None, None, :] == 0, -1e4)
            v_attn = F.softmax(v_scores.float(), dim=-1).to(v.dtype)
            v_attn = torch.nan_to_num(v_attn, nan=0.0, posinf=1.0, neginf=0.0)
            v_tilde = v_attn @ v
            out = qk @ qk_pinv @ v_tilde
        out = out.transpose(1, 2).contiguous().view(bsz, n_tokens, dim)
        return torch.nan_to_num(self.proj(out), nan=0.0, posinf=20.0, neginf=-20.0)


class SafeLayerNorm(nn.Module):
    """LayerNorm implemented with explicit FP32 math for Apple MPS stability."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        dims = tuple(range(xf.dim() - len(self.normalized_shape), xf.dim()))
        mean = xf.mean(dim=dims, keepdim=True)
        var = (xf - mean).pow(2).mean(dim=dims, keepdim=True)
        y = (xf - mean) * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight.float() + self.bias.float()
        return torch.nan_to_num(y, nan=0.0, posinf=20.0, neginf=-20.0).to(dtype)


class TransMILBlock(nn.Module):
    def __init__(self, dim=512, num_heads=8, num_landmarks=64, mlp_ratio=4.0, dropout=0.25, sd_prob=0.0):
        super().__init__()
        self.sd_prob = sd_prob
        self.norm1 = SafeLayerNorm(dim)
        self.attn = NystromAttention(dim, num_heads, num_landmarks, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = SafeLayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        if self.training and self.sd_prob > 0 and torch.rand(1).item() < self.sd_prob:
            return x
        x = x + self.drop1(self.attn(self.norm1(x), mask))
        x = x + self.mlp(self.norm2(x))
        return x


class GatedAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.V = nn.Linear(dim, dim)
        self.U = nn.Linear(dim, dim)
        self.w = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        attn = self.w(torch.tanh(self.V(x)) + torch.sigmoid(self.U(x)))
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        attn = F.softmax(attn.float(), dim=1).to(x.dtype)
        return torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)


class TransMIL(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        n_classes=3,
        dim=768,
        n_layers=3,
        num_heads=8,
        num_landmarks=64,
        dropout=0.3,
        max_len=512,
        stochastic_depth=0.1,
        context_fusion_dim=512,
    ):
        super().__init__()
        self.dim = dim
        self.fc = nn.Sequential(
            SafeLayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            SafeLayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList(
            [
                TransMILBlock(
                    dim,
                    num_heads,
                    num_landmarks,
                    dropout=dropout,
                    sd_prob=stochastic_depth * (i / max(n_layers - 1, 1)),
                )
                for i in range(n_layers)
            ]
        )
        self.norm = SafeLayerNorm(dim)
        self.attention = GatedAttention(dim)
        self.context_fusion = nn.Sequential(
            nn.Linear(dim, context_fusion_dim),
            SafeLayerNorm(context_fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_fusion_dim, dim),
        )
        self.classifier = nn.Linear(dim, n_classes)
        self.uncertainty_head = None

    def forward(self, x, mask=None):
        if isinstance(x, dict):
            feats_list = []
            sorted_keys = sorted(x.keys())
            for sz in sorted_keys:
                x_sz = x[sz]
                if x_sz.size(-1) != self.dim:
                    x_sz = self.fc(x_sz)
                feats_list.append(x_sz)
            x = torch.cat(feats_list, dim=1)
            if isinstance(mask, dict):
                mask = torch.cat([mask[sz] for sz in sorted_keys], dim=1)
            _, n_tokens, _ = x.shape
        else:
            _, n_tokens, _ = x.shape
            x = self.fc(x)
        if n_tokens <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :n_tokens]
        else:
            pos = F.interpolate(self.pos_embed.transpose(1, 2), size=n_tokens, mode="linear", align_corners=False)
            x = x + pos.transpose(1, 2)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        attn = self.attention(x, mask)
        pooled = (x * attn).sum(dim=1)
        return torch.nan_to_num(self.classifier(self.context_fusion(pooled)), nan=0.0, posinf=20.0, neginf=-20.0)


class DTFDWrapper(nn.Module):
    def __init__(self, base_model, n_pseudo=8):
        super().__init__()
        self.base = base_model
        self.n_pseudo = n_pseudo

    def forward(self, x, mask=None):
        return self.base(x, mask)


def _normalize_state_keys(state: dict[str, torch.Tensor], target: nn.Module) -> dict[str, torch.Tensor]:
    target_keys = set(target.state_dict().keys())
    state_keys = set(state.keys())
    target_dp = any(k.startswith("module.") for k in target_keys)
    state_dp = any(k.startswith("module.") for k in state_keys)
    if target_dp and not state_dp:
        return {"module." + k: v for k, v in state.items()}
    if not target_dp and state_dp:
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def build_mil_model(config: dict[str, Any]) -> nn.Module:
    model = TransMIL(
        in_dim=int(config.get("feature_dim", 1024)),
        n_classes=int(config.get("num_classes", 3)),
        dim=int(config.get("transmil_dim", 768)),
        n_layers=int(config.get("transmil_layers", 3)),
        num_heads=int(config.get("transmil_heads", 8)),
        num_landmarks=int(config.get("transmil_landmarks", 64)),
        dropout=float(config.get("dropout", 0.3)),
        max_len=int(config.get("max_patches_per_bag", 512)),
        stochastic_depth=float(config.get("stochastic_depth", 0.1)),
        context_fusion_dim=int(config.get("context_fusion_dim", 512)),
    )
    if int(config.get("n_pseudo_bags", 8)) > 0:
        return DTFDWrapper(model, n_pseudo=int(config.get("n_pseudo_bags", 8)))
    return model


def discover_checkpoint_paths(limit_models: int = 3) -> list[Path]:
    manifests = sorted(
        WORK_DIR.glob("transmil_dtfd_output_validation_*/main_checkpoint_manifest.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    complete_manifests: list[tuple[Path, list[Path]]] = []
    fallback_manifests: list[tuple[Path, list[Path]]] = []
    for manifest in manifests:
        try:
            rows = pd.read_csv(manifest).sort_values("val_f1", ascending=False)
            paths = [Path(p) for p in rows["path"].tolist() if Path(p).exists()]
        except Exception:
            paths = []
        if paths:
            if len(paths) >= EXPECTED_MODELS:
                complete_manifests.append((manifest, paths))
            else:
                fallback_manifests.append((manifest, paths))
    selected = complete_manifests[0][1] if complete_manifests else (fallback_manifests[0][1] if fallback_manifests else [])
    if selected:
        return selected[: max(1, int(limit_models))]
    fallback = sorted(WORK_DIR.glob("transmil_dtfd_output/transmil_seed*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return fallback[: max(1, int(limit_models))]


def _prepare_bag(features: dict[int, torch.Tensor], device: str, max_patches: int = 512):
    bag: dict[int, torch.Tensor] = {}
    masks: dict[int, torch.Tensor] = {}
    for sz, feats in features.items():
        if feats.shape[0] > max_patches:
            idx = torch.linspace(0, feats.shape[0] - 1, max_patches).long()
            feats = feats[idx]
        n = feats.shape[0]
        pad = torch.zeros((max_patches, feats.shape[1]), dtype=torch.float32)
        mask = torch.zeros(max_patches, dtype=torch.long)
        pad[:n] = feats
        mask[:n] = 1
        bag[int(sz)] = pad.unsqueeze(0).to(device)
        masks[int(sz)] = mask.unsqueeze(0).to(device)
    return bag, masks


def predict_roi_image(
    image: Image.Image,
    device: str = "auto",
    limit_models: int = 3,
    max_patches_per_scale: int = 64,
    batch_size: int = 8,
    threshold: float = DEFAULT_ATYPIA_THRESHOLD,
) -> dict[str, Any]:
    device = select_device(device)
    ckpt_paths = discover_checkpoint_paths(limit_models)
    if not ckpt_paths:
        raise FileNotFoundError("No trained MIL checkpoints found yet.")

    uni = load_uni(device)
    transform = get_uni_transform()
    features = encode_roi_features(
        image,
        uni,
        transform,
        device,
        DEFAULT_MULTI_SCALE_SIZES,
        batch_size=batch_size,
        max_patches_per_scale=max_patches_per_scale,
    )
    del uni
    clear_device_cache(device)

    all_probs = []
    used_paths = []
    max_patches = 512
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if ckpt.get("checkpoint_role") not in (None, "main"):
            continue
        config = dict(ckpt.get("config") or {})
        max_patches = int(config.get("max_patches_per_bag", max_patches))
        model = build_mil_model(config)
        model.load_state_dict(_normalize_state_keys(ckpt["model_state"], model), strict=True)
        if config.get("use_swa", True) and ckpt.get("swa_state"):
            model.load_state_dict(_normalize_state_keys(ckpt["swa_state"], model), strict=True)
        elif config.get("use_ema", True) and ckpt.get("ema_state"):
            model.load_state_dict(_normalize_state_keys(ckpt["ema_state"], model), strict=True)
        model.to(device).eval()
        bag, masks = _prepare_bag(features, device, max_patches=max_patches)
        with torch.inference_mode():
            logits = model(bag, masks)
            temperature = ckpt.get("temperature")
            if temperature:
                logits = logits / float(temperature)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        all_probs.append(probs)
        used_paths.append(str(ckpt_path))
        del model, ckpt
        clear_device_cache(device)

    if not all_probs:
        raise RuntimeError("No usable main checkpoints were loaded.")

    probs = np.mean(np.vstack(all_probs), axis=0)
    pred = int(atypia_threshold_predictions(probs.reshape(1, -1), threshold=threshold)[0])
    sorted_probs = np.sort(probs)
    confidence = float(probs[pred])
    margin = float(sorted_probs[-1] - sorted_probs[-2])
    atypia_boundary_distance = float(abs(probs[1] - threshold))
    review = bool(
        margin <= REVIEW_MARGIN_THRESHOLD
        or confidence < 0.60
        or atypia_boundary_distance <= ATYPIA_BOUNDARY_MARGIN
    )
    return {
        "pred_label": pred,
        "predicted_class": CLASS_NAMES[pred],
        "confidence": confidence,
        "margin": margin,
        "review_recommended": review,
        "atypia_boundary_distance": atypia_boundary_distance,
        "probabilities": {PROB_COLS[i]: float(probs[i]) for i in range(3)},
        "models_used": len(used_paths),
        "checkpoint_paths": used_paths,
        "device": device,
        "max_patches_per_scale": int(max_patches_per_scale),
    }


def result_frame(result: dict[str, Any]) -> pd.DataFrame:
    row = {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "margin": result["margin"],
        "review_recommended": result["review_recommended"],
        "models_used": result["models_used"],
        "device": result["device"],
    }
    row.update(result["probabilities"])
    return pd.DataFrame([row])
