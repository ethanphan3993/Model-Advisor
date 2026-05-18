"""Image-generation recommender.

Same 3-axis structure as the text recommender:

  fit_score = 0.55 * use_case_score      (benchmark match: GenEval, ELO, FID)
            + 0.30 * hardware_fit         (RAM fit + time-per-image)
            + 0.15 * harness_fit          (Drawthings / ComfyUI / ... compatibility)

But the cost model is different:

  Text inference is memory-bandwidth-bound — TPS scales with bandwidth_GB_s
  divided by active_params * bytes_per_param. (See
  backend/services/recommender.py.)

  Diffusion inference is compute-bound at typical 1024² resolutions: each
  denoising step is a forward pass of the full UNet/DiT, and on Apple Silicon
  we're FLOPs-limited not bandwidth-limited. So we model:

      time_per_image = default_steps * scaled_time_per_step + overhead

  We keep empirical reference times per chip (M3 Max, M4 Max) in the catalog
  and scale to other chips by relative GPU FP16 TFLOPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from backend.services.hardware_fit import (
    bucket as _bucket, combine_subscores, score_storage_fit,
)
from backend.services.images.catalog import (
    ImageModel, ImageScore, get_image_harness, get_image_use_case, image_models,
)


# ---------------------------------------------------------------------------
# Apple Silicon GPU FP16 TFLOPS — what actually drives diffusion step time.
# Numbers compiled from Apple Metal benchmarks, llama.cpp ggml-metal results,
# and Geekbench Metal compute scores. These are GPU-only TFLOPS at FP16; the
# Neural Engine isn't used by current diffusion runtimes (Drawthings, ComfyUI,
# Diffusers all use Metal/MPS).
#
# We treat M3 Max as the reference point (matches catalog `time_per_step_seconds.m3_max`)
# and scale others by relative TFLOPS. M4 Max is also used as a reference if
# present in the catalog row.
# ---------------------------------------------------------------------------

CHIP_FP16_TFLOPS = {
    "Apple M1":       2.6,
    "Apple M1 Pro":   5.2,
    "Apple M1 Max":  10.4,
    "Apple M1 Ultra":21.0,
    "Apple M2":       3.6,
    "Apple M2 Pro":   6.8,
    "Apple M2 Max":  13.6,
    "Apple M2 Ultra":27.0,
    "Apple M3":       4.1,
    "Apple M3 Pro":   6.5,
    "Apple M3 Max":  14.2,         # reference
    "Apple M4":       4.6,
    "Apple M4 Pro":   9.3,
    "Apple M4 Max":  18.4,         # reference
    "Apple M5":       5.0,
    "Apple M5 Pro":  10.0,
    "Apple M5 Max":  20.0,
}

REF_M3_MAX_TFLOPS = CHIP_FP16_TFLOPS["Apple M3 Max"]
REF_M4_MAX_TFLOPS = CHIP_FP16_TFLOPS["Apple M4 Max"]

# Loader, VAE encode/decode, scheduler overhead. Roughly chip-independent and
# small relative to the per-step cost on the models that matter.
PIPELINE_OVERHEAD_SECONDS = 1.5


# ---------------------------------------------------------------------------
# Score normalization scales — analog to BENCHMARK_SCALE in the text recommender.
# ---------------------------------------------------------------------------

# GenEval is reported as fraction 0..1 (percent of prompts where requested
# attributes are correct). Higher is better.
# imagen-arena ELO sits roughly in 850..1150 for current open-weights models.
# We anchor a 1100 ELO to 1.0 normalized — anything above is exceptional.
# Emu-Edit instruct accuracy is also 0..1.
BENCHMARK_SCALE = {
    "geneval": 1.0,
    "imagen_arena_elo": 1100.0,
    "emu_edit": 1.0,
    "mjhq30k_fid": 30.0,            # FID — lower is better, special-cased below
}


def normalize_score(score: ImageScore) -> float:
    """Map raw benchmark value to 0..1. FID is inverted (lower is better)."""
    scale = BENCHMARK_SCALE.get(score.benchmark, 1.0)
    if score.benchmark == "mjhq30k_fid":
        # FID 0 = perfect, 30 = poor. Invert and clamp.
        return max(0.0, min(1.0, (scale - score.value) / scale))
    return max(0.0, min(1.0, score.value / scale))


# ---------------------------------------------------------------------------
# Hardware snapshot — reuses the text-track dataclass so router code is
# uniform. We add helpers for compute-bound modeling.
# ---------------------------------------------------------------------------

@dataclass
class ImageHardwareSnapshot:
    chip: str
    generation: str
    gpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    storage_free_gb: float

    def fp16_tflops(self) -> float:
        if self.chip in CHIP_FP16_TFLOPS:
            return CHIP_FP16_TFLOPS[self.chip]
        for k, v in CHIP_FP16_TFLOPS.items():
            if self.chip.startswith(k):
                return v
        # Fallback by generation, conservative.
        return {"gen5": 5.0, "gen4": 4.6, "gen3": 4.1, "gen2": 3.6, "gen1": 2.6}.get(self.generation, 4.0)

    def ram_budget_gb(self, headroom: float = 0.70) -> float:
        return max(self.available_memory_gb, self.total_memory_gb * headroom)


def _scaled_time_per_step(model: ImageModel, hw: ImageHardwareSnapshot) -> float:
    """Estimate seconds per denoising step for this model on this chip.

    Uses the catalog's empirical M3 Max reference (or M4 Max if present, picking
    whichever yields a more accurate estimate for the user's chip generation),
    then scales by FP16 TFLOPS ratio.

    Returns 0.0 when the catalog has no reference time — caller should treat
    as missing data.
    """
    chip_tflops = hw.fp16_tflops()
    m3_ref = model.time_per_step_seconds.get("m3_max")
    m4_ref = model.time_per_step_seconds.get("m4_max")

    # Prefer the closer reference: gen4/gen5 chips → M4 Max ref, else M3 Max ref.
    if hw.generation in {"gen4", "gen5"} and m4_ref:
        return float(m4_ref) * (REF_M4_MAX_TFLOPS / max(chip_tflops, 0.1))
    if m3_ref:
        return float(m3_ref) * (REF_M3_MAX_TFLOPS / max(chip_tflops, 0.1))
    if m4_ref:
        return float(m4_ref) * (REF_M4_MAX_TFLOPS / max(chip_tflops, 0.1))
    return 0.0


def estimate_time_per_image(model: ImageModel, hw: ImageHardwareSnapshot,
                            steps_override: Optional[int] = None) -> float:
    """Total wall time for one 1024² image at default (or overridden) step count."""
    per_step = _scaled_time_per_step(model, hw)
    if per_step <= 0:
        return 0.0
    steps = steps_override or model.default_steps
    return round(per_step * steps + PIPELINE_OVERHEAD_SECONDS, 1)


def pick_quantization(model: ImageModel, ram_budget_gb: float) -> str:
    """Pick the highest-quality quant whose VRAM footprint fits the budget.

    Falls back to q4 if even fp16 would be the only option but doesn't fit —
    UI will surface a 'too big' warning separately.
    """
    for q in ("fp16", "q8", "q4"):
        size = model.vram_gb.get(q)
        if size is None:
            continue
        if size <= ram_budget_gb:
            return q
    # Nothing fits cleanly — return q4 as the most permissive option
    return "q4"


# ---------------------------------------------------------------------------
# Recommendation result types
# ---------------------------------------------------------------------------

@dataclass
class ImageScoreEvidence:
    benchmark: str
    value: float
    normalized: float
    source: str
    confidence: str


@dataclass
class ImageProvenance:
    use_case_components: list[ImageScoreEvidence] = field(default_factory=list)
    hardware_components: dict[str, float] = field(default_factory=dict)
    harness_components: dict[str, float] = field(default_factory=dict)
    missing_data: list[str] = field(default_factory=list)


@dataclass
class ImageRecommendation:
    rank: int
    canonical_id: str
    display_name: str
    family: str
    variant: str
    architecture: str
    fit_score: float
    use_case_score: float
    hardware_fit: float
    harness_fit: float
    confidence: str
    confidence_pct: int
    benchmarks_measured: int
    benchmarks_expected: int
    quantization_recommended: str
    estimated_vram_gb: float
    estimated_time_per_image_s: float
    default_steps: int
    fits_currently_free: bool
    license: str
    supports: list[str]
    install_options: list[dict]
    warnings: list[str]
    provenance: ImageProvenance
    why: str
    notes: str


# ---------------------------------------------------------------------------
# Use-case scoring
# ---------------------------------------------------------------------------

def score_use_case(model: ImageModel, use_case_id: str,
                   prov: ImageProvenance) -> tuple[float, int, int]:
    uc = get_image_use_case(use_case_id)
    if not uc:
        return 0.0, 0, 0
    benchmarks: dict[str, float] = uc.get("benchmarks") or {}
    expected = len(benchmarks)
    if not benchmarks:
        return 5.0, 0, 0

    total_weight = 0.0
    weighted_sum = 0.0
    measured = 0
    for bench, weight in benchmarks.items():
        sc = model.scores.get(bench)
        if sc is None:
            prov.missing_data.append(f"use_case:{bench}")
            continue
        norm = normalize_score(sc)
        prov.use_case_components.append(ImageScoreEvidence(
            benchmark=bench, value=sc.value, normalized=norm,
            source=sc.source, confidence=sc.confidence,
        ))
        weighted_sum += norm * float(weight)
        total_weight += float(weight)
        measured += 1

    if total_weight == 0:
        return 0.0, 0, expected
    raw = weighted_sum / total_weight * 10
    return round(min(10.0, raw), 2), measured, expected


def filter_use_case(model: ImageModel, use_case_id: str) -> tuple[bool, list[str]]:
    uc = get_image_use_case(use_case_id)
    if not uc:
        return True, []
    requires = uc.get("requires") or {}
    if requires.get("supports_editing") and "image_editing" not in model.supports:
        return False, ["Doesn't support editing"]
    if use_case_id == "image_generation" and "image_generation" not in model.supports:
        return False, ["Editing-only model"]
    return True, []


# ---------------------------------------------------------------------------
# Hardware fit
# ---------------------------------------------------------------------------

def score_hardware(model: ImageModel, hw: ImageHardwareSnapshot,
                   prov: ImageProvenance) -> tuple[float, str, float, float, bool]:
    """Compute-bound hardware fit.

    Returns (score 0..10, recommended_quant, vram_gb, time_per_image_s,
              fits_currently_free).
    """
    budget_gb = hw.ram_budget_gb()
    avail_gb = hw.available_memory_gb
    quant = pick_quantization(model, budget_gb)
    vram = model.vram_gb.get(quant) or model.vram_gb.get("q4") or 0.0

    # 1. Memory fit out of 10. Image-track thresholds are stricter than the
    # text track once ratio > 0.85 because diffusion runtimes (Drawthings,
    # ComfyUI, Diffusers) tend to hang or crash when VRAM is overcommitted,
    # whereas text inference just slows down.
    mem_ratio = vram / max(budget_gb, 0.5)
    mem_score = _bucket(mem_ratio, [
        (0.40, 10), (0.60, 9), (0.75, 8), (0.90, 6), (1.05, 3),
    ], default=0)
    fits_currently = vram <= avail_gb * 0.95
    prov.hardware_components["memory_score"] = mem_score
    prov.hardware_components["vram_gb"] = round(vram, 2)
    prov.hardware_components["budget_gb"] = round(budget_gb, 2)
    prov.hardware_components["fits_currently"] = 1.0 if fits_currently else 0.0

    # 2. Speed score — time per image at default step count
    t_per_image = estimate_time_per_image(model, hw)
    if t_per_image <= 0:
        speed_score = 5  # no reference data, neutral
        prov.missing_data.append("hardware:time_per_step")
    elif t_per_image < 10:
        speed_score = 10
    elif t_per_image < 20:
        speed_score = 9
    elif t_per_image < 40:
        speed_score = 7
    elif t_per_image < 80:
        speed_score = 5
    elif t_per_image < 180:
        speed_score = 3
    else:
        speed_score = 1
    prov.hardware_components["speed_score"] = speed_score
    prov.hardware_components["time_per_image_s"] = t_per_image
    prov.hardware_components["fp16_tflops"] = hw.fp16_tflops()

    # 3. Storage check — single-file safetensors typically 2-25 GB
    download_gb = model.vram_gb.get("fp16", 0)  # roughly == disk size for fp16
    st_score = score_storage_fit(download_gb, hw.storage_free_gb)
    prov.hardware_components["storage_score"] = st_score

    combined = combine_subscores(mem_score, speed_score, st_score)
    return combined, quant, round(vram, 2), t_per_image, fits_currently


# ---------------------------------------------------------------------------
# Harness scoring
# ---------------------------------------------------------------------------

def score_harness(model: ImageModel, harness_id: Optional[str],
                  prov: ImageProvenance) -> tuple[float, bool, list[str]]:
    if not harness_id:
        return 7.0, True, []
    h = get_image_harness(harness_id)
    if not h:
        return 7.0, True, []

    warnings: list[str] = []
    requires = h.get("requires") or {}
    prefers = h.get("prefers") or {}
    passes = True

    if harness_id not in model.harnesses_compatible:
        passes = False
        warnings.append(f"{h['name']} doesn't support {model.display_name}")

    family_in = requires.get("family_in")
    if family_in and model.family not in family_in:
        passes = False
        warnings.append(f"{h['name']} only supports families: {', '.join(family_in)}")

    score = 8.0 if passes else 0.0
    family_bonus = (prefers.get("family_bonus") or {}) if isinstance(prefers, dict) else {}
    if isinstance(family_bonus, dict):
        score += float(family_bonus.get(model.family, 0.0))
    score = max(0.0, min(10.0, score))
    prov.harness_components["harness_pass"] = 1.0 if passes else 0.0
    prov.harness_components["harness_score"] = score
    return score, passes, warnings


# ---------------------------------------------------------------------------
# Install options
# ---------------------------------------------------------------------------

def install_options_for(model: ImageModel, harness_id: Optional[str]) -> list[dict]:
    """Return install hints with placeholders substituted from the model's
    hf_id, display_name, and comfyui_folder. Includes a Hugging Face download
    URL when the model has an hf_id; otherwise just the harness command.
    """
    out: list[dict] = []
    target_ids = [harness_id] if harness_id else model.harnesses_compatible
    for hid in target_ids:
        h = get_image_harness(hid)
        if not h:
            continue
        if hid not in model.harnesses_compatible:
            continue
        cmd = h.get("install_command_template", "")
        cmd = (cmd
               .replace("{display_name}", model.display_name)
               .replace("{hf_id}", model.hf_id or model.canonical_id)
               .replace("{comfyui_folder}", model.comfyui_folder)
               .replace("{slug}", model.canonical_id))
        url = f"https://huggingface.co/{model.hf_id}" if model.hf_id else ""
        out.append({
            "harness": h["name"],
            "harness_id": hid,
            "command": cmd,
            "homepage": h.get("homepage", ""),
            "download_url": url,
        })
    return out


# ---------------------------------------------------------------------------
# Confidence + why
# ---------------------------------------------------------------------------

def confidence_label(prov: ImageProvenance) -> str:
    if len(prov.use_case_components) >= 2 and len(prov.missing_data) == 0:
        return "high"
    if len(prov.use_case_components) >= 1:
        return "medium"
    return "low"


def confidence_pct(measured: int, expected: int) -> int:
    if expected == 0:
        return 0
    return min(100, int(round(measured / expected * 100)))


def build_why(model: ImageModel, prov: ImageProvenance,
              t_per_image: float, harness_name: Optional[str]) -> str:
    parts: list[str] = []
    if prov.use_case_components:
        top = sorted(prov.use_case_components, key=lambda e: e.normalized, reverse=True)[:2]
        rendered = []
        for e in top:
            if e.benchmark == "imagen_arena_elo":
                rendered.append(f"ELO {int(e.value)}")
            elif e.benchmark == "geneval":
                rendered.append(f"GenEval {e.value:.2f}")
            elif e.benchmark == "emu_edit":
                rendered.append(f"Emu-Edit {e.value:.2f}")
            else:
                rendered.append(f"{e.benchmark} {e.value:.2f}")
        parts.append("strong on " + ", ".join(rendered))
    if t_per_image > 0:
        if t_per_image < 15:
            parts.append(f"~{t_per_image:.0f}s per image")
        else:
            parts.append(f"~{t_per_image:.0f}s per image (slow)")
    if harness_name and prov.harness_components.get("harness_pass") == 1.0:
        parts.append(f"works in {harness_name}")
    if "memory_score" in prov.hardware_components:
        ms = prov.hardware_components["memory_score"]
        if ms >= 8:
            parts.append("fits comfortably in RAM")
        elif ms >= 4:
            parts.append("tight RAM fit")
        elif ms > 0:
            parts.append("RAM-constrained")
    return f"{model.display_name}: " + "; ".join(parts) if parts else f"{model.display_name}: limited benchmark data"


# ---------------------------------------------------------------------------
# Top-level recommend
# ---------------------------------------------------------------------------

def recommend(use_case_id: str, harness_id: Optional[str],
              hw: ImageHardwareSnapshot, limit: int = 10,
              include_too_big: bool = False) -> list[ImageRecommendation]:
    h_name = None
    if harness_id:
        h = get_image_harness(harness_id)
        h_name = h["name"] if h else None

    results: list[ImageRecommendation] = []
    for m in image_models():
        passes_uc, uc_warns = filter_use_case(m, use_case_id)
        if not passes_uc:
            continue
        prov = ImageProvenance()
        h_score, h_pass, h_warns = score_harness(m, harness_id, prov)
        if harness_id and not h_pass:
            continue
        uc_score, measured, expected = score_use_case(m, use_case_id, prov)
        if measured == 0:
            # Image catalog is small and curated, so an unscored entry is a real
            # data gap. Skip rather than ranking neutral.
            continue
        hw_score, quant, vram, t_per_image, fits_currently = score_hardware(m, hw, prov)
        if not include_too_big and prov.hardware_components.get("vram_gb", 0) > hw.ram_budget_gb() * 1.10:
            continue

        fit = 0.55 * uc_score + 0.30 * hw_score + 0.15 * h_score
        warnings = uc_warns + h_warns
        if not fits_currently and prov.hardware_components.get("vram_gb", 0) <= hw.ram_budget_gb():
            warnings.insert(0, f"Free up RAM to run — needs {vram:.1f} GB but only {hw.available_memory_gb:.1f} GB free")
        if t_per_image >= 60:
            warnings.append(f"Slow on this chip — ~{t_per_image:.0f}s per 1024² image")
        if quant in ("q4",) and m.vram_gb.get("fp16", 0) > 0:
            warnings.append("Q4 quantization — quality below the published numbers (which assume FP16)")

        results.append(ImageRecommendation(
            rank=0,
            canonical_id=m.canonical_id,
            display_name=m.display_name,
            family=m.family,
            variant=m.variant,
            architecture=m.architecture,
            fit_score=round(fit, 2),
            use_case_score=uc_score,
            hardware_fit=hw_score,
            harness_fit=h_score,
            confidence=confidence_label(prov),
            confidence_pct=confidence_pct(measured, expected),
            benchmarks_measured=measured,
            benchmarks_expected=expected,
            quantization_recommended=quant,
            estimated_vram_gb=vram,
            estimated_time_per_image_s=t_per_image,
            default_steps=m.default_steps,
            fits_currently_free=fits_currently,
            license=m.license,
            supports=list(m.supports),
            install_options=install_options_for(m, harness_id),
            warnings=warnings,
            provenance=prov,
            why=build_why(m, prov, t_per_image, h_name),
            notes=m.notes,
        ))

    results.sort(key=lambda r: r.fit_score, reverse=True)
    top = results[:limit]
    for i, r in enumerate(top):
        r.rank = i + 1
    return top
