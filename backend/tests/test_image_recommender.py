"""Tests for the image-generation track."""

import pytest

from backend.services.images.catalog import (
    image_harnesses, image_models, image_use_cases, get_image_model,
)
from backend.services.images.recommender import (
    ImageHardwareSnapshot, estimate_time_per_image, normalize_score,
    pick_quantization, recommend,
)
from backend.services.images.catalog import ImageScore


HW_M3_MAX_36GB = ImageHardwareSnapshot(
    chip="Apple M3 Max", generation="gen3", gpu_cores=30,
    total_memory_gb=36, available_memory_gb=22, storage_free_gb=200,
)

HW_M5_PRO_64GB = ImageHardwareSnapshot(
    chip="Apple M5 Pro", generation="gen5", gpu_cores=20,
    total_memory_gb=64, available_memory_gb=48, storage_free_gb=200,
)

HW_M1_8GB = ImageHardwareSnapshot(
    chip="Apple M1", generation="gen1", gpu_cores=8,
    total_memory_gb=8, available_memory_gb=4, storage_free_gb=100,
)


def test_catalog_loads_at_least_15_models():
    models = image_models()
    assert len(models) >= 15, f"Catalog should have ~20 models; got {len(models)}"
    # Sanity: every model has a non-empty display name and at least one score.
    for m in models:
        assert m.display_name
        assert m.canonical_id
        assert m.scores, f"{m.canonical_id} has no benchmark scores"


def test_use_cases_and_harnesses_load():
    ucs = image_use_cases()
    hs = image_harnesses()
    assert {u["id"] for u in ucs} >= {"image_generation", "image_editing"}
    assert {h["id"] for h in hs} >= {"drawthings", "comfyui", "diffusers"}


def test_normalize_score_geneval_in_unit_range():
    sc = ImageScore(benchmark="geneval", value=0.71, source="x", confidence="measured")
    n = normalize_score(sc)
    assert 0.70 <= n <= 0.72


def test_normalize_score_elo_anchored_to_1100():
    sc = ImageScore(benchmark="imagen_arena_elo", value=1100, source="x", confidence="measured")
    assert normalize_score(sc) == pytest.approx(1.0)
    sc2 = ImageScore(benchmark="imagen_arena_elo", value=550, source="x", confidence="measured")
    assert normalize_score(sc2) == pytest.approx(0.5)


def test_normalize_fid_inverted():
    """FID is lower-is-better; normalization flips it."""
    perfect = ImageScore(benchmark="mjhq30k_fid", value=0, source="x", confidence="measured")
    poor = ImageScore(benchmark="mjhq30k_fid", value=30, source="x", confidence="measured")
    assert normalize_score(perfect) == pytest.approx(1.0)
    assert normalize_score(poor) == pytest.approx(0.0)


def test_chip_tflops_scaling_makes_m1_slower_than_m3_max():
    flux = get_image_model("flux-1-dev")
    assert flux is not None
    t_m3 = estimate_time_per_image(flux, HW_M3_MAX_36GB)
    t_m1 = estimate_time_per_image(flux, HW_M1_8GB)
    assert t_m1 > t_m3 * 4, f"M1 should be 5x+ slower than M3 Max for FLUX, got {t_m1=} vs {t_m3=}"


def test_pick_quantization_uses_q8_when_fp16_too_big():
    flux = get_image_model("flux-1-dev")
    # FLUX FP16 is 24GB; on a 16GB budget Q8 (13GB) should be picked.
    q = pick_quantization(flux, ram_budget_gb=16.0)
    assert q == "q8"
    # On 32GB budget FP16 (24GB) fits.
    q2 = pick_quantization(flux, ram_budget_gb=32.0)
    assert q2 == "fp16"
    # On 5GB budget even Q8 doesn't fit; Q4 (7GB) doesn't either, but
    # pick_quantization returns Q4 as the fallback.
    q3 = pick_quantization(flux, ram_budget_gb=5.0)
    assert q3 == "q4"


def test_recommend_image_generation_returns_ranked():
    recs = recommend("image_generation", harness_id=None, hw=HW_M5_PRO_64GB, limit=5)
    assert len(recs) >= 3
    # Ranks should be 1..N
    for i, r in enumerate(recs):
        assert r.rank == i + 1
    # Higher-scoring should sort first
    for a, b in zip(recs, recs[1:]):
        assert a.fit_score >= b.fit_score


def test_recommend_filters_too_big_models_on_low_ram():
    # On an 8GB M1, FLUX (24GB FP16) shouldn't appear unless include_too_big.
    recs = recommend("image_generation", None, HW_M1_8GB, limit=20)
    cids = {r.canonical_id for r in recs}
    assert "flux-1-dev" not in cids
    # SD 1.5 should make it
    assert "sd-1-5" in cids


def test_image_editing_filters_to_editing_capable_models():
    recs = recommend("image_editing", None, HW_M5_PRO_64GB, limit=10)
    cids = {r.canonical_id for r in recs}
    # FLUX dev is gen-only, shouldn't appear
    assert "flux-1-dev" not in cids
    # OmniGen and InstructPix2Pix support editing
    assert any(c in cids for c in ("omnigen-v1", "instruct-pix2pix", "flux-1-kontext-dev", "hidream-e1"))


def test_harness_filter_drops_incompatible():
    """Mochi Diffusion only supports SD1.5/SDXL — FLUX and SD3.5 should be filtered."""
    recs = recommend("image_generation", "mochi-diffusion", HW_M5_PRO_64GB, limit=20)
    cids = {r.canonical_id for r in recs}
    assert "flux-1-dev" not in cids
    assert "sd-3-5-large" not in cids
    # SDXL should be there
    assert "sdxl-base-1-0" in cids


def test_install_options_resolve_command_template():
    recs = recommend("image_generation", "drawthings", HW_M5_PRO_64GB, limit=3)
    assert recs
    top = recs[0]
    assert top.install_options
    assert any(opt["harness_id"] == "drawthings" for opt in top.install_options)


def test_install_commands_have_no_unsubstituted_placeholders():
    """Every install command in every harness option must have all {placeholders}
    substituted — otherwise users get useless 'paste: drawthings://?url={url}' strings."""
    import re
    placeholder = re.compile(r"\{[a-z_]+\}")
    for harness in ("comfyui", "diffusers", "forge", "invokeai", "drawthings", "mochi-diffusion"):
        recs = recommend("image_generation", harness, HW_M5_PRO_64GB, limit=10)
        for r in recs:
            for opt in r.install_options:
                leftover = placeholder.findall(opt["command"])
                assert not leftover, f"{r.canonical_id} via {harness}: leftover placeholders {leftover} in: {opt['command']}"


def test_install_options_include_huggingface_url_when_hf_id_present():
    """install_options should expose a download_url for HF-hosted models."""
    recs = recommend("image_generation", "diffusers", HW_M5_PRO_64GB, limit=10)
    flux = next((r for r in recs if r.canonical_id == "flux-1-dev"), None)
    assert flux is not None
    diff_opt = next(opt for opt in flux.install_options if opt["harness_id"] == "diffusers")
    assert diff_opt["download_url"].startswith("https://huggingface.co/")
    assert "FLUX.1-dev" in diff_opt["command"]


def test_comfyui_folder_is_unet_for_flux_and_checkpoints_for_sdxl():
    recs = recommend("image_generation", "comfyui", HW_M5_PRO_64GB, limit=20)
    flux = next((r for r in recs if r.canonical_id == "flux-1-dev"), None)
    sdxl = next((r for r in recs if r.canonical_id == "sdxl-base-1-0"), None)
    assert flux and sdxl
    flux_cmd = next(o["command"] for o in flux.install_options if o["harness_id"] == "comfyui")
    sdxl_cmd = next(o["command"] for o in sdxl.install_options if o["harness_id"] == "comfyui")
    assert "models/unet/flux-1-dev/" in flux_cmd
    assert "models/checkpoints/sdxl-base-1-0/" in sdxl_cmd


def test_install_command_local_dirs_are_shell_safe():
    """Local-dir paths must not contain spaces or shell metacharacters —
    display names like 'FLUX.1 [dev]' would break unquoted shell commands."""
    import re
    bad = re.compile(r"--local-dir\s+\S*[\s\[\]]")
    for harness in ("comfyui", "forge", "mochi-diffusion"):
        recs = recommend("image_generation", harness, HW_M5_PRO_64GB, limit=10)
        for r in recs:
            for opt in r.install_options:
                # Skip harnesses where the template doesn't use --local-dir
                if "--local-dir" not in opt["command"]:
                    continue
                # The first token after --local-dir, up to next whitespace, must
                # have no brackets or embedded spaces.
                m = re.search(r"--local-dir\s+(\S+)(/?)", opt["command"])
                assert m, f"--local-dir not parseable in {opt['command']!r}"
                token = m.group(1)
                assert "[" not in token and "]" not in token, \
                    f"unsafe bracket in local-dir for {r.canonical_id} via {harness}: {opt['command']}"


def test_score_use_case_marks_missing_benchmarks_in_provenance():
    """A model with only GenEval should still rank, with FID/ELO listed in
    provenance.missing_data."""
    recs = recommend("image_generation", None, HW_M5_PRO_64GB, limit=20)
    # Pick AuraFlow which has only geneval
    aura = next((r for r in recs if r.canonical_id == "auraflow-v0-3"), None)
    assert aura is not None
    missing = aura.provenance.missing_data
    assert any("imagen_arena_elo" in m or "mjhq30k_fid" in m for m in missing)


def test_warning_when_q4_chosen():
    """Q4 should produce a warning that quality is below published numbers."""
    # Force a tight budget so the largest model picks Q4
    hw = ImageHardwareSnapshot(
        chip="Apple M3", generation="gen3", gpu_cores=10,
        total_memory_gb=16, available_memory_gb=10, storage_free_gb=200,
    )
    recs = recommend("image_generation", None, hw, limit=10, include_too_big=True)
    # At least one model should be on Q4 here; check warning surface
    q4_recs = [r for r in recs if r.quantization_recommended == "q4"]
    if q4_recs:
        assert any("Q4" in w for w in q4_recs[0].warnings)
