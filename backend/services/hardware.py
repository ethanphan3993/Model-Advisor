"""Hardware scanning service — refactored from scan_hardware.py.

Runs on macOS only. Uses system_profiler, sysctl, vm_stat to extract
device capabilities relevant to AI model deployment.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Optional


NEURAL_ENGINE_CORES = {
    "M4": 16, "M4 Pro": 16, "M4 Max": 16, "M4 Ultra": 16,
    "M3": 16, "M3 Pro": 16, "M3 Max": 16, "M3 Ultra": 16,
    "M2": 16, "M2 Pro": 16, "M2 Max": 16, "M2 Ultra": 16,
    "M1": 16, "M1 Pro": 16, "M1 Max": 16, "M1 Ultra": 16,
}

CHIP_TIER = {
    "M1": "gen1", "M1 Pro": "gen1", "M1 Max": "gen1", "M1 Ultra": "gen1",
    "M2": "gen2", "M2 Pro": "gen2", "M2 Max": "gen2", "M2 Ultra": "gen2",
    "M3": "gen3", "M3 Pro": "gen3", "M3 Max": "gen3", "M3 Ultra": "gen3",
    "M4": "gen4", "M4 Pro": "gen4", "M4 Max": "gen4", "M4 Ultra": "gen4",
    "M5": "gen5", "M5 Pro": "gen5", "M5 Max": "gen5", "M5 Ultra": "gen5",
}


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def _system_profiler_json(data_type: str) -> dict:
    raw = _run(["system_profiler", data_type, "-json"])
    if raw:
        return json.loads(raw)
    return {}


def _parse_chip_info(chip_name: str) -> dict:
    stripped = chip_name.removeprefix("Apple ")
    gen = "unknown"
    for prefix, tier in CHIP_TIER.items():
        if stripped.startswith(prefix):
            gen = tier
            break

    neural_cores = NEURAL_ENGINE_CORES.get(chip_name, NEURAL_ENGINE_CORES.get(stripped, 16))
    return {"chip": chip_name, "generation": gen, "neural_engine_cores": neural_cores, "cpu_cores": {}}


def get_hardware() -> dict:
    data = _system_profiler_json("SPHardwareDataType")
    items = data.get("SPHardwareDataType", [])
    if not items:
        return {}
    hw = items[0]
    chip = hw.get("chip_type", "Unknown")
    info = _parse_chip_info(chip)

    raw_cores = hw.get("number_processors", "")
    if raw_cores:
        m = re.search(r"(\d+):(\d+):(\d+):(\d+)", raw_cores)
        if m:
            info["cpu_cores"] = {
                "total": int(m.group(1)),
                "super": int(m.group(2)),
                "efficiency": int(m.group(3)),
                "performance": int(m.group(4)),
            }

    mem_raw = hw.get("physical_memory", "0 GB")
    mem_gb = int(re.search(r"(\d+)", str(mem_raw)).group(1)) if mem_raw else 0

    return {
        "model": hw.get("machine_name", ""),
        "model_identifier": hw.get("machine_model", ""),
        "model_number": hw.get("model_number", ""),
        "serial_number": hw.get("serial_number", ""),
        "chip": info,
        "total_memory_gb": mem_gb,
        "memory_type": "",
        "boot_rom_version": hw.get("boot_rom_version", ""),
        "activation_lock": hw.get("activation_lock_status", ""),
    }


def get_gpu() -> dict:
    data = _system_profiler_json("SPDisplaysDataType")
    gpu_data = data.get("SPDisplaysDataType", [])
    if not gpu_data:
        return {"gpu_cores": 0, "vendor": "", "metal_support": "", "model": "", "connection_type": "integrated"}
    gpu = gpu_data[0]
    return {
        "gpu_cores": int(gpu.get("sppci_cores", "0")),
        "vendor": gpu.get("sppci_vendor", "").replace("sppci_vendor_", ""),
        "metal_support": gpu.get("spdisplays_mtlgpufamilysupport", "").replace("spdisplays_metal", "Metal "),
        "model": gpu.get("sppci_model", ""),
        "connection_type": "integrated",
    }


def get_memory() -> dict:
    data = _system_profiler_json("SPMemoryDataType")
    items = data.get("SPMemoryDataType", [])
    mem_type = ""
    manufacturer = ""
    if items:
        mem = items[0]
        mem_type = mem.get("dimm_type", "")
        manufacturer = mem.get("dimm_manufacturer", "")

    vm_raw = _run(["vm_stat"])
    page_size = 16384
    wired_pages = active_pages = compressed_pages = free_pages = purgeable_pages = 0

    for line in vm_raw.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().strip('"')
        val = val.strip().rstrip(".")
        try:
            pages = int(val)
        except ValueError:
            continue
        if key == "Pages free":
            free_pages = pages
        elif key == "Pages active":
            active_pages = pages
        elif key == "Pages wired down":
            wired_pages = pages
        elif key == "Pages stored in compressor":
            compressed_pages = pages
        elif key == "Pages purgeable":
            purgeable_pages = pages

    total_mem_bytes = int(_run(["sysctl", "-n", "hw.memsize"]))
    available_gb = free_pages * page_size / (1024**3)
    used_gb = (total_mem_bytes // page_size - free_pages) * page_size / (1024**3)
    wired_gb = wired_pages * page_size / (1024**3)
    compressed_gb = compressed_pages * page_size / (1024**3)

    return {
        "total_gb": total_mem_bytes / (1024**3),
        "available_gb": round(available_gb, 2),
        "used_gb": round(used_gb, 2),
        "wired_down_gb": round(wired_gb, 2),
        "compressed_gb": round(compressed_gb, 2),
        "memory_type": mem_type,
        "manufacturer": manufacturer,
        "page_size_bytes": page_size,
    }


def get_storage() -> list[dict]:
    data = _system_profiler_json("SPStorageDataType")
    volumes = data.get("SPStorageDataType", [])
    result = []
    for v in volumes:
        name = v.get("_name", v.get("volume_name", ""))
        capacity_bytes = v.get("size_in_bytes", 0) or 0
        free_bytes = v.get("free_space_in_bytes", 0) or 0
        pd = v.get("physical_drive", {})
        media = protocol = smart = ""
        if pd and isinstance(pd, dict):
            media = pd.get("medium_type", "")
            protocol = pd.get("protocol", "")
            smart = pd.get("smart_status", "")
        result.append({
            "name": name,
            "capacity_gb": round(capacity_bytes / (1024**3), 2) if capacity_bytes else 0,
            "free_gb": round(free_bytes / (1024**3), 2) if free_bytes else 0,
            "free_pct": round(free_bytes / capacity_bytes * 100, 1) if capacity_bytes else 0,
            "media_type": media,
            "protocol": protocol,
            "smart_status": smart,
        })
    return result


def get_display() -> dict:
    data = _system_profiler_json("SPDisplaysDataType")
    gpu_data = data.get("SPDisplaysDataType", [])
    if not gpu_data:
        return {}
    displays = gpu_data[0].get("spdisplays_ndrvs", [])
    if not displays:
        return {}
    d = displays[0]
    return {
        "model": d.get("_name", ""),
        "resolution": d.get("_spdisplays_pixels", ""),
        "refresh_rate": d.get("_spdisplays_resolution", ""),
        "type": d.get("spdisplays_display_type", "").replace("spdisplays_", ""),
        "main": d.get("spdisplays_main", "") == "spdisplays_yes",
        "internal": d.get("spdisplays_connection_type", "") == "spdisplays_internal",
        "ambient_brightness": d.get("spdisplays_ambient_brightness", "") == "spdisplays_yes",
    }


def get_os() -> dict:
    raw = _run(["sw_vers"])
    version = build = ""
    for line in raw.splitlines():
        if line.startswith("ProductVersion:"):
            version = line.split("\t")[-1].strip()
        elif line.startswith("BuildVersion:"):
            build = line.split("\t")[-1].strip()
    return {"name": "macOS", "version": version, "build": build}


def get_cpu_extra() -> dict:
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return {
        "brand_string": brand,
        "logical_cores": int(_run(["sysctl", "-n", "hw.ncpu"]) or "0"),
        "physical_cores": int(_run(["sysctl", "-n", "hw.physicalcpu"]) or "0"),
    }


def get_power() -> dict:
    health = _run(["system_profiler", "SPPowerDataType", "-json"])
    if not health:
        return {}
    data = json.loads(health)
    items = data.get("SPPowerDataType", [])
    if not items:
        return {}
    p = items[0]
    health_info = p.get("sppower_battery_health_info", {})
    charge_info = p.get("sppower_battery_charge_info", {})
    model_info = p.get("sppower_battery_model_info", {})
    return {
        "battery_health": health_info.get("sppower_battery_health", ""),
        "max_capacity": health_info.get("sppower_battery_health_maximum_capacity", ""),
        "cycle_count": health_info.get("sppower_battery_cycle_count", 0),
        "is_charging": charge_info.get("sppower_battery_is_charging", "FALSE") == "TRUE",
        "charge_pct": charge_info.get("sppower_battery_state_of_charge", 0),
        "battery_model": model_info.get("sppower_battery_device_name", ""),
        "firmware_version": model_info.get("sppower_battery_firmware_version", ""),
    }


def ai_capability_score(hw: dict, gpu: dict, mem: dict) -> dict:
    chip = hw.get("chip", {})
    chip_name = chip.get("chip", "Unknown")
    gen = chip.get("generation", "unknown")

    gpu_score = min(10, round(gpu.get("gpu_cores", 0) / 3))
    total_gb = mem.get("total_gb", 0)

    if total_gb >= 128:
        mem_score = 10
    elif total_gb >= 96:
        mem_score = 9
    elif total_gb >= 64:
        mem_score = 8
    elif total_gb >= 48:
        mem_score = 7
    elif total_gb >= 36:
        mem_score = 6
    elif total_gb >= 24:
        mem_score = 5
    elif total_gb >= 16:
        mem_score = 4
    else:
        mem_score = 2

    neural_cores = chip.get("neural_engine_cores", 0)
    neural_score = min(10, neural_cores // 2)
    composite = round(gpu_score * 0.35 + mem_score * 0.40 + neural_score * 0.25, 1)

    templates = {
        "gen1": "{chip} — capable of 7B-13B GGUF models (quantized)",
        "gen2": "{chip} — capable of 13B-30B GGUF models (quantized)",
        "gen3": "{chip} — capable of 13B-70B GGUF models (quantized)",
        "gen4": "{chip} — capable of 14B-70B GGUF models (quantized), strong NPU acceleration",
        "gen5": "{chip} — capable of 14B-70B+ GGUF models (quantized), best NPU in lineup",
    }
    template = templates.get(gen, "{chip} — chip generation unrecognized")

    return {
        "gpu_score": gpu_score,
        "memory_score": mem_score,
        "neural_engine_score": neural_score,
        "composite_score": composite,
        "max_composite": 10,
        "interpretation": template.format(chip=chip_name),
    }


def scan_hardware() -> dict:
    """Run full hardware scan and return structured result."""
    hw = get_hardware()
    if not hw:
        return {"error": "Unable to read hardware info. This tool requires macOS."}

    gpu = get_gpu()
    mem = get_memory()
    storage = get_storage()
    display = get_display()
    os_info = get_os()
    cpu_extra = get_cpu_extra()
    power = get_power()
    ai_score = ai_capability_score(hw, gpu, mem)

    hw["chip"]["gpu_cores"] = gpu.get("gpu_cores", 0)

    return {
        "hardware": hw,
        "gpu": gpu,
        "memory": mem,
        "storage": storage,
        "display": display,
        "os": os_info,
        "cpu_extra": cpu_extra,
        "power": power if power else None,
        "ai_capability": ai_score,
    }
