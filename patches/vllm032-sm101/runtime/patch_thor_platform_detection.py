#!/usr/bin/env python3
from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()
old = """        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            )
"""
new = """        def cuda_is_jetson() -> bool:
            if os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            ):
                return True
            try:
                with open("/proc/device-tree/compatible", "rb") as stream:
                    return b"nvidia,tegra" in stream.read()
            except OSError:
                return False
"""
if source.count(old) != 1:
    raise RuntimeError("unexpected vLLM platform detection layout")
path.write_text(source.replace(old, new))
