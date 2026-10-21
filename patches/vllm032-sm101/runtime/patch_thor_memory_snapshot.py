#!/usr/bin/env python3
"""Keep CUDA's reserved-pool accounting on DRIVE Thor."""
from pathlib import Path
import sys

OLD = "            self.free_memory = psutil.virtual_memory().available\n"
NEW = '''            # DRIVE Thor reserves hugetlb RAM for CUDA; host available
            # excludes that pool and must not replace CUDA's free count.
            thor_pool = False
            if torch.cuda.get_device_properties(device).name == "Thor":
                with open("/proc/meminfo") as stream:
                    for line in stream:
                        if line.startswith("Hugetlb:"):
                            thor_pool = int(line.split()[1]) * 1024 >= self.total_memory
                            break
            if not thor_pool:
                self.free_memory = psutil.virtual_memory().available
'''


def patch(path):
    source = path.read_text()
    if source.count(OLD) != 1 or "thor_pool = False" in source:
        raise RuntimeError("unexpected MemorySnapshot layout or already patched")
    path.write_text(source.replace(OLD, NEW))


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
