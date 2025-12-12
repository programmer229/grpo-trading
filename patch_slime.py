import os
import re

file_path = "Slime/slime/backends/fsdp_utils/update_weight_utils.py"

if not os.path.exists(file_path):
    # Try absolute path in container
    file_path = "/workspace/Slime/slime/backends/fsdp_utils/update_weight_utils.py"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping patch.")
        exit(1)

print(f"Patching {file_path}...")

with open(file_path, "r") as f:
    content = f.read()

# 1. Inject helper function if not present
helper_func = """
def _safe_redistribute(param):
    # Patched by grpo-trader
    if param.device_mesh.size() == 1:
        print(f"[DEBUG] Patch active: Skipping redistribute for single-device mesh")
        return param.to_local()
    return param.redistribute(
        placements=[Replicate()] * param.device_mesh.ndim,
        async_op=True,
    ).to_local()
"""

if "_safe_redistribute" not in content:
    # Insert after imports (look for the last import or the logger definition)
    # A safe place is before "class UpdateWeight"
    if "class UpdateWeight" in content:
        content = content.replace("class UpdateWeight", f"{helper_func}\n\nclass UpdateWeight")
        print("Injected helper function.")
    else:
        print("Could not find 'class UpdateWeight' insertion point.")
        exit(1)

# 2. Replace the problematic block with the helper call
# We look for the specific block structure
# The block is:
# param = param.redistribute(
#     placements=[Replicate()] * param.device_mesh.ndim,
#     async_op=True,
# ).to_local()

# We'll use a regex to be robust against whitespace
pattern = r"param\s*=\s*param\.redistribute\s*\(\s*placements=\[Replicate\(\)\]\s*\*\s*param\.device_mesh\.ndim,\s*async_op=True,\s*\)\.to_local\(\)"

match = re.search(pattern, content, re.DOTALL)
if match:
    print("Found problematic block.")
    # We need to preserve the indentation of the match
    start_idx = match.start()
    # Find the start of the line to get indentation
    line_start = content.rfind('\n', 0, start_idx) + 1
    indent = content[line_start:start_idx]
    
    # Construct replacement
    replacement = f"param = _safe_redistribute(param)"
    
    # Replace
    content = content[:start_idx] + replacement + content[match.end():]
    print("Replaced block with helper call.")
    
    with open(file_path, "w") as f:
        f.write(content)
    print("Successfully patched update_weight_utils.py")
    
    # Verify
    print("--- Verification (grep _safe_redistribute) ---")
    with open(file_path, "r") as f:
        for line in f:
            if "_safe_redistribute" in line:
                print(line.rstrip())
    print("--------------------------------------------")

else:
    if "_safe_redistribute(param)" in content:
        print("Patch already applied (helper call found).")
    else:
        print("Could not find problematic block via regex. Dumping snippet for debug:")
        idx = content.find("redistribute")
        if idx != -1:
            print(content[idx-50:idx+200])
        else:
            print("redistribute not found in file.")
