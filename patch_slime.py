import os

file_path = "Slime/slime/backends/fsdp_utils/update_weight_utils.py"

if not os.path.exists(file_path):
    # Try absolute path in container
    file_path = "/workspace/Slime/slime/backends/fsdp_utils/update_weight_utils.py"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping patch.")
        exit(1)

print(f"Patching {file_path}...")

with open(file_path, "r") as f:
    lines = f.readlines()

# 1. Inject helper function
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

# Check if already patched
content = "".join(lines)
if "_safe_redistribute" in content:
    print("Helper function already present.")
else:
    # Insert before class UpdateWeight
    new_lines = []
    inserted = False
    for line in lines:
        if "class UpdateWeight" in line and not inserted:
            new_lines.append(helper_func + "\n")
            inserted = True
        new_lines.append(line)
    lines = new_lines
    print("Injected helper function.")

# 2. Replace the block
# We scan for 'if isinstance(param, DTensor):'
# Then we look for 'param = param.redistribute'
# And replace that statement with 'param = _safe_redistribute(param)'

final_lines = []
skip_mode = False
patched_block = False

i = 0
while i < len(lines):
    line = lines[i]
    
    if "if isinstance(param, DTensor):" in line:
        final_lines.append(line)
        # We are entering the block.
        # We expect the next lines to contain the redistribute call.
        # We will consume lines until we see the end of the redistribute call (ending in .to_local())
        
        # Get indentation
        indent = line[:line.find("if")]
        # Assume standard 4-space indent increase
        inner_indent = indent + "    "
        
        # Look ahead to find the redistribute call
        j = i + 1
        found_redistribute = False
        while j < len(lines) and j < i + 10: # Look ahead a bit
            if "param = param.redistribute" in lines[j]:
                found_redistribute = True
                break
            if "bucket.append" in lines[j]: # End of block
                break
            j += 1
            
        if found_redistribute:
            print(f"Found redistribute call at line {j+1}")
            # Keep lines between i and j (e.g. comments)
            for k in range(i+1, j):
                final_lines.append(lines[k])
            
            # Insert our fix
            final_lines.append(f"{inner_indent}param = _safe_redistribute(param)\n")
            
            # Skip the original redistribute lines
            # It starts at j. We need to find where it ends.
            # It ends with .to_local()
            k = j
            while k < len(lines):
                if ".to_local()" in lines[k]:
                    break
                k += 1
            
            # k is the line with .to_local(). We skip it too.
            i = k + 1
            patched_block = True
            continue
        else:
            print("Warning: Found 'if isinstance' but not 'redistribute' call nearby.")
            
    final_lines.append(line)
    i += 1

with open(file_path, "w") as f:
    f.writelines(final_lines)

if patched_block:
    print("Successfully patched update_weight_utils.py using line scanner.")
    
    # Verify
    print("--- Verification ---")
    with open(file_path, "r") as f:
        for line in f:
            if "_safe_redistribute(param)" in line:
                print(f"Found call: {line.rstrip()}")
    print("--------------------")
else:
    print("Warning: Did not find block to patch.")
    # Dump relevant section
    print("--- Dump of relevant section ---")
    for line in lines:
        if "DTensor" in line or "redistribute" in line:
            print(line.rstrip())

