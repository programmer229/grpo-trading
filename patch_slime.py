import os

file_path = "Slime/slime/backends/fsdp_utils/update_weight_utils.py"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    # Try absolute path in container
    file_path = "/workspace/Slime/slime/backends/fsdp_utils/update_weight_utils.py"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping patch.")
        exit(1)

print(f"Patching {file_path}...")

with open(file_path, "r") as f:
    content = f.read()

# Robust patcher
found = False
lines = content.splitlines()
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if "if isinstance(param, DTensor):" in line:
        # Found the start of the block
        print(f"Found target block at line {i+1}")
        # Check if it looks like the block we want to replace
        # We expect the next few lines to be the redistribute call
        # We will replace this entire block with our logic
        
        # Construct our replacement block with HARDCODED indentation (12 spaces)
        # This avoids issues with mixed tabs/spaces or incorrect indent detection
        base_indent = " " * 12
        
        new_lines.append(line) # Keep the 'if isinstance(param, DTensor):'
        
        # Add our new logic
        new_lines.append(f'{base_indent}    # Patched by grpo-trader')
        new_lines.append(f'{base_indent}    if param.device_mesh.size() == 1:')
        new_lines.append(f'{base_indent}        print(f"[DEBUG] Patch active: Skipping redistribute for single-device mesh")')
        new_lines.append(f'{base_indent}        param = param.to_local()')
        new_lines.append(f'{base_indent}    else:')
        new_lines.append(f'{base_indent}        param = param.redistribute(')
        new_lines.append(f'{base_indent}            placements=[Replicate()] * param.device_mesh.ndim,')
        new_lines.append(f'{base_indent}            async_op=True,')
        new_lines.append(f'{base_indent}        ).to_local()')
        
        # Skip the original lines until we pass the .to_local() call
        j = i + 1
        while j < len(lines):
            if ".to_local()" in lines[j]:
                i = j + 1 # Continue after this line
                found = True
                break
            j += 1
        
        if not found:
             print("Warning: Could not find end of block (.to_local()), aborting patch for this block")
             new_lines.append(lines[i+1]) 
             i += 1
    else:
        new_lines.append(line)
        i += 1

if found:
    with open(file_path, "w") as f:
        f.write("\n".join(new_lines))
    print("Successfully patched update_weight_utils.py with hardcoded indentation")
    
    # Verify by printing the patched lines
    print("--- Verifying patched content (lines 60-80) ---")
    with open(file_path, "r") as f:
        patched_lines = f.readlines()
        for k in range(max(0, 60), min(len(patched_lines), 80)):
            print(f"{k+1}: {patched_lines[k].rstrip()}")
    print("---------------------------------------------")
else:
    print("Target code block not found. Dumping file content for debugging:")
    print(content)
