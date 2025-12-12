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
        
        # Construct our replacement block with correct indentation
        indent = line[:line.find("if")]
        new_lines.append(line) # Keep the if
        
        # Add our new logic
        new_lines.append(f'{indent}    # Patched by grpo-trader')
        new_lines.append(f'{indent}    if param.device_mesh.size() == 1:')
        new_lines.append(f'{indent}        print(f"[DEBUG] Patch active: Skipping redistribute for single-device mesh")')
        new_lines.append(f'{indent}        param = param.to_local()')
        new_lines.append(f'{indent}    else:')
        new_lines.append(f'{indent}        param = param.redistribute(')
        new_lines.append(f'{indent}            placements=[Replicate()] * param.device_mesh.ndim,')
        new_lines.append(f'{indent}            async_op=True,')
        new_lines.append(f'{indent}        ).to_local()')
        
        # Skip the original lines until we pass the .to_local() call
        # The original block was:
        #     param = param.redistribute(
        #         placements=[Replicate()] * param.device_mesh.ndim,
        #         async_op=True,
        #     ).to_local()
        
        # We skip lines until we see .to_local()
        j = i + 1
        while j < len(lines):
            if ".to_local()" in lines[j]:
                i = j + 1 # Continue after this line
                found = True
                break
            j += 1
        
        if not found:
             # Fallback if we couldn't find the end of the block, just append the rest
             print("Warning: Could not find end of block (.to_local()), aborting patch for this block")
             new_lines.append(lines[i+1]) # Just append the next line and continue normal processing
             i += 1
    else:
        new_lines.append(line)
        i += 1

if found:
    with open(file_path, "w") as f:
        f.write("\n".join(new_lines))
    print("Successfully patched update_weight_utils.py with robust method")
else:
    print("Target code block not found. Dumping file content for debugging:")
    print(content)
