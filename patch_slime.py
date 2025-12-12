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

# The code block to replace
target_code = """            if isinstance(param, DTensor):
                # async version of param.full_tensor
                param = param.redistribute(
                    placements=[Replicate()] * param.device_mesh.ndim,
                    async_op=True,
                ).to_local()"""

# The replacement code
replacement_code = """            if isinstance(param, DTensor):
                # async version of param.full_tensor
                if param.device_mesh.size() == 1:
                    param = param.to_local()
                else:
                    param = param.redistribute(
                        placements=[Replicate()] * param.device_mesh.ndim,
                        async_op=True,
                    ).to_local()"""

if target_code in content:
    new_content = content.replace(target_code, replacement_code)
    with open(file_path, "w") as f:
        f.write(new_content)
    print("Successfully patched update_weight_utils.py")
else:
    print("Target code not found. Already patched or content changed.")
    # Debug: print a snippet to see what's there
    start_idx = content.find("if isinstance(param, DTensor):")
    if start_idx != -1:
        print("Found snippet:")
        print(content[start_idx:start_idx+300])
    else:
        print("Could not find 'if isinstance(param, DTensor):'")
