import os

file_path = "Slime/slime/backends/fsdp_utils/update_weight_utils.py"

if not os.path.exists(file_path):
    # Try absolute path in container
    file_path = "/workspace/Slime/slime/backends/fsdp_utils/update_weight_utils.py"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping patch.")
        exit(1)

print(f"Patching {file_path}...")

# Reset the file first to ensure we are patching clean code
# This fixes issues if previous patches corrupted the file
dir_path = os.path.dirname(file_path)
repo_root = os.path.abspath(os.path.join(dir_path, "../../../.."))
print(f"Resetting {file_path} in {repo_root}...")
os.system(f"cd {repo_root} && git checkout slime/backends/fsdp_utils/update_weight_utils.py")

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


# --- Patch sglang_rollout.py ---
print("\n--- Patching sglang_rollout.py ---")
rollout_file = "Slime/slime/rollout/sglang_rollout.py"
if not os.path.exists(rollout_file):
    rollout_file = "/workspace/Slime/slime/rollout/sglang_rollout.py"

if os.path.exists(rollout_file):
    print(f"Patching {rollout_file}...")
    
    # Reset file
    dir_path = os.path.dirname(rollout_file)
    repo_root = os.path.abspath(os.path.join(dir_path, "../../.."))
    print(f"Resetting {rollout_file} in {repo_root}...")
    os.system(f"cd {repo_root} && git checkout slime/rollout/sglang_rollout.py")

    with open(rollout_file, "r") as f:
        lines = f.readlines()
        
    new_lines = []
    patched = False
    
    # We look for:
    # samples = data_source(args.over_sampling_batch_size)
    # state.submit_generate_tasks(samples)
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        if "samples = data_source(args.over_sampling_batch_size)" in line:
            # Inject check after this line
            indent = line[:line.find("samples")]
            check_code = [
                f"{indent}if not samples:\n",
                f"{indent}    print(f'[DEBUG] data_source returned empty samples. remaining={{state.remaining_batch_size}}')\n",
                f"{indent}    print(f'[DEBUG] data_source type: {{type(data_source)}}')\n",
                f"{indent}    try:\n",
                f"{indent}        if hasattr(data_source, '__self__'):\n",
                f"{indent}             ds = data_source.__self__\n",
                f"{indent}             print(f'[DEBUG] data_source.dataset len: {{len(ds) if hasattr(ds, \"__len__\") else \"N/A\"}}')\n",
                f"{indent}             print(f'[DEBUG] data_source.dataset samples: {{len(ds.samples) if hasattr(ds, \"samples\") else \"N/A\"}}')\n",
                f"{indent}             if hasattr(ds, 'samples') and len(ds.samples) > 0:\n",
                f"{indent}                 print(f'[DEBUG] First sample: {{ds.samples[0]}}')\n",
                f"{indent}    except Exception as e: print(f'[DEBUG] Error inspecting data_source: {{e}}')\n",
                f"{indent}    if not state.pendings:\n",
                f"{indent}        raise RuntimeError('Cannot fill batch size: data_source exhausted and no pending tasks. Check your train_data.jsonl!')\n",
                f"{indent}    break\n"
            ]
            new_lines.extend(check_code)
            patched = True
            print("Injected empty sample check.")
            
    if patched:
        with open(rollout_file, "w") as f:
            f.writelines(new_lines)
        print("Successfully patched sglang_rollout.py")
    else:
        print("Warning: Could not find target line in sglang_rollout.py")
else:
    print(f"File not found: {rollout_file}")


# --- Patch slime/utils/data.py ---
print("\n--- Patching slime/utils/data.py ---")
data_file = "Slime/slime/utils/data.py"
if not os.path.exists(data_file):
    data_file = "/workspace/Slime/slime/utils/data.py"

if os.path.exists(data_file):
    print(f"Patching {data_file}...")
    
    # Reset file
    dir_path = os.path.dirname(data_file)
    repo_root = os.path.abspath(os.path.join(dir_path, "../../.."))
    print(f"Resetting {data_file} in {repo_root}...")
    os.system(f"cd {repo_root} && git checkout slime/utils/data.py")

    with open(data_file, "r") as f:
        lines = f.readlines()
        
    new_lines = []
    patched = False
    
    # We look for:
    # if _should_skip_prompt(prompt, tokenizer, processor, max_length, apply_chat_template_kwargs):
    #     continue
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        if "if _should_skip_prompt(prompt, tokenizer, processor, max_length, apply_chat_template_kwargs):" in line:
            # Inject debug print before continue
            indent = line[:line.find("if")]
            # The next line should be 'continue' with one more indent level
            # We will insert print before it
            
            # Actually, we can just insert before the if check to see what's happening
            # But better to insert inside the if block
            pass
            
    # Re-read and do it properly with index
    final_lines = []
    for i, line in enumerate(lines):
        final_lines.append(line)
        if "if _should_skip_prompt(prompt, tokenizer, processor, max_length, apply_chat_template_kwargs):" in line:
            # Look ahead for continue
            if i + 1 < len(lines) and "continue" in lines[i+1]:
                indent = lines[i+1][:lines[i+1].find("continue")]
                final_lines.append(f"{indent}print(f'[DEBUG] Skipped prompt due to length. max_length={{max_length}}')\n")
                patched = True
                
    if patched:
        with open(data_file, "w") as f:
            f.writelines(final_lines)
        print("Successfully patched slime/utils/data.py")
    else:
        print("Warning: Could not find target line in slime/utils/data.py")
else:
    print(f"File not found: {data_file}")

