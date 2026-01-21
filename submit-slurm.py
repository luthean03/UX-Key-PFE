#!/usr/bin/python

import base64
import os
import sys
import subprocess
import tempfile
import shlex
import yaml


def makejob(commit_id, config_b64, nruns, command, data_src, archetypes_src, extra_args: str = ""):
    exclude_list = "dani[01-17],tx[00-16],sh[10-19],sh00"
    return f"""#!/bin/bash

#SBATCH --job-name=vae-ux-key
#SBATCH --nodes=1
#SBATCH --exclude={exclude_list}
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

current_dir=`pwd`
export current_dir
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
echo "Running on " $(hostname)

echo "Copying the source directory"
date
mkdir -p $TMPDIR/code

# Copy source code to local node for fast I/O (exclude heavy logs/datasets)
rsync -r --exclude logslurms --exclude configs --exclude archetypes --exclude samir_lom --exclude 'vae_dataset*' . $TMPDIR/code

# Copy archetypes and dataset only for training
if [[ "{command}" == "train" || "{command}" == "train_test" ]]; then
    # Copy archetypes for latent metrics (from YAML config path)
    SRC_ARCHETYPES="{archetypes_src}"
    # Handle relative paths (relative to execution directory)
    if [[ "$SRC_ARCHETYPES" != /* ]]; then
        SRC_ARCHETYPES="$current_dir/$SRC_ARCHETYPES"
    fi
    if [[ -d "$SRC_ARCHETYPES" ]]; then
        echo "[OK] Copying archetypes from $SRC_ARCHETYPES to node..."
        rsync -r "$SRC_ARCHETYPES/" "$TMPDIR/code/archetypes_png/"
        echo "[OK] Archetypes copied: $(ls $TMPDIR/code/archetypes_png | wc -l) files"
    fi

    # Copy training dataset to node (from YAML config path)
    SRC_DATA="{data_src}"
    if [[ "$SRC_DATA" != /* ]]; then
        SRC_DATA="$current_dir/$SRC_DATA"
    fi
    if [[ -d "$SRC_DATA" ]]; then
        echo "[OK] Copying training dataset to node from $SRC_DATA..."
        rsync -r "$SRC_DATA/" "$TMPDIR/code/vae_dataset/"
        echo "[OK] Dataset copied: $(find $TMPDIR/code/vae_dataset -type f | wc -l) files"
    fi
else
    echo "Skipping dataset/archetypes copy (not needed for {command})"
fi

echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/code
git checkout {commit_id}

echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate

# Installation compatible vieux GPU + dépendances
python -m pip install -U pip
python -m pip install 'numpy<2'
python -m pip install --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118

# Install project and dependencies via pyproject.toml
python -m pip install .

# Configure PyTorch memory allocator for improved efficiency
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Recreate the config file inside $TMPDIR
CONFIG_IN="$TMPDIR/code/input_config.yaml"
echo "{config_b64}" | base64 -d > "$CONFIG_IN"

# Patch the config inside the job so that relative paths point to shared filesystem
job_config="$TMPDIR/code/job_config.yaml"
export CONFIG_PATH="$CONFIG_IN"
export JOB_CONFIG="$job_config"

python3 - <<'PY'
import os
import pathlib
import yaml

cfg_path = pathlib.Path(os.environ.get('CONFIG_PATH', '')).expanduser()
cfg = yaml.safe_load(cfg_path.read_text())

base = pathlib.Path(os.environ['current_dir'])

# Patch logging dir
logging_cfg = cfg.get('logging') or dict()
logdir = logging_cfg.get('logdir')
if logdir:
    logdir_path = pathlib.Path(logdir).expanduser()
    if not logdir_path.is_absolute():
        logging_cfg['logdir'] = str(base / logdir_path)
        cfg['logging'] = logging_cfg

# Patch data dir (use local node dataset if available)
data_cfg = cfg.get('data') or dict()
data_dir = data_cfg.get('data_dir')
if data_dir:
    dd_path = pathlib.Path(data_dir).expanduser()
    # Priority 1: Local dataset on node ($TMPDIR/code/vae_dataset)
    local_dataset = pathlib.Path(os.environ['TMPDIR']) / 'code' / 'vae_dataset'
    if local_dataset.exists():
        data_cfg['data_dir'] = str(local_dataset)
        print(f"[OK] Using local dataset on node: {{local_dataset}}")
    # Priority 2: Absolute path (shared filesystem - slower)
    elif dd_path.is_absolute():
        data_cfg['data_dir'] = str(dd_path)
        print(f"[!] Using shared filesystem (slower): {{dd_path}}")
    # Priority 3: Relative path from launch directory
    else:
        data_cfg['data_dir'] = str(base / dd_path)
        print(f"[!] Using shared filesystem (slower): {{base / dd_path}}")
    cfg['data'] = data_cfg

# Patch archetypes dir (same logic as dataset)
archetypes_dir = data_cfg.get('archetypes_dir')
if archetypes_dir:
    arch_path = pathlib.Path(archetypes_dir).expanduser()
    # Priority 1: Local archetypes on node
    local_archetypes = pathlib.Path(os.environ['TMPDIR']) / 'code' / 'archetypes_png'
    if local_archetypes.exists():
        data_cfg['archetypes_dir'] = str(local_archetypes)
        print(f"[OK] Using local archetypes on node: {{local_archetypes}}")
    # Priority 2: Absolute path (shared filesystem)
    elif arch_path.is_absolute():
        data_cfg['archetypes_dir'] = str(arch_path)
        print(f"[!] Using shared filesystem archetypes: {{arch_path}}")
    # Priority 3: Relative path
    else:
        data_cfg['archetypes_dir'] = str(base / arch_path)
        print(f"[!] Using shared filesystem archetypes: {{base / arch_path}}")
    cfg['data'] = data_cfg

# Patch resume checkpoint path
resume_path = cfg.get('resume')
if resume_path:
    r_path = pathlib.Path(resume_path).expanduser()
    if not r_path.is_absolute():
        cfg['resume'] = str(base / r_path)
        print(f"[OK] Patched resume path to absolute: {{cfg['resume']}}")

# Patch test checkpoint
test_cfg = cfg.get('test') or dict()
ckpt = test_cfg.get('checkpoint')
if ckpt:
    ckpt_path = pathlib.Path(ckpt).expanduser()
    if not ckpt_path.is_absolute():
        test_cfg['checkpoint'] = str(base / ckpt_path)

cfg['test'] = test_cfg

pathlib.Path(os.environ['JOB_CONFIG']).write_text(yaml.safe_dump(cfg))
PY

if [[ "{command}" == "train" || "{command}" == "train_test" ]]; then
    echo "Training"
    # Launch with patched config
    python3 -m torchtmpl.main "$job_config" train

    if [[ $? != 0 ]]; then
        exit -1
    fi
fi

if [[ "{command}" == "test" || "{command}" == "train_test" ]]; then
    echo "Inference (test)"
    python3 -m torchtmpl.main "$job_config" test {extra_args}

    if [[ $? != 0 ]]; then
        exit -1
    fi
    
    # Copy test results back to shared filesystem
    # Extract test output directory from config
    TEST_OUTPUT_DIR=$(python3 -c "
import yaml
import pathlib
cfg = yaml.safe_load(open('$job_config'))
test_output = cfg.get('test', {{}}).get('test_output_dir', './test_output')
print(test_output)
")
    
    # Copy results from compute node to shared directory
    if [[ -d "$TEST_OUTPUT_DIR" ]]; then
        echo "[OK] Copying test results back to shared filesystem..."
        # Create destination folder if not exists
        mkdir -p "$current_dir/$TEST_OUTPUT_DIR"
        # Copy comparison PNG files
        rsync -r "$TEST_OUTPUT_DIR/" "$current_dir/$TEST_OUTPUT_DIR/" 2>/dev/null || true
        echo "[OK] Test results copied to: $current_dir/$TEST_OUTPUT_DIR"
        echo "   Found $(ls $current_dir/$TEST_OUTPUT_DIR/*.png 2>/dev/null | wc -l) comparison images"
    fi
fi

if [[ "{command}" == "interpolate" || "{command}" == "train_test" ]]; then
    echo "Interpolation"
    python3 -m torchtmpl.main "$job_config" interpolate {extra_args}

    if [[ $? != 0 ]]; then
        exit -1
    fi
    
    # Copy interpolation results back to shared filesystem
    # Extract interpolation output directory from config
    INTERP_OUTPUT_DIR=$(python3 -c "
import yaml
import pathlib
cfg = yaml.safe_load(open('$job_config'))
interp_output = cfg.get('interpolate', {{}}).get('output_dir', './interpolate_output')
print(interp_output)
")
    
    # Copy results from compute node to shared directory
    if [[ -d "$INTERP_OUTPUT_DIR" ]]; then
        echo "[OK] Copying interpolation results back to shared filesystem..."
        # Create destination folder if not exists
        mkdir -p "$current_dir/$INTERP_OUTPUT_DIR"
        # Copy interpolation PNG files
        rsync -r "$INTERP_OUTPUT_DIR/" "$current_dir/$INTERP_OUTPUT_DIR/" 2>/dev/null || true
        echo "[OK] Interpolation results copied to: $current_dir/$INTERP_OUTPUT_DIR"
        echo "   Found $(ls $current_dir/$INTERP_OUTPUT_DIR/*.png 2>/dev/null | wc -l) interpolation grids"
    fi
fi

echo "Done. Artifacts are written to the paths specified in the YAML (under $current_dir)."
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


def _pop_flag(argv, flag: str) -> bool:
    present = False
    while flag in argv:
        argv.remove(flag)
        present = True
    return present


def _print_usage_and_exit(prog: str) -> None:
    print(
        "Usage : {} config.yaml [nruns|1] [train|test|train_test|interpolate] [extra args...]\n"
        "Optional flags:\n"
        "  --dry-run       Print the generated sbatch script and exit\n"
        "  --allow-dirty   Allow uncommitted changes".format(
            prog
        )
    )
    sys.exit(-1)


argv = sys.argv[1:]
dry_run = _pop_flag(argv, "--dry-run")
allow_dirty = _pop_flag(argv, "--allow-dirty")

if len(argv) < 1:
    _print_usage_and_exit(sys.argv[0])

remaining = list(argv)
configpath = os.path.abspath(remaining.pop(0))

nruns = 1
command = "train"

if remaining:
    if remaining[0].isdigit():
        nruns = int(remaining.pop(0))
        if remaining and remaining[0] in {"train", "test", "train_test", "interpolate"}:
            command = remaining.pop(0)
    elif remaining[0] in {"train", "test", "train_test", "interpolate"}:
        command = remaining.pop(0)
        if remaining and remaining[0].isdigit():
            nruns = int(remaining.pop(0))

if command not in {"train", "test", "train_test", "interpolate"}:
    raise ValueError("command must be one of: train, test, train_test, interpolate")

if remaining and remaining[0] == "--":
    remaining.pop(0)
extra_args = " ".join(shlex.quote(a) for a in remaining)

# Verification Git
result = int(
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0 and not (allow_dirty or dry_run):
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError("You must stage and commit every modification before submission")

try:
    commit_id = subprocess.check_output("git log --pretty=format:'%H' -n 1", shell=True).decode()
except:
    commit_id = "unknown"
print(f"I will be using the commit id {commit_id}")

if result > 0 and (allow_dirty or dry_run):
    print("WARNING: repository is dirty; local changes will NOT be included in the job.")

os.system("mkdir -p logslurms")

with open(configpath, "rb") as fp:
    config_b64 = base64.b64encode(fp.read()).decode("ascii")

# Parse yaml to extract paths for rsync (avoid hardcoded paths)
with open(configpath, "r") as fp:
    try:
        cfg_rsync = yaml.safe_load(fp)
        data_src = cfg_rsync.get('data', {}).get('data_dir', 'vae_dataset_scaled')
        archetypes_src = cfg_rsync.get('data', {}).get('archetypes_dir', 'archetypes_png')
    except Exception as e:
        print(f"Warning: Could not parse paths from yaml ({e}), using default fallback.")
        data_src = 'vae_dataset_scaled'
        archetypes_src = 'archetypes_png'

job = makejob(commit_id, config_b64, nruns, command, data_src, archetypes_src, extra_args=extra_args)
if dry_run:
    print(job)
    sys.exit(0)

submit_job(job)