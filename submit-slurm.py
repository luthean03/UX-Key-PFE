#!/usr/bin/python

import base64
import os
import sys
import subprocess
import tempfile
import shlex


def makejob(commit_id, config_b64, nruns, command, extra_args: str = ""):
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

# Copie du code + dataset sur le noeud local pour I/O rapide
rsync -r --exclude logslurms --exclude configs --exclude archetypes --exclude archetypes_png --exclude samir_lom . $TMPDIR/code

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

# Installe le projet + scikit-learn/matplotlib/etc via pyproject.toml
python -m pip install .

# === MODIF 2 : Gestion Mémoire VAE ===
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

# Patch data dir (Utilise dataset local sur noeud si disponible)
data_cfg = cfg.get('data') or dict()
data_dir = data_cfg.get('data_dir')
if data_dir:
    dd_path = pathlib.Path(data_dir).expanduser()
    # Priorité 1: Dataset local sur le noeud ($TMPDIR/code/vae_dataset)
    local_dataset = pathlib.Path(os.environ['TMPDIR']) / 'code' / 'vae_dataset'
    if local_dataset.exists():
        data_cfg['data_dir'] = str(local_dataset)
        print(f"✅ Using local dataset on node: {{local_dataset}}")
    # Priorité 2: Chemin absolu (shared filesystem - plus lent)
    elif dd_path.is_absolute():
        data_cfg['data_dir'] = str(dd_path)
        print(f"⚠️  Using shared filesystem (slower): {{dd_path}}")
    # Priorité 3: Chemin relatif depuis le dossier de lancement
    else:
        data_cfg['data_dir'] = str(base / dd_path)
        print(f"⚠️  Using shared filesystem (slower): {{base / dd_path}}")
    cfg['data'] = data_cfg

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
    # Lancement avec la config patchée
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
        "Usage : {} config.yaml [nruns|1] [train|test|train_test] [extra args...]\n"
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
        if remaining and remaining[0] in {"train", "test", "train_test"}:
            command = remaining.pop(0)
    elif remaining[0] in {"train", "test", "train_test"}:
        command = remaining.pop(0)
        if remaining and remaining[0].isdigit():
            nruns = int(remaining.pop(0))

if command not in {"train", "test", "train_test"}:
    raise ValueError("command must be one of: train, test, train_test")

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

job = makejob(commit_id, config_b64, nruns, command, extra_args=extra_args)
if dry_run:
    print(job)
    sys.exit(0)

submit_job(job)