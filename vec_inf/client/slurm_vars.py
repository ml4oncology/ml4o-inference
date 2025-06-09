"""Slurm cluster configuration variables."""

from pathlib import Path

from typing_extensions import Literal


CACHED_CONFIG = Path("/", "model-weights", "vec-inf-shared", "models_latest.yaml")
LD_LIBRARY_PATH = "/scratch/ssd001/pkgs/cudnn-11.7-v8.5.0.96/lib/:/scratch/ssd001/pkgs/cuda-11.7/targets/x86_64-linux/lib/"
SINGULARITY_IMAGE = "/cluster/projects/gliugroup/2BLAST/containers/vec-inf-image-2025-05-15.sif"
SINGULARITY_LOAD_CMD = "module load singularity/3.11.0"
VLLM_NCCL_SO_PATH = "/vec-inf/nccl/libnccl.so.2.18.1"
LLM_PATH = "/cluster/projects/gliugroup/2BLAST/LLMs" # Path to all the tokenizers, model weights, etc.
HOME_PATH = Path("~/").expanduser()
MAX_GPUS_PER_NODE = 8
MAX_NUM_NODES = 16
MAX_CPUS_PER_TASK = 128

# Quality of Service (QoS) options for Slurm jobs (I don't think H4H supports these, so just ignore them)
QOS = Literal[
    "normal",
    "m",
    "m2",
    "m3",
    "m4",
    "m5",
    "long",
    "deadline",
    "high",
    "scavenger",
    "llm",
    "a100",
]

PARTITION = Literal[
    "gpu"
]

DEFAULT_ARGS = {
    "cpus_per_task": 16,
    "mem_per_node": "64G",
    "qos": "m2",
    "time": "08:00:00",
    "partition": "gpu",
    "data_type": "auto",
    "log_dir": f"{HOME_PATH}/.vec-inf-logs",
    "model_weights_parent_dir": LLM_PATH,
}
