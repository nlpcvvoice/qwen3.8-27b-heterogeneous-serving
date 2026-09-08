import os
import json
import subprocess
import sys
import time

lines = []
lines.append("== probe start ==")
lines.append("hostname: " + subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip())

env_keys = ["TPU_ACCELERATOR_TYPE", "TPU_WORKER_HOSTNAMES", "TPU_NUM_DEVICES",
            "TPU_TYPE", "PJRT_DEVICE", "XRT_TPU_CONFIG", "KAGGLE_KERNEL_RUN_TYPE",
            "TPU_DEVICE_NAMES", "LIBTPU_INIT_ARGS", "GRPC_MAX_RECV",
            "TPU_VISIBLE_DEVICES", "ACCELERATOR_TYPE"]
for k in env_keys:
    lines.append(f"env {k}={os.environ.get(k)!r}")

tpu_env = [k for k, v in os.environ.items() if "TPU" in k or "PJRT" in k or "ACCEL" in k]
lines.append("tpu-ish env vars: " + json.dumps(sorted(tpu_env)))
lines.append("KAGGLE-ish env vars: " + json.dumps(sorted(k for k in os.environ if k.startswith("KAGGLE"))))

dev = subprocess.run(["bash", "-lc", "ls -la /dev | grep -Ei 'vfio|accel|tpu|dri|pcm'"],
                     capture_output=True, text=True)
lines.append("ls -la /dev (accel/tpu/vfio):\n" + (dev.stdout or "") + (dev.stderr or ""))
for p in ["/dev/vfio", "/dev/accel0", "/dev/accel1", "/dev/accel2", "/dev/accel3",
          "/dev/accel4", "/dev/accel5", "/dev/accel6", "/dev/accel7", "/sys/bus/pci/devices"]:
    lines.append(f"exists {p}={os.path.exists(p)}")

try:
    import jax
    lines.append("jax version: " + jax.__version__)
    lines.append("jax.default_backend: " + str(jax.default_backend()))
    lines.append("jax.device_count: " + str(jax.device_count()))
    devs = jax.devices()
    lines.append("jax.devices: " + str(devs)[:2000])
except Exception as e:
    lines.append("jax error: " + repr(e))

sys.stdout.write("\n".join(lines) + "\n")
with open("/kaggle/working/probe_result.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("== probe done, wrote /kaggle/working/probe_result.txt ==")