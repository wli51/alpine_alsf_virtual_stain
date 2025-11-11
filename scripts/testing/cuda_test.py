import os
import time

# for import testing purpose
try:
    import virtual_stain_flow

    # instead of exiting upon error, just print message and move on
except ImportError:
    print("Error: virtual_stain_flow module not found.")
    pass
except Exception as e:
    print("Error during virtual_stain_flow import:", repr(e))
    pass

print("=== Test GPU job starting ===")
print(f"HOSTNAME={os.uname().nodename}")
print(f"SLURM_JOB_ID={os.getenv('SLURM_JOB_ID')}")
print(f"SLURM_JOB_ACCOUNT={os.getenv('SLURM_JOB_ACCOUNT')}")
print(f"SLURM_JOB_PARTITION={os.getenv('SLURM_JOB_PARTITION')}")
print(f"SLURM_JOB_QOS={os.getenv('SLURM_JOB_QOS')}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        dev = torch.device("cuda:0")
        x = torch.randn((1024, 1024), device=dev)
        y = x @ x.t()
        print("Dummy CUDA matmul success. y.mean() =", float(y.mean()))
    else:
        print("No CUDA visible inside job (unexpected for GPU alloc).")
except Exception as e:
    print("Error during CUDA test:", repr(e))

print("Sleeping briefly to keep job alive...")
time.sleep(10)

print("=== Test GPU job completed successfully ===")
