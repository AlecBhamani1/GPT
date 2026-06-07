You will not be able to run this on a macbook or something like that. Even on a spark it takes a long time (ie > 15 min).

This is a decoder only transformer.

we use the lower triangular mask to decode.

This is ~10 million parameter model trained on the linux code text file.

## Setup (any computer)

This project needs **Python 3.9+** and **PyTorch**. The recommended way to make sure
any machine has the right packages is a virtual environment + `requirements.txt`.

```bash
# 1. Create an isolated environment (so you don't touch the system Python)
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell/cmd)

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Run it
python DecoderOnlyLinuxGPT.py
```

When you're done, run `deactivate` to leave the environment.

### GPU note

`pip install -r requirements.txt` installs a build of PyTorch that works on CPU
everywhere, and on CUDA GPUs on Linux/Windows. If a machine has an NVIDIA GPU and
you want a build matched to its CUDA version, follow the selector at
https://pytorch.org/get-started/locally/ instead. The script auto-detects the GPU
via `torch.cuda.is_available()`, so no code changes are needed.
