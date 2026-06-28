"""
Check which scaled_dot_product_attention backend is actually usable on this
GPU. Memory-efficient/Flash backends need compute capability >= 8.0 (Ampere+);
older cards (e.g. Turing/RTX 8000, compute capability 7.5) fall back to the
naive "math" kernel, which materializes the full attention matrix and uses
far more memory for the same batch size and sequence length.

Run on a GPU node:
    python3 experiments/euclid-cosmos/check_sdpa_backend.py
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    print("Device:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print()
    print("Global backend flags:")
    print("  flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("  mem_efficient_sdp_enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("  math_sdp_enabled:", torch.backends.cuda.math_sdp_enabled())
    print()

    q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)

    backends = [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]
    for name, backend in backends:
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, k, v)
            print(f"{name}: SUPPORTED")
        except Exception as e:
            print(f"{name}: NOT SUPPORTED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
