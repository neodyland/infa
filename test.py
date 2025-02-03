import torch
import time

def main():
    a = torch.full((2, 48, 32), 2.0, dtype=torch.bfloat16, device="cuda")
    b = torch.full((2, 32, 16), 3.0, dtype=torch.bfloat16, device="cuda")
    now = time.time()
    for _ in range(1000):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    print("Time: ", time.time() - now)


if __name__ == "__main__":
    main()
