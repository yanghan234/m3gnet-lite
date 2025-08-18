import platform
import time

import torch
from loguru import logger


def benchmark_device(device_name, tensor_size, iterations):
    """
    Runs a matrix multiplication benchmark on a specified device.

    Args:
        device_name (str): The device to run on ('cpu' or 'mps').
        tensor_size (int): The dimension for the square matrices
            (e.g., 5000 for 5000x5000).
        iterations (int): The number of times to repeat the multiplication.

    Returns:
        float: The total time taken for all iterations in seconds.
    """
    logger.info(f"\n--- Benchmarking on: {device_name.upper()} ---")

    # 1. Set up the device
    try:
        device = torch.device(device_name)
    except Exception as e:
        logger.error(f"Error setting up device {device_name}: {e}")
        return float("inf")  # Return infinity if device setup fails

    # 2. Create large tensors on the CPU first to not include this in timing
    a = torch.randn(tensor_size, tensor_size)
    b = torch.randn(tensor_size, tensor_size)

    # Move tensors to the target device
    a = a.to(device)
    b = b.to(device)

    # 3. WARM-UP PHASE: Run a few iterations without timing to prepare the device
    logger.info("Running warm-up...")
    warmup_iterations = 10
    for _ in range(warmup_iterations):
        _ = torch.matmul(a, b)

    # For MPS, it's crucial to synchronize to ensure warm-up is complete
    if device.type == "mps":
        torch.mps.synchronize()

    # 4. BENCHMARK PHASE: Time the actual operation
    logger.info(f"Running benchmark ({iterations} iterations)...")
    start_time = time.perf_counter()
    for _ in range(iterations):
        # The core operation to benchmark
        _ = torch.matmul(a, b)

    # Synchronize the MPS stream to make sure all operations are
    # finished before stopping the timer.
    # This is the correct way to time asynchronous GPU operations.
    if device.type == "mps":
        torch.mps.synchronize()

    end_time = time.perf_counter()
    total_time = end_time - start_time

    logger.info(f"Total time: {total_time:.4f} seconds")
    return total_time


if __name__ == "__main__":
    # --- Configuration ---
    TENSOR_SIZE = 5000  # Dimension for the square matrices (e.g., 5000x5000)
    ITERATIONS = 100  # Number of matrix multiplications to perform

    logger.info("PyTorch Performance Benchmark: CPU vs. MPS")
    logger.info("=" * 40)
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Platform: {platform.platform()}")

    # --- Check for MPS Availability ---
    if not torch.backends.mps.is_available():
        logger.error(
            "\n❌ Metal Performance Shaders (MPS) not available on this system."
        )
        logger.error(
            "This test requires an Apple Silicon Mac with macOS "
            "12.3+ and PyTorch 1.12+."
        )
    else:
        logger.info("\n✅ Metal Performance Shaders (MPS) is available.")

        # --- Run Benchmarks ---
        cpu_time = benchmark_device("cpu", TENSOR_SIZE, ITERATIONS)
        mps_time = benchmark_device("mps", TENSOR_SIZE, ITERATIONS)

        # --- Report Results ---
        logger.info("\n--- Benchmark Results ---")
        logger.info(f"CPU Time: {cpu_time:.4f} seconds")
        logger.info(f"MPS Time: {mps_time:.4f} seconds")

        if mps_time < cpu_time:
            speedup = cpu_time / mps_time
            logger.info(f"\n🚀 MPS was {speedup:.2f}x faster than CPU.")
        else:
            speedup = mps_time / cpu_time
            logger.info(
                f"\n🤔 CPU was {speedup:.2f}x faster than MPS. "
                "This can happen for smaller tensors."
            )
