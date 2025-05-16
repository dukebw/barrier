from pathlib import Path

from max.driver import Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, ops


def run_barrier_kernel() -> None:
    """Run the barrier kernel."""
    M = 16384
    N = 4096

    # Ensure this path points to the mojopkg containing kernels/barrier.mojo
    kernels_path = Path(__file__).parent / "kernels.mojopkg"
    if not kernels_path.exists():
        msg = f"Could not find kernels package at {kernels_path.absolute()}"
        raise FileNotFoundError(msg)

    # Define the graph.
    graph = Graph(
        "barrier_test",
        forward=lambda x: ops.inplace_custom(
            "simple_kernel",
            values=[x],
            out_types=[],
            device=DeviceRef.GPU(),
        ),
        input_types=[
            BufferType(
                dtype=DType.float32, shape=(M, N), device=DeviceRef.GPU()
            ),
        ],
        custom_extensions=[kernels_path],
    )

    print(f"Using kernels from: {kernels_path.resolve()}")
    print("Setting up session...")
    # Need CPU and Accelerator devices
    device = Accelerator()
    session = InferenceSession(devices=[device])

    print("Loading graph...")
    # Load the graph, providing the path to the custom ops package
    model = session.load(graph, custom_ops_path=str(kernels_path))
    print("Graph loaded.")

    model(Tensor.zeros(shape=(M, N), dtype=DType.float32, device=device))

    print("\nSuccessfully executed barrier kernel via InferenceSession.")


if __name__ == "__main__":
    run_barrier_kernel()
