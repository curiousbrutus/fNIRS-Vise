import torch
import time
from lightning.pytorch.callbacks import Callback

class MemoryGuard:
    """Monitors GPU RAM and suggests reducing batch size if OOM."""
    def __init__(self, gpu_index: int = 0, threshold: float = 0.9):
        self.gpu_index = gpu_index
        self.threshold = threshold
        self.oom_triggered = False

    def check_memory(self):
        """Checks current GPU memory usage."""
        if not torch.cuda.is_available():
            print("CUDA not available, skipping memory check.")
            return False

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(self.gpu_index)
        cached = torch.cuda.memory_reserved(self.gpu_index)
        total = torch.cuda.get_device_properties(self.gpu_index).total_memory

        allocated_gb = allocated / (1024**3)
        cached_gb = cached / (1024**3)
        total_gb = total / (1024**3)

        print(f"GPU {self.gpu_index} Memory: Allocated {allocated_gb:.2f} GB, Cached {cached_gb:.2f} GB, Total {total_gb:.2f} GB")

        if allocated / total > self.threshold:
            print("WARNING: GPU memory usage is high. Consider reducing batch size.")
            return True
        return False

    def reduce_batch_size_on_oom(self, exception: Exception, current_batch_size: int):
        """Suggests reducing batch size if an OOM error occurs."""
        if not isinstance(exception, RuntimeError) or "out of memory" not in str(exception).lower():
            return current_batch_size

        if not self.oom_triggered:
            self.oom_triggered = True
            new_batch_size = max(1, current_batch_size // 2)
            print(f"ERROR: CUDA Out of Memory. Reducing batch size from {current_batch_size} to {new_batch_size}")
            time.sleep(5) # Wait a bit for memory to clear
            return new_batch_size
        else:
            print("ERROR: CUDA Out of Memory again. Automatic batch size reduction failed. Please reduce batch size manually.")
            return current_batch_size

    @staticmethod
    def check_oom():
        """Checks for Out of Memory condition and clears cache."""
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 9_000_000_000:
            print("Clearing CUDA cache due to high memory usage.")
            torch.cuda.empty_cache()

class MemoryGuardCallback(Callback):
    """Lightning Callback to integrate MemoryGuard."""
    def on_before_backward(self, trainer, pl_module, loss):
        MemoryGuard.check_oom()
