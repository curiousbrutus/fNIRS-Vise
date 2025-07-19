"""
Memory Guard Utility for GPU Memory Management

Prevents OOM errors during training by monitoring and managing CUDA memory.
"""

import torch
import psutil
import gc
from typing import Optional, Tuple
import logging


class MemoryGuard:
    """
    Static utility class for GPU and system memory monitoring.
    
    Provides automatic memory cleanup when approaching limits.
    Designed for Colab K80 (12GB) and similar constrained environments.
    """
    
    # Memory thresholds (in bytes)
    GPU_MEMORY_LIMIT = 9_000_000_000  # 9 GB (leaving 3GB buffer)
    SYSTEM_MEMORY_LIMIT = 0.85        # 85% of available RAM
    
    @staticmethod
    def check_oom() -> bool:
        """
        Check if system is approaching OOM and perform cleanup if needed.
        
        Returns:
            bool: True if cleanup was performed, False otherwise
        """
        cleanup_performed = False
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            
            if gpu_memory > MemoryGuard.GPU_MEMORY_LIMIT:
                MemoryGuard._cleanup_gpu_memory()
                cleanup_performed = True
                
        # Check system memory
        system_memory_percent = psutil.virtual_memory().percent / 100.0
        
        if system_memory_percent > MemoryGuard.SYSTEM_MEMORY_LIMIT:
            MemoryGuard._cleanup_system_memory()
            cleanup_performed = True
            
        return cleanup_performed
    
    @staticmethod
    def _cleanup_gpu_memory():
        """Aggressive GPU memory cleanup."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Log memory stats
            current_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            
            logging.info(f"GPU Memory Cleanup: {current_memory/1e9:.2f}GB / {max_memory/1e9:.2f}GB peak")
    
    @staticmethod
    def _cleanup_system_memory():
        """System memory cleanup."""
        # Force garbage collection
        gc.collect()
        
        # Log memory stats
        memory_info = psutil.virtual_memory()
        logging.info(f"System Memory: {memory_info.percent:.1f}% used ({memory_info.used/1e9:.2f}GB)")
    
    @staticmethod
    def get_memory_stats() -> dict:
        """
        Get detailed memory statistics.
        
        Returns:
            dict: Memory statistics for GPU and system
        """
        stats = {
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_memory_used_gb": psutil.virtual_memory().used / 1e9,
            "system_memory_total_gb": psutil.virtual_memory().total / 1e9,
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "gpu_memory_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            })
        else:
            stats.update({
                "gpu_memory_allocated_gb": 0,
                "gpu_memory_reserved_gb": 0,
                "gpu_memory_max_allocated_gb": 0,
                "gpu_name": "No GPU",
                "gpu_total_memory_gb": 0,
            })
            
        return stats
    
    @staticmethod
    def log_memory_stats():
        """Log current memory statistics."""
        stats = MemoryGuard.get_memory_stats()
        
        print(f"Memory Stats:")
        print(f"  GPU: {stats['gpu_memory_allocated_gb']:.2f}GB / {stats['gpu_total_memory_gb']:.1f}GB")
        print(f"  System: {stats['system_memory_percent']:.1f}% ({stats['system_memory_used_gb']:.2f}GB)")
        
    @staticmethod
    def memory_efficient_forward(model, *args, **kwargs):
        """
        Wrapper for memory-efficient forward passes.
        
        Automatically performs memory cleanup if approaching limits.
        """
        # Check memory before forward pass
        MemoryGuard.check_oom()
        
        try:
            # Perform forward pass
            output = model(*args, **kwargs)
            
            # Check memory after forward pass
            if MemoryGuard.check_oom():
                logging.warning("Memory cleanup performed during forward pass")
                
            return output
            
        except torch.cuda.OutOfMemoryError as e:
            # Emergency cleanup on OOM
            MemoryGuard._cleanup_gpu_memory()
            logging.error(f"CUDA OOM Error: {e}")
            raise
    
    @staticmethod
    def set_memory_limits(gpu_limit_gb: float = 9.0, system_limit_percent: float = 85.0):
        """
        Set custom memory limits.
        
        Args:
            gpu_limit_gb: GPU memory limit in GB
            system_limit_percent: System memory limit as percentage (0-100)
        """
        MemoryGuard.GPU_MEMORY_LIMIT = int(gpu_limit_gb * 1e9)
        MemoryGuard.SYSTEM_MEMORY_LIMIT = system_limit_percent / 100.0
        
        print(f"Memory limits set:")
        print(f"  GPU: {gpu_limit_gb:.1f} GB")
        print(f"  System: {system_limit_percent:.1f}%")


class MemoryProfiler:
    """Context manager for profiling memory usage during operations."""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_stats = None
        
    def __enter__(self):
        self.start_stats = MemoryGuard.get_memory_stats()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_stats = MemoryGuard.get_memory_stats()
        
        # Calculate differences
        gpu_diff = end_stats['gpu_memory_allocated_gb'] - self.start_stats['gpu_memory_allocated_gb']
        system_diff = end_stats['system_memory_percent'] - self.start_stats['system_memory_percent']
        
        print(f"{self.operation_name} Memory Usage:")
        print(f"  GPU: {gpu_diff:+.3f} GB")
        print(f"  System: {system_diff:+.1f}%")


# Convenience functions
def check_memory_and_cleanup():
    """Convenience function for memory check and cleanup."""
    return MemoryGuard.check_oom()


def log_memory():
    """Convenience function for logging memory stats."""
    MemoryGuard.log_memory_stats()


if __name__ == "__main__":
    # Test memory guard functionality
    print("Testing MemoryGuard...")
    
    # Log initial memory state
    MemoryGuard.log_memory_stats()
    
    # Test memory profiler
    with MemoryProfiler("Tensor Creation"):
        # Create some tensors to use memory
        if torch.cuda.is_available():
            large_tensor = torch.randn(1000, 1000, device='cuda')
            del large_tensor
            torch.cuda.empty_cache()
        else:
            large_tensor = torch.randn(1000, 1000)
            del large_tensor
            
    # Test memory check
    cleanup_performed = MemoryGuard.check_oom()
    print(f"Cleanup performed: {cleanup_performed}")
    
    # Get detailed stats
    stats = MemoryGuard.get_memory_stats()
    print(f"Memory statistics: {stats}")
    
    print("✓ MemoryGuard test completed!")
