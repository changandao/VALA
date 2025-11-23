import torch
import gc

def print_gpu_memory_info(prefix=""):
    """
    Print detailed GPU memory information
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device) / (1024**3)   # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
        free_memory = total_memory - allocated
        
        print(f"{prefix}GPU Memory Info:")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
        print(f"  Reserved: {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
        print(f"  Free: {free_memory:.2f} GB ({free_memory/total_memory*100:.1f}%)")
        print(f"  Max allocated: {max_allocated:.2f} GB")
        
        return {
            'total': total_memory,
            'allocated': allocated,
            'reserved': reserved,
            'free': free_memory,
            'max_allocated': max_allocated
        }
    else:
        print(f"{prefix}CUDA not available")
        return None

def cleanup_memory():
    """
    Aggressive memory cleanup
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def check_memory_threshold(threshold_gb=2.0, cleanup=True):
    """
    Check if available memory is below threshold and optionally cleanup
    
    Args:
        threshold_gb: minimum free memory in GB
        cleanup: whether to perform cleanup if below threshold
        
    Returns:
        bool: True if memory is sufficient, False otherwise
    """
    if not torch.cuda.is_available():
        return True
        
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    free_memory = total_memory - allocated
    
    if free_memory < threshold_gb:
        print(f"Warning: Low GPU memory! Free: {free_memory:.2f} GB, Threshold: {threshold_gb:.2f} GB")
        if cleanup:
            print("Performing aggressive memory cleanup...")
            cleanup_memory()
            # Check again after cleanup
            allocated_after = torch.cuda.memory_allocated(device) / (1024**3)
            free_after = total_memory - allocated_after
            print(f"After cleanup - Free: {free_after:.2f} GB")
            return free_after >= threshold_gb
        return False
    return True

class MemoryMonitor:
    """
    Context manager for monitoring memory usage
    """
    def __init__(self, name="Operation", cleanup_on_exit=True):
        self.name = name
        self.cleanup_on_exit = cleanup_on_exit
        self.start_memory = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            cleanup_memory()  # Start with clean slate
            self.start_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"Starting {self.name} - Initial GPU memory: {self.start_memory:.2f} GB")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated() / (1024**3)
            memory_diff = end_memory - self.start_memory if self.start_memory else 0
            print(f"Finished {self.name} - Final GPU memory: {end_memory:.2f} GB (diff: {memory_diff:+.2f} GB)")
            
            if self.cleanup_on_exit:
                cleanup_memory()
                after_cleanup = torch.cuda.memory_allocated() / (1024**3)
                print(f"After cleanup: {after_cleanup:.2f} GB")

def estimate_tensor_memory(shape, dtype=torch.float32):
    """
    Estimate memory usage of a tensor with given shape and dtype
    
    Args:
        shape: tensor shape (tuple)
        dtype: tensor data type
        
    Returns:
        float: estimated memory in GB
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    memory_bytes = total_elements * element_size
    memory_gb = memory_bytes / (1024**3)
    return memory_gb

def suggest_batch_size(total_items, target_memory_gb=1.0, item_memory_gb=None):
    """
    Suggest optimal batch size based on available memory
    
    Args:
        total_items: total number of items to process
        target_memory_gb: target memory usage in GB
        item_memory_gb: memory per item in GB (if known)
        
    Returns:
        int: suggested batch size
    """
    if not torch.cuda.is_available():
        return total_items
        
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    available = total_memory - allocated
    
    # Use conservative estimate if available memory is limited
    safe_memory = min(target_memory_gb, available * 0.8)
    
    if item_memory_gb:
        suggested_batch = max(1, int(safe_memory / item_memory_gb))
    else:
        # Conservative estimate without knowing item size
        suggested_batch = max(1, min(total_items, int(total_items * safe_memory / available)))
    
    return min(suggested_batch, total_items) 