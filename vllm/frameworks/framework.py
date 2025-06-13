from types import ModuleType


class Framework:
    """Base class for all framework plugins.

    Framework plugins should inherit from this class and implement the
    required methods.
    """
    cuda: ModuleType
    npu: ModuleType
    framework_name: str
    nn: ModuleType
    dtype: object
    bfloat16: object
    float16: object
    float32: object
    float64: object
    long: object
    int: object
    int8: object
    uint8: object
    int32: object
    int64: object
    Tensor: object
    distributed: ModuleType
    device: object
    Size: object
    multiprocessing: ModuleType
    profiler: ModuleType

    def __init__(self, name: str):
        pass

    def cat(self, tensors, dim=0):
        """Concatenate a list of tensors along a specified dimension."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def empty(self, size, dtype=None, device=None):
        """Create an empty tensor with the given size, dtype and device."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def empty_like(self, tensor):
        """Create an empty tensor with the same size, dtype and device as the given tensor."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def tensor(self, data, dtype=None, device=None):
        """Create a tensor from the given data."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def inference_mode(self):
        """Context manager for inference mode."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def ByteTensor(self, data=None, device=None):
        """Create a ByteTensor."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def zeros(self, size, dtype=None, device=None):
        """Create a tensor filled with zeros."""
        raise NotImplementedError("This method should be implemented by subclasses.")