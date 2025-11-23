# Significance and Activated Tracking

This implementation adds support for tracking Gaussian significance and activation similar to the `rasterize_to_pixels_fwd.cu` implementation in gsplat.

## New Features

### Parameters Added

- `activated`: An array of size `[P]` (number of Gaussians) that tracks which Gaussians are directly activated (contribute to at least one pixel center)
- `significance`: An array of size `[P]` that tracks the maximum visibility value each Gaussian achieves across all pixels

### How It Works

For each Gaussian during rendering:

1. **Direct Projection Check**: We check if the Gaussian center falls within ±0.5 pixels of a pixel center:
   ```cpp
   bool is_direct_projection = (d.x >= -0.5f && d.x < 0.5f) && 
                              (d.y >= -0.5f && d.y < 0.5f);
   ```
   where `d = gaussian_center - pixel_center`

2. **Significance Update**: If it's a direct projection, we atomically update:
   ```cpp
   atomicMaxFloat(&significance[gaussian_id], vis);
   atomicExch(&activated[gaussian_id], 1);
   ```
   where `vis = alpha * T` is the visibility contribution

### Usage

The new parameters are optional in the public API:

```cpp
CudaRasterizer::Rasterizer::forward(
    // ... existing parameters ...
    out_activated,    // int32_t* [P] - optional, can be nullptr
    out_significance  // float* [P] - optional, can be nullptr
);
```

### Use Cases

- **Gaussian Pruning**: Remove Gaussians with low significance values
- **Adaptive Densification**: Focus on areas where Gaussians are highly significant
- **Training Optimization**: Track which Gaussians are actually contributing to rendering
- **Memory Optimization**: Identify inactive Gaussians for removal

### Example

```cpp
// Allocate output buffers
int32_t* activated = nullptr;
float* significance = nullptr;
cudaMalloc(&activated, P * sizeof(int32_t));
cudaMalloc(&significance, P * sizeof(float));

// Call rendering with significance tracking
CudaRasterizer::Rasterizer::forward(
    // ... other parameters ...
    activated,
    significance
);

// Now you can use activated and significance arrays
// for Gaussian management and optimization
```

## Implementation Details

- Uses atomic operations to handle concurrent updates from multiple threads
- Compatible with both normal and gated rendering modes
- Minimal performance overhead when not used (pass nullptr)
- Follows the same logic as gsplat's `rasterize_to_pixels_fwd.cu` 