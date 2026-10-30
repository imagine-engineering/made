#include <metal_stdlib>
using namespace metal;

kernel void zed_sim_metal_add_f32_v1(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        output[index] = left[index] + right[index];
    }
}

kernel void zed_sim_metal_add_f16_v1(
    device const half *left [[buffer(0)]],
    device const half *right [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index < count) {
        output[index] = left[index] + right[index];
    }
}
