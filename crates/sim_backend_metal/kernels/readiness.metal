#include <metal_stdlib>
using namespace metal;

kernel void zed_sim_metal_readiness_v1(
    device const uint *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        output[0] = input[0] ^ 0x53494d31u;
    }
}
