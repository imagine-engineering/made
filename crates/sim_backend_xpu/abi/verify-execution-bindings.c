#include <stddef.h>
#include <stdint.h>
#include <oneapi/dnnl/dnnl.h>
#include <ze_api.h>

#define CHECK_LAYOUT(type, expected_size, expected_alignment) \
    _Static_assert(sizeof(type) == (expected_size), "size differs: " #type); \
    _Static_assert(_Alignof(type) == (expected_alignment), "alignment differs: " #type)
#define CHECK_OFFSET(type, field, expected) \
    _Static_assert(offsetof(type, field) == (expected), "offset differs: " #type "." #field)

_Static_assert(ZE_API_VERSION_1_6 == 65542, "Level Zero API floor differs");
_Static_assert(ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES == 3, "device properties stype differs");
_Static_assert(ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES == 6, "queue group stype differs");
_Static_assert(ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES == 7, "memory properties stype differs");
_Static_assert(ZE_STRUCTURE_TYPE_CONTEXT_DESC == 13, "context stype differs");
_Static_assert(ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC == 14, "queue stype differs");
_Static_assert(ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE == 1, "compute flag differs");
_Static_assert(ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS == 2, "queue mode differs");
_Static_assert(ZE_COMMAND_QUEUE_PRIORITY_NORMAL == 0, "queue priority differs");
_Static_assert(ZE_DEVICE_TYPE_GPU == 1, "GPU type differs");
_Static_assert(dnnl_success == 0, "oneDNN success differs");
_Static_assert(dnnl_gpu == 2, "oneDNN GPU kind differs");
_Static_assert(dnnl_stream_default_flags == 1, "oneDNN stream flags differ");
_Static_assert(dnnl_f16 == 1, "oneDNN f16 differs");
_Static_assert(dnnl_f32 == 3, "oneDNN f32 differs");
_Static_assert(dnnl_binary_add == 131056, "oneDNN binary Add differs");
_Static_assert(DNNL_ARG_SRC_0 == 1, "oneDNN SRC_0 differs");
_Static_assert(DNNL_ARG_SRC_1 == 2, "oneDNN SRC_1 differs");
_Static_assert(DNNL_ARG_DST == 17, "oneDNN DST differs");
_Static_assert((intptr_t)DNNL_MEMORY_ALLOCATE == -1, "oneDNN allocation sentinel differs");

CHECK_LAYOUT(ze_result_t, 4, 4);
CHECK_LAYOUT(ze_driver_handle_t, 8, 8);
CHECK_LAYOUT(ze_device_handle_t, 8, 8);
CHECK_LAYOUT(ze_context_handle_t, 8, 8);
CHECK_LAYOUT(ze_command_queue_handle_t, 8, 8);
CHECK_LAYOUT(ze_context_desc_t, 24, 8);
CHECK_LAYOUT(ze_command_queue_desc_t, 40, 8);
CHECK_LAYOUT(ze_command_queue_group_properties_t, 40, 8);
CHECK_LAYOUT(ze_device_uuid_t, 16, 1);
CHECK_LAYOUT(ze_device_properties_t, 368, 8);
CHECK_LAYOUT(ze_device_memory_properties_t, 296, 8);
CHECK_LAYOUT(dnnl_status_t, 4, 4);
CHECK_LAYOUT(dnnl_engine_t, 8, 8);
CHECK_LAYOUT(dnnl_stream_t, 8, 8);
CHECK_LAYOUT(dnnl_memory_desc_t, 8, 8);
CHECK_LAYOUT(dnnl_memory_t, 8, 8);
CHECK_LAYOUT(dnnl_primitive_desc_t, 8, 8);
CHECK_LAYOUT(dnnl_primitive_t, 8, 8);
CHECK_LAYOUT(dnnl_exec_arg_t, 16, 8);
CHECK_LAYOUT(dnnl_version_t, 32, 8);

CHECK_OFFSET(ze_context_desc_t, stype, 0);
CHECK_OFFSET(ze_context_desc_t, pNext, 8);
CHECK_OFFSET(ze_context_desc_t, flags, 16);
CHECK_OFFSET(ze_command_queue_desc_t, stype, 0);
CHECK_OFFSET(ze_command_queue_desc_t, pNext, 8);
CHECK_OFFSET(ze_command_queue_desc_t, ordinal, 16);
CHECK_OFFSET(ze_command_queue_desc_t, index, 20);
CHECK_OFFSET(ze_command_queue_desc_t, flags, 24);
CHECK_OFFSET(ze_command_queue_desc_t, mode, 28);
CHECK_OFFSET(ze_command_queue_desc_t, priority, 32);
CHECK_OFFSET(ze_command_queue_group_properties_t, stype, 0);
CHECK_OFFSET(ze_command_queue_group_properties_t, pNext, 8);
CHECK_OFFSET(ze_command_queue_group_properties_t, flags, 16);
CHECK_OFFSET(ze_command_queue_group_properties_t, maxMemoryFillPatternSize, 24);
CHECK_OFFSET(ze_command_queue_group_properties_t, numQueues, 32);
CHECK_OFFSET(ze_device_properties_t, stype, 0);
CHECK_OFFSET(ze_device_properties_t, pNext, 8);
CHECK_OFFSET(ze_device_properties_t, type, 16);
CHECK_OFFSET(ze_device_properties_t, vendorId, 20);
CHECK_OFFSET(ze_device_properties_t, deviceId, 24);
CHECK_OFFSET(ze_device_properties_t, flags, 28);
CHECK_OFFSET(ze_device_properties_t, subdeviceId, 32);
CHECK_OFFSET(ze_device_properties_t, coreClockRate, 36);
CHECK_OFFSET(ze_device_properties_t, maxMemAllocSize, 40);
CHECK_OFFSET(ze_device_properties_t, maxHardwareContexts, 48);
CHECK_OFFSET(ze_device_properties_t, maxCommandQueuePriority, 52);
CHECK_OFFSET(ze_device_properties_t, numThreadsPerEU, 56);
CHECK_OFFSET(ze_device_properties_t, physicalEUSimdWidth, 60);
CHECK_OFFSET(ze_device_properties_t, numEUsPerSubslice, 64);
CHECK_OFFSET(ze_device_properties_t, numSubslicesPerSlice, 68);
CHECK_OFFSET(ze_device_properties_t, numSlices, 72);
CHECK_OFFSET(ze_device_properties_t, timerResolution, 80);
CHECK_OFFSET(ze_device_properties_t, timestampValidBits, 88);
CHECK_OFFSET(ze_device_properties_t, kernelTimestampValidBits, 92);
CHECK_OFFSET(ze_device_properties_t, uuid, 96);
CHECK_OFFSET(ze_device_properties_t, name, 112);
CHECK_OFFSET(ze_device_memory_properties_t, stype, 0);
CHECK_OFFSET(ze_device_memory_properties_t, pNext, 8);
CHECK_OFFSET(ze_device_memory_properties_t, flags, 16);
CHECK_OFFSET(ze_device_memory_properties_t, maxClockRate, 20);
CHECK_OFFSET(ze_device_memory_properties_t, maxBusWidth, 24);
CHECK_OFFSET(ze_device_memory_properties_t, totalSize, 32);
CHECK_OFFSET(ze_device_memory_properties_t, name, 40);
CHECK_OFFSET(dnnl_exec_arg_t, arg, 0);
CHECK_OFFSET(dnnl_exec_arg_t, memory, 8);
CHECK_OFFSET(dnnl_version_t, major, 0);
CHECK_OFFSET(dnnl_version_t, minor, 4);
CHECK_OFFSET(dnnl_version_t, patch, 8);
CHECK_OFFSET(dnnl_version_t, hash, 16);
CHECK_OFFSET(dnnl_version_t, cpu_runtime, 24);
CHECK_OFFSET(dnnl_version_t, gpu_runtime, 28);

#define CHECK_FUNCTION(symbol, pointer_type) \
    _Static_assert(_Generic(&(symbol), pointer_type: 1, default: 0), "signature differs: " #symbol)

typedef ze_result_t (*ze_init_pointer)(ze_init_flags_t);
typedef ze_result_t (*ze_driver_get_pointer)(uint32_t *, ze_driver_handle_t *);
typedef ze_result_t (*ze_driver_version_pointer)(ze_driver_handle_t, ze_api_version_t *);
typedef ze_result_t (*ze_device_get_pointer)(ze_driver_handle_t, uint32_t *, ze_device_handle_t *);
typedef ze_result_t (*ze_device_properties_pointer)(ze_device_handle_t, ze_device_properties_t *);
typedef ze_result_t (*ze_device_memory_pointer)(ze_device_handle_t, uint32_t *, ze_device_memory_properties_t *);
typedef ze_result_t (*ze_queue_groups_pointer)(ze_device_handle_t, uint32_t *, ze_command_queue_group_properties_t *);
typedef ze_result_t (*ze_context_create_pointer)(ze_driver_handle_t, const ze_context_desc_t *, ze_context_handle_t *);
typedef ze_result_t (*ze_context_destroy_pointer)(ze_context_handle_t);
typedef ze_result_t (*ze_queue_create_pointer)(ze_context_handle_t, ze_device_handle_t, const ze_command_queue_desc_t *, ze_command_queue_handle_t *);
typedef ze_result_t (*ze_queue_synchronize_pointer)(ze_command_queue_handle_t, uint64_t);
typedef ze_result_t (*ze_queue_destroy_pointer)(ze_command_queue_handle_t);
typedef size_t (*dnnl_engine_count_pointer)(dnnl_engine_kind_t);
typedef dnnl_status_t (*dnnl_engine_create_pointer)(dnnl_engine_t *, dnnl_engine_kind_t, size_t);
typedef dnnl_status_t (*dnnl_engine_destroy_pointer)(dnnl_engine_t);
typedef dnnl_status_t (*dnnl_stream_create_pointer)(dnnl_stream_t *, dnnl_engine_t, unsigned);
typedef dnnl_status_t (*dnnl_stream_wait_pointer)(dnnl_stream_t);
typedef dnnl_status_t (*dnnl_stream_destroy_pointer)(dnnl_stream_t);
typedef dnnl_status_t (*dnnl_memory_desc_create_pointer)(dnnl_memory_desc_t *, int, const dnnl_dim_t *, dnnl_data_type_t, const dnnl_dim_t *);
typedef dnnl_status_t (*dnnl_memory_desc_destroy_pointer)(dnnl_memory_desc_t);
typedef dnnl_status_t (*dnnl_memory_create_pointer)(dnnl_memory_t *, const_dnnl_memory_desc_t, dnnl_engine_t, void *);
typedef dnnl_status_t (*dnnl_memory_map_pointer)(const_dnnl_memory_t, void **);
typedef dnnl_status_t (*dnnl_memory_unmap_pointer)(const_dnnl_memory_t, void *);
typedef dnnl_status_t (*dnnl_memory_destroy_pointer)(dnnl_memory_t);
typedef dnnl_status_t (*dnnl_binary_desc_create_pointer)(dnnl_primitive_desc_t *, dnnl_engine_t, dnnl_alg_kind_t, const_dnnl_memory_desc_t, const_dnnl_memory_desc_t, const_dnnl_memory_desc_t, const_dnnl_primitive_attr_t);
typedef dnnl_status_t (*dnnl_primitive_desc_destroy_pointer)(dnnl_primitive_desc_t);
typedef dnnl_status_t (*dnnl_primitive_create_pointer)(dnnl_primitive_t *, const_dnnl_primitive_desc_t);
typedef dnnl_status_t (*dnnl_primitive_execute_pointer)(const_dnnl_primitive_t, dnnl_stream_t, int, const dnnl_exec_arg_t *);
typedef dnnl_status_t (*dnnl_primitive_destroy_pointer)(dnnl_primitive_t);
typedef const dnnl_version_t *(*dnnl_version_pointer)(void);

CHECK_FUNCTION(zeInit, ze_init_pointer);
CHECK_FUNCTION(zeDriverGet, ze_driver_get_pointer);
CHECK_FUNCTION(zeDriverGetApiVersion, ze_driver_version_pointer);
CHECK_FUNCTION(zeDeviceGet, ze_device_get_pointer);
CHECK_FUNCTION(zeDeviceGetProperties, ze_device_properties_pointer);
CHECK_FUNCTION(zeDeviceGetMemoryProperties, ze_device_memory_pointer);
CHECK_FUNCTION(zeDeviceGetCommandQueueGroupProperties, ze_queue_groups_pointer);
CHECK_FUNCTION(zeContextCreate, ze_context_create_pointer);
CHECK_FUNCTION(zeContextDestroy, ze_context_destroy_pointer);
CHECK_FUNCTION(zeCommandQueueCreate, ze_queue_create_pointer);
CHECK_FUNCTION(zeCommandQueueSynchronize, ze_queue_synchronize_pointer);
CHECK_FUNCTION(zeCommandQueueDestroy, ze_queue_destroy_pointer);
CHECK_FUNCTION(dnnl_engine_get_count, dnnl_engine_count_pointer);
CHECK_FUNCTION(dnnl_engine_create, dnnl_engine_create_pointer);
CHECK_FUNCTION(dnnl_engine_destroy, dnnl_engine_destroy_pointer);
CHECK_FUNCTION(dnnl_stream_create, dnnl_stream_create_pointer);
CHECK_FUNCTION(dnnl_stream_wait, dnnl_stream_wait_pointer);
CHECK_FUNCTION(dnnl_stream_destroy, dnnl_stream_destroy_pointer);
CHECK_FUNCTION(dnnl_memory_desc_create_with_strides, dnnl_memory_desc_create_pointer);
CHECK_FUNCTION(dnnl_memory_desc_destroy, dnnl_memory_desc_destroy_pointer);
CHECK_FUNCTION(dnnl_memory_create, dnnl_memory_create_pointer);
CHECK_FUNCTION(dnnl_memory_map_data, dnnl_memory_map_pointer);
CHECK_FUNCTION(dnnl_memory_unmap_data, dnnl_memory_unmap_pointer);
CHECK_FUNCTION(dnnl_memory_destroy, dnnl_memory_destroy_pointer);
CHECK_FUNCTION(dnnl_binary_primitive_desc_create, dnnl_binary_desc_create_pointer);
CHECK_FUNCTION(dnnl_primitive_desc_destroy, dnnl_primitive_desc_destroy_pointer);
CHECK_FUNCTION(dnnl_primitive_create, dnnl_primitive_create_pointer);
CHECK_FUNCTION(dnnl_primitive_execute, dnnl_primitive_execute_pointer);
CHECK_FUNCTION(dnnl_primitive_destroy, dnnl_primitive_destroy_pointer);
CHECK_FUNCTION(dnnl_version, dnnl_version_pointer);

int main(void) {
    return 0;
}
