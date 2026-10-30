#include <cnnl.h>
#include <cnrt.h>

_Static_assert(CNRT_MAJOR_VERSION == 6, "CNRT major version differs");
_Static_assert(CNRT_MINOR_VERSION == 6, "CNRT minor version differs");
_Static_assert(CNRT_PATCH_VERSION == 0, "CNRT patch version differs");
_Static_assert(CNNL_MAJOR == 1, "CNNL major version differs");
_Static_assert(CNNL_MINOR == 20, "CNNL minor version differs");
_Static_assert(CNNL_PATCHLEVEL == 4, "CNNL patch version differs");

_Static_assert(sizeof(cnrtRet_t) == 4, "cnrtRet_t size differs");
_Static_assert(sizeof(cnnlStatus_t) == 4, "cnnlStatus_t size differs");
_Static_assert(cnrtSuccess == 0, "cnrtSuccess differs");
_Static_assert(cnrtErrorNoDevice == 100004, "cnrtErrorNoDevice differs");
_Static_assert(cnrtErrorNoMem == 100100, "cnrtErrorNoMem differs");
_Static_assert(CNNL_STATUS_SUCCESS == 0, "CNNL_STATUS_SUCCESS differs");
_Static_assert(CNNL_STATUS_ALLOC_FAILED == 2,
               "CNNL_STATUS_ALLOC_FAILED differs");

int main(void) { return 0; }
