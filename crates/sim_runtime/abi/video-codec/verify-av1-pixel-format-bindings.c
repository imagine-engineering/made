#include <libavutil/pixfmt.h>

_Static_assert(AV_PIX_FMT_RGB24 == 2, "AV_PIX_FMT_RGB24");
_Static_assert(AV_PIX_FMT_YUV420P10LE == 62, "AV_PIX_FMT_YUV420P10LE");

int zed_verify_av1_pixel_format_bindings(void) {
    return 0;
}
