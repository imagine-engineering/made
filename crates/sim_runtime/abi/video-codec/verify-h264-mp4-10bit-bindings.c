#include <libavutil/pixfmt.h>

_Static_assert(AV_PIX_FMT_RGB48LE == 35, "AV_PIX_FMT_RGB48LE");
_Static_assert(AV_PIX_FMT_YUV420P10LE == 62, "AV_PIX_FMT_YUV420P10LE");

int zed_verify_h264_mp4_10bit_bindings(void) {
    return 0;
}
