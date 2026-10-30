#include <libavutil/pixfmt.h>

_Static_assert(AV_PIX_FMT_RGBA == 26, "AV_PIX_FMT_RGBA");
_Static_assert(AV_PIX_FMT_YUVA420P == 33, "AV_PIX_FMT_YUVA420P");

int zed_verify_vp9_alpha_bindings(void) {
    return 0;
}
