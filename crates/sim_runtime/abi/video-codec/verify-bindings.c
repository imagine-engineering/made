#include <stddef.h>
#include "libavcodec/avcodec.h"
#include "libavcodec/codec_par.h"
#include "libavcodec/packet.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavutil/avutil.h"
#include "libavutil/channel_layout.h"
#include "libavutil/dict.h"
#include "libavutil/frame.h"
#include "libavutil/opt.h"
#include "libavutil/rational.h"
#include "libswresample/swresample.h"
#include "libswscale/swscale.h"

_Static_assert(sizeof(AVRational) == 8, "AVRational size");
_Static_assert(_Alignof(AVRational) == 4, "AVRational alignment");
_Static_assert(offsetof(AVRational, num) == 0, "AVRational num offset");
_Static_assert(offsetof(AVRational, den) == 4, "AVRational den offset");
_Static_assert(sizeof(AVChannelLayout) == 24, "AVChannelLayout size");
_Static_assert(_Alignof(AVChannelLayout) == 8, "AVChannelLayout alignment");
_Static_assert(offsetof(AVChannelLayout, order) == 0, "AVChannelLayout order offset");
_Static_assert(offsetof(AVChannelLayout, nb_channels) == 4, "AVChannelLayout channel offset");
_Static_assert(offsetof(AVChannelLayout, u) == 8, "AVChannelLayout data offset");
_Static_assert(offsetof(AVChannelLayout, opaque) == 16, "AVChannelLayout opaque offset");

#define TYPE_MATCH(symbol, signature) \
    _Static_assert(__builtin_types_compatible_p(__typeof__(&(symbol)), signature), #symbol " signature")

// GCC includes FFmpeg's av_const function attribute in strict type compatibility,
// so assignment checks the public signature without discarding that declaration attribute.
static int64_t (*const checked_av_rescale_q)(int64_t, AVRational, AVRational) = av_rescale_q;

TYPE_MATCH(av_packet_alloc, AVPacket *(*)(void));
TYPE_MATCH(av_packet_free, void (*)(AVPacket **));
TYPE_MATCH(av_packet_unref, void (*)(AVPacket *));
TYPE_MATCH(avcodec_alloc_context3, AVCodecContext *(*)(const AVCodec *));
TYPE_MATCH(avcodec_find_decoder, const AVCodec *(*)(enum AVCodecID));
TYPE_MATCH(avcodec_find_encoder_by_name, const AVCodec *(*)(const char *));
TYPE_MATCH(avcodec_free_context, void (*)(AVCodecContext **));
TYPE_MATCH(avcodec_open2, int (*)(AVCodecContext *, const AVCodec *, AVDictionary **));
TYPE_MATCH(avcodec_parameters_from_context, int (*)(AVCodecParameters *, const AVCodecContext *));
TYPE_MATCH(avcodec_parameters_to_context, int (*)(AVCodecContext *, const AVCodecParameters *));
TYPE_MATCH(avcodec_receive_frame, int (*)(AVCodecContext *, AVFrame *));
TYPE_MATCH(avcodec_receive_packet, int (*)(AVCodecContext *, AVPacket *));
TYPE_MATCH(avcodec_send_frame, int (*)(AVCodecContext *, const AVFrame *));
TYPE_MATCH(avcodec_send_packet, int (*)(AVCodecContext *, const AVPacket *));
TYPE_MATCH(avcodec_version, unsigned (*)(void));
TYPE_MATCH(av_find_best_stream, int (*)(AVFormatContext *, enum AVMediaType, int, int, const AVCodec **, int));
TYPE_MATCH(av_interleaved_write_frame, int (*)(AVFormatContext *, AVPacket *));
TYPE_MATCH(av_read_frame, int (*)(AVFormatContext *, AVPacket *));
TYPE_MATCH(av_write_trailer, int (*)(AVFormatContext *));
TYPE_MATCH(avformat_alloc_context, AVFormatContext *(*)(void));
TYPE_MATCH(avformat_alloc_output_context2, int (*)(AVFormatContext **, const AVOutputFormat *, const char *, const char *));
TYPE_MATCH(avformat_close_input, void (*)(AVFormatContext **));
TYPE_MATCH(avformat_find_stream_info, int (*)(AVFormatContext *, AVDictionary **));
TYPE_MATCH(avformat_free_context, void (*)(AVFormatContext *));
TYPE_MATCH(avformat_new_stream, AVStream *(*)(AVFormatContext *, const AVCodec *));
TYPE_MATCH(avformat_open_input, int (*)(AVFormatContext **, const char *, const AVInputFormat *, AVDictionary **));
TYPE_MATCH(avformat_version, unsigned (*)(void));
TYPE_MATCH(avformat_write_header, int (*)(AVFormatContext *, AVDictionary **));
TYPE_MATCH(avio_alloc_context, AVIOContext *(*)(unsigned char *, int, int, void *, int (*)(void *, uint8_t *, int), int (*)(void *, const uint8_t *, int), int64_t (*)(void *, int64_t, int)));
TYPE_MATCH(avio_context_free, void (*)(AVIOContext **));
TYPE_MATCH(av_channel_layout_default, void (*)(AVChannelLayout *, int));
TYPE_MATCH(av_channel_layout_uninit, void (*)(AVChannelLayout *));
TYPE_MATCH(av_dict_free, void (*)(AVDictionary **));
TYPE_MATCH(av_dict_set, int (*)(AVDictionary **, const char *, const char *, int));
TYPE_MATCH(av_frame_alloc, AVFrame *(*)(void));
TYPE_MATCH(av_frame_free, void (*)(AVFrame **));
TYPE_MATCH(av_frame_get_buffer, int (*)(AVFrame *, int));
TYPE_MATCH(av_frame_make_writable, int (*)(AVFrame *));
TYPE_MATCH(av_free, void (*)(void *));
TYPE_MATCH(av_malloc, void *(*)(size_t));
TYPE_MATCH(av_opt_set, int (*)(void *, const char *, const char *, int));
TYPE_MATCH(av_opt_set_int, int (*)(void *, const char *, int64_t, int));
TYPE_MATCH(avutil_version, unsigned (*)(void));
TYPE_MATCH(swr_alloc, SwrContext *(*)(void));
TYPE_MATCH(swr_alloc_set_opts2, int (*)(SwrContext **, const AVChannelLayout *, enum AVSampleFormat, int, const AVChannelLayout *, enum AVSampleFormat, int, int, void *));
TYPE_MATCH(swr_convert, int (*)(SwrContext *, uint8_t *const *, int, const uint8_t *const *, int));
TYPE_MATCH(swr_free, void (*)(SwrContext **));
TYPE_MATCH(swr_init, int (*)(SwrContext *));
TYPE_MATCH(swresample_version, unsigned (*)(void));
TYPE_MATCH(sws_freeContext, void (*)(struct SwsContext *));
TYPE_MATCH(sws_getContext, struct SwsContext *(*)(int, int, enum AVPixelFormat, int, int, enum AVPixelFormat, int, SwsFilter *, SwsFilter *, const double *));
TYPE_MATCH(sws_scale, int (*)(struct SwsContext *, const uint8_t *const *, const int *, int, int, uint8_t *const *, const int *));
TYPE_MATCH(swscale_version, unsigned (*)(void));

int main(void) {
    (void)checked_av_rescale_q;
    return 0;
}
