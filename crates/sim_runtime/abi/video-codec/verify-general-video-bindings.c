#include <stddef.h>
#include "libavcodec/avcodec.h"
#include "libavcodec/codec_par.h"
#include "libavcodec/packet.h"
#include "libavformat/avformat.h"
#include "libavutil/channel_layout.h"
#include "libavutil/dict.h"
#include "libavutil/display.h"
#include "libavutil/frame.h"
#include "libavutil/mathematics.h"
#include "libavutil/pixdesc.h"
#include "libswresample/swresample.h"
#include "libswscale/swscale.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/buffersrc.h"
#include "libavfilter/buffersink.h"

#define TYPE_MATCH(symbol, signature) \
    _Static_assert(__builtin_types_compatible_p(__typeof__(&(symbol)), signature), #symbol " signature")

_Static_assert(LIBAVCODEC_VERSION_MAJOR == 61, "libavcodec ABI");
_Static_assert(LIBAVFORMAT_VERSION_MAJOR == 61, "libavformat ABI");
_Static_assert(LIBAVUTIL_VERSION_MAJOR == 59, "libavutil ABI");
_Static_assert(LIBSWRESAMPLE_VERSION_MAJOR == 5, "libswresample ABI");
_Static_assert(LIBSWSCALE_VERSION_MAJOR == 8, "libswscale ABI");
_Static_assert(LIBAVFILTER_VERSION_MAJOR == 10, "libavfilter ABI");

_Static_assert(sizeof(AVCodecParameters) == 176, "AVCodecParameters size");
_Static_assert(_Alignof(AVCodecParameters) == 8, "AVCodecParameters alignment");
_Static_assert(offsetof(AVCodecParameters, codec_type) == 0, "codec_type offset");
_Static_assert(offsetof(AVCodecParameters, codec_id) == 4, "codec_id offset");
_Static_assert(offsetof(AVCodecParameters, format) == 44, "format offset");
_Static_assert(offsetof(AVCodecParameters, bits_per_raw_sample) == 60, "raw depth offset");
_Static_assert(offsetof(AVCodecParameters, width) == 72, "width offset");
_Static_assert(offsetof(AVCodecParameters, height) == 76, "height offset");
_Static_assert(offsetof(AVCodecParameters, ch_layout) == 128, "channel layout offset");
_Static_assert(offsetof(AVCodecParameters, sample_rate) == 152, "sample rate offset");

_Static_assert(sizeof(AVStream) == 232, "AVStream size");
_Static_assert(_Alignof(AVStream) == 8, "AVStream alignment");
_Static_assert(offsetof(AVStream, codecpar) == 16, "codecpar offset");
_Static_assert(offsetof(AVStream, time_base) == 32, "time base offset");
_Static_assert(offsetof(AVStream, start_time) == 40, "stream start offset");
_Static_assert(offsetof(AVStream, duration) == 48, "stream duration offset");
_Static_assert(offsetof(AVStream, nb_frames) == 56, "frame count offset");
_Static_assert(offsetof(AVStream, metadata) == 80, "stream metadata offset");
_Static_assert(offsetof(AVStream, avg_frame_rate) == 88, "average frame rate offset");

_Static_assert(sizeof(AVFormatContext) == 472, "AVFormatContext size");
_Static_assert(_Alignof(AVFormatContext) == 8, "AVFormatContext alignment");
_Static_assert(offsetof(AVFormatContext, nb_streams) == 44, "stream count offset");
_Static_assert(offsetof(AVFormatContext, streams) == 48, "streams offset");
_Static_assert(offsetof(AVFormatContext, start_time) == 96, "format start offset");
_Static_assert(offsetof(AVFormatContext, duration) == 104, "format duration offset");
_Static_assert(offsetof(AVFormatContext, metadata) == 192, "format metadata offset");

_Static_assert(sizeof(AVFrame) == 440, "AVFrame size");
_Static_assert(_Alignof(AVFrame) == 8, "AVFrame alignment");
_Static_assert(offsetof(AVFrame, pts) == 136, "frame pts offset");
_Static_assert(offsetof(AVFrame, sample_rate) == 192, "frame sample rate offset");
_Static_assert(offsetof(AVFrame, best_effort_timestamp) == 320, "best effort timestamp offset");
_Static_assert(offsetof(AVFrame, metadata) == 336, "frame metadata offset");
_Static_assert(offsetof(AVFrame, ch_layout) == 408, "frame channel layout offset");
_Static_assert(offsetof(AVFrame, duration) == 432, "frame duration offset");

_Static_assert(sizeof(AVDictionaryEntry) == 16, "AVDictionaryEntry size");
_Static_assert(offsetof(AVDictionaryEntry, key) == 0, "dictionary key offset");
_Static_assert(offsetof(AVDictionaryEntry, value) == 8, "dictionary value offset");
_Static_assert(sizeof(AVFrameSideData) == 40, "AVFrameSideData size");
_Static_assert(offsetof(AVFrameSideData, type) == 0, "side data type offset");
_Static_assert(offsetof(AVFrameSideData, data) == 8, "side data bytes offset");
_Static_assert(offsetof(AVFrameSideData, size) == 16, "side data length offset");
_Static_assert(offsetof(AVFrameSideData, metadata) == 24, "side data metadata offset");
_Static_assert(offsetof(AVFrameSideData, buf) == 32, "side data buffer offset");
_Static_assert(sizeof(AVComponentDescriptor) == 20, "AVComponentDescriptor size");
_Static_assert(sizeof(AVPixFmtDescriptor) == 112, "AVPixFmtDescriptor size");
_Static_assert(offsetof(AVPixFmtDescriptor, name) == 0, "pixel descriptor name offset");
_Static_assert(offsetof(AVPixFmtDescriptor, nb_components) == 8, "component count offset");
_Static_assert(offsetof(AVPixFmtDescriptor, flags) == 16, "pixel flags offset");
_Static_assert(offsetof(AVPixFmtDescriptor, comp) == 24, "pixel components offset");
_Static_assert(offsetof(AVPixFmtDescriptor, alias) == 104, "pixel alias offset");

_Static_assert(AVMEDIA_TYPE_AUDIO == 1, "audio media type");
_Static_assert(AV_SAMPLE_FMT_FLTP == 8, "float planar sample format");
_Static_assert(AV_PKT_DATA_DISPLAYMATRIX == 5, "packet display matrix");
_Static_assert(AV_FRAME_DATA_DISPLAYMATRIX == 6, "frame display matrix");
_Static_assert(AV_BUFFERSRC_FLAG_KEEP_REF == 8, "buffer source keep-reference flag");
_Static_assert(AV_ROUND_NEAR_INF == 5, "nearest rescale rounding");
_Static_assert(AV_ROUND_PASS_MINMAX == 8192, "timestamp passthrough rounding");
_Static_assert(AV_PIX_FMT_FLAG_ALPHA == 128, "alpha descriptor flag");
_Static_assert(AV_PIX_FMT_PAL8 == 11, "pal8 pixel format");
_Static_assert(AV_PIX_FMT_YUVJ420P == 12, "yuvj420p pixel format");
_Static_assert(AV_PIX_FMT_YUVJ422P == 13, "yuvj422p pixel format");
_Static_assert(AV_PIX_FMT_YUVJ444P == 14, "yuvj444p pixel format");
_Static_assert(AV_PIX_FMT_GBRPF32LE == 175, "gbrpf32le pixel format");
_Static_assert(AV_PIX_FMT_GBRAPF32LE == 177, "gbrapf32le pixel format");

TYPE_MATCH(avcodec_flush_buffers, void (*)(AVCodecContext *));
TYPE_MATCH(avcodec_parameters_copy, int (*)(AVCodecParameters *, const AVCodecParameters *));
TYPE_MATCH(av_packet_rescale_ts, void (*)(AVPacket *, AVRational, AVRational));
TYPE_MATCH(av_seek_frame, int (*)(AVFormatContext *, int, int64_t, int));
TYPE_MATCH(avformat_seek_file, int (*)(AVFormatContext *, int, int64_t, int64_t, int64_t, int));
TYPE_MATCH(av_guess_frame_rate, AVRational (*)(AVFormatContext *, AVStream *, AVFrame *));
TYPE_MATCH(av_channel_layout_copy, int (*)(AVChannelLayout *, const AVChannelLayout *));
TYPE_MATCH(av_dict_iterate, const AVDictionaryEntry *(*)(const AVDictionary *, const AVDictionaryEntry *));
TYPE_MATCH(av_frame_get_side_data, AVFrameSideData *(*)(const AVFrame *, enum AVFrameSideDataType));
TYPE_MATCH(av_frame_unref, void (*)(AVFrame *));
TYPE_MATCH(av_get_pix_fmt_name, const char *(*)(enum AVPixelFormat));
TYPE_MATCH(av_pix_fmt_desc_get, const AVPixFmtDescriptor *(*)(enum AVPixelFormat));
TYPE_MATCH(swr_get_delay, int64_t (*)(SwrContext *, int64_t));
TYPE_MATCH(avfilter_version, unsigned (*)(void));
TYPE_MATCH(avfilter_get_by_name, const AVFilter *(*)(const char *));
TYPE_MATCH(avfilter_graph_alloc, AVFilterGraph *(*)(void));
TYPE_MATCH(avfilter_graph_create_filter, int (*)(AVFilterContext **, const AVFilter *, const char *, const char *, void *, AVFilterGraph *));
TYPE_MATCH(avfilter_link, int (*)(AVFilterContext *, unsigned, AVFilterContext *, unsigned));
TYPE_MATCH(avfilter_graph_config, int (*)(AVFilterGraph *, void *));
TYPE_MATCH(avfilter_graph_free, void (*)(AVFilterGraph **));
TYPE_MATCH(av_buffersrc_add_frame_flags, int (*)(AVFilterContext *, AVFrame *, int));
TYPE_MATCH(av_buffersink_get_frame, int (*)(AVFilterContext *, AVFrame *));

static double (*const checked_av_display_rotation_get)(const int32_t matrix[9]) = av_display_rotation_get;
static int64_t (*const checked_av_rescale_rnd)(int64_t, int64_t, int64_t, enum AVRounding) = av_rescale_rnd;

int main(void) {
    (void)checked_av_display_rotation_get;
    (void)checked_av_rescale_rnd;
    return 0;
}
