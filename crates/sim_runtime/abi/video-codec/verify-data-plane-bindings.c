#include <stddef.h>
#include <stdint.h>
#include "libavcodec/avcodec.h"
#include "libavcodec/codec.h"
#include "libavcodec/codec_id.h"
#include "libavcodec/packet.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavutil/error.h"
#include "libavutil/frame.h"
#include "libavutil/mathematics.h"
#include "libavutil/opt.h"
#include "libavutil/pixfmt.h"
#include "libswscale/swscale.h"

typedef struct SimAvFramePrefix {
    uint8_t *data[AV_NUM_DATA_POINTERS];
    int linesize[AV_NUM_DATA_POINTERS];
    uint8_t **extended_data;
    int width;
    int height;
    int nb_samples;
    int format;
    int key_frame;
    enum AVPictureType pict_type;
    AVRational sample_aspect_ratio;
    int64_t pts;
} SimAvFramePrefix;

typedef struct SimAvPacketPrefix {
    AVBufferRef *buf;
    int64_t pts;
    int64_t dts;
    uint8_t *data;
    int size;
    int stream_index;
    int flags;
    AVPacketSideData *side_data;
    int side_data_elems;
    int64_t duration;
} SimAvPacketPrefix;

typedef struct SimAvStreamPrefix {
    const AVClass *av_class;
    int index;
    int id;
    AVCodecParameters *codecpar;
    void *priv_data;
    AVRational time_base;
} SimAvStreamPrefix;

typedef struct SimAvFormatContextPrefix {
    const AVClass *av_class;
    const AVInputFormat *iformat;
    const AVOutputFormat *oformat;
    void *priv_data;
    AVIOContext *pb;
    int ctx_flags;
    unsigned int nb_streams;
    AVStream **streams;
} SimAvFormatContextPrefix;

typedef struct SimAvIoContextPrefix {
    const AVClass *av_class;
    unsigned char *buffer;
} SimAvIoContextPrefix;

#define PREFIX_LAYOUT(prefix, size_value, alignment_value) \
    _Static_assert(sizeof(prefix) == size_value, #prefix " size"); \
    _Static_assert(_Alignof(prefix) == alignment_value, #prefix " alignment")
#define FIELD_OFFSET(actual, prefix, field, offset_value) \
    _Static_assert(offsetof(actual, field) == offset_value, #actual "." #field " offset"); \
    _Static_assert(offsetof(prefix, field) == offset_value, #prefix "." #field " offset")

PREFIX_LAYOUT(SimAvFramePrefix, 144, 8);
FIELD_OFFSET(AVFrame, SimAvFramePrefix, data, 0);
FIELD_OFFSET(AVFrame, SimAvFramePrefix, linesize, 64);
FIELD_OFFSET(AVFrame, SimAvFramePrefix, extended_data, 96);
FIELD_OFFSET(AVFrame, SimAvFramePrefix, width, 104);
FIELD_OFFSET(AVFrame, SimAvFramePrefix, height, 108);
FIELD_OFFSET(AVFrame, SimAvFramePrefix, nb_samples, 112);
FIELD_OFFSET(AVFrame, SimAvFramePrefix, format, 116);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
FIELD_OFFSET(AVFrame, SimAvFramePrefix, key_frame, 120);
#pragma GCC diagnostic pop
FIELD_OFFSET(AVFrame, SimAvFramePrefix, pict_type, 124);
FIELD_OFFSET(AVFrame, SimAvFramePrefix, sample_aspect_ratio, 128);
FIELD_OFFSET(AVFrame, SimAvFramePrefix, pts, 136);

PREFIX_LAYOUT(SimAvPacketPrefix, 72, 8);
FIELD_OFFSET(AVPacket, SimAvPacketPrefix, buf, 0);
FIELD_OFFSET(AVPacket, SimAvPacketPrefix, pts, 8);
FIELD_OFFSET(AVPacket, SimAvPacketPrefix, dts, 16);
FIELD_OFFSET(AVPacket, SimAvPacketPrefix, data, 24);
FIELD_OFFSET(AVPacket, SimAvPacketPrefix, size, 32);
FIELD_OFFSET(AVPacket, SimAvPacketPrefix, stream_index, 36);
FIELD_OFFSET(AVPacket, SimAvPacketPrefix, flags, 40);
FIELD_OFFSET(AVPacket, SimAvPacketPrefix, side_data, 48);
FIELD_OFFSET(AVPacket, SimAvPacketPrefix, side_data_elems, 56);
FIELD_OFFSET(AVPacket, SimAvPacketPrefix, duration, 64);

PREFIX_LAYOUT(SimAvStreamPrefix, 40, 8);
FIELD_OFFSET(AVStream, SimAvStreamPrefix, av_class, 0);
FIELD_OFFSET(AVStream, SimAvStreamPrefix, index, 8);
FIELD_OFFSET(AVStream, SimAvStreamPrefix, id, 12);
FIELD_OFFSET(AVStream, SimAvStreamPrefix, codecpar, 16);
FIELD_OFFSET(AVStream, SimAvStreamPrefix, priv_data, 24);
FIELD_OFFSET(AVStream, SimAvStreamPrefix, time_base, 32);

PREFIX_LAYOUT(SimAvFormatContextPrefix, 56, 8);
FIELD_OFFSET(AVFormatContext, SimAvFormatContextPrefix, av_class, 0);
FIELD_OFFSET(AVFormatContext, SimAvFormatContextPrefix, iformat, 8);
FIELD_OFFSET(AVFormatContext, SimAvFormatContextPrefix, oformat, 16);
FIELD_OFFSET(AVFormatContext, SimAvFormatContextPrefix, priv_data, 24);
FIELD_OFFSET(AVFormatContext, SimAvFormatContextPrefix, pb, 32);
FIELD_OFFSET(AVFormatContext, SimAvFormatContextPrefix, ctx_flags, 40);
FIELD_OFFSET(AVFormatContext, SimAvFormatContextPrefix, nb_streams, 44);
FIELD_OFFSET(AVFormatContext, SimAvFormatContextPrefix, streams, 48);

PREFIX_LAYOUT(SimAvIoContextPrefix, 16, 8);
FIELD_OFFSET(AVIOContext, SimAvIoContextPrefix, av_class, 0);
FIELD_OFFSET(AVIOContext, SimAvIoContextPrefix, buffer, 8);

_Static_assert(AV_NUM_DATA_POINTERS == 8, "AV_NUM_DATA_POINTERS");
_Static_assert(AVMEDIA_TYPE_VIDEO == 0, "AVMEDIA_TYPE_VIDEO");
_Static_assert(AV_CODEC_ID_H264 == 27, "AV_CODEC_ID_H264");
_Static_assert(AV_CODEC_ID_VP9 == 167, "AV_CODEC_ID_VP9");
_Static_assert(AV_CODEC_ID_AV1 == 225, "AV_CODEC_ID_AV1");
_Static_assert(AV_CODEC_ID_AAC == 86018, "AV_CODEC_ID_AAC");
_Static_assert(AV_PIX_FMT_YUV420P == 0, "AV_PIX_FMT_YUV420P");
_Static_assert(AV_PIX_FMT_RGB24 == 2, "AV_PIX_FMT_RGB24");
_Static_assert(AV_NOPTS_VALUE == INT64_MIN, "AV_NOPTS_VALUE");
_Static_assert(AVSEEK_SIZE == 0x10000, "AVSEEK_SIZE");
_Static_assert(AVSEEK_FORCE == 0x20000, "AVSEEK_FORCE");
_Static_assert(AVFMT_FLAG_CUSTOM_IO == 0x0080, "AVFMT_FLAG_CUSTOM_IO");
_Static_assert(AV_CODEC_FLAG_GLOBAL_HEADER == (1 << 22), "AV_CODEC_FLAG_GLOBAL_HEADER");
_Static_assert(AV_OPT_SEARCH_CHILDREN == 1, "AV_OPT_SEARCH_CHILDREN");
_Static_assert(SWS_BILINEAR == 2, "SWS_BILINEAR");
_Static_assert(AVERROR(EAGAIN) == -11, "AVERROR(EAGAIN)");
_Static_assert(AVERROR(ENOMEM) == -12, "AVERROR(ENOMEM)");
_Static_assert(AVERROR(EINVAL) == -22, "AVERROR(EINVAL)");
_Static_assert(AVERROR(ENOSPC) == -28, "AVERROR(ENOSPC)");
_Static_assert(AVERROR_EOF == -541478725, "AVERROR_EOF");
_Static_assert(AVERROR_EXIT == -1414092869, "AVERROR_EXIT");

int main(void) {
    return 0;
}
