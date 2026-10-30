#include <stddef.h>
#include <stdint.h>

#include <libavformat/avformat.h>

#define STATIC_ASSERT(condition, message) _Static_assert((condition), message)

typedef struct SimAvFormatContextMetadataProjection {
    const AVClass *av_class;
    const AVInputFormat *iformat;
    const AVOutputFormat *oformat;
    void *priv_data;
    AVIOContext *pb;
    int ctx_flags;
    unsigned int nb_streams;
    AVStream **streams;
    unsigned char opaque_stream_groups_through_data_codec_id[136];
    AVDictionary *metadata;
} SimAvFormatContextMetadataProjection;

STATIC_ASSERT(sizeof(SimAvFormatContextMetadataProjection) == 200,
              "metadata projection size");
STATIC_ASSERT(_Alignof(SimAvFormatContextMetadataProjection) == 8,
              "metadata projection alignment");
STATIC_ASSERT(offsetof(AVFormatContext, metadata) ==
                  offsetof(SimAvFormatContextMetadataProjection, metadata),
              "metadata projection offset");
STATIC_ASSERT(offsetof(SimAvFormatContextMetadataProjection, metadata) == 192,
              "reviewed metadata offset");

int main(void) { return 0; }
