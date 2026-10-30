"""Observe pinned Beeble alpha frames before the canonical H.264 codec boundary."""
import argparse
import ast
import asyncio
import json
import hashlib
from fractions import Fraction
from enum import Enum
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import torch
from verify_source_media import SOURCE, SOURCES as MEDIA_SOURCES, source_namespace

SOURCES = {**MEDIA_SOURCES, "sim_api/latest/_input_impl/video_types.py": "b74216bc76178e9a21769e60cf4fd910183a19236cfa7fb352e74f2b13e482e2"}

FIXTURE = Path(__file__).with_name('source_alpha_video.json')


async def observations():
    namespace = source_namespace()
    path = SOURCE / 'sim_api_nodes/nodes_beeble.py'
    body = [node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == '_upload_mask_batch_as_video']
    assert len(body) == 1
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[])), str(path), 'exec'), namespace)
    namespace['Types'] = SimpleNamespace(VideoComponents=lambda **fields: SimpleNamespace(**fields))
    video_path = SOURCE / 'sim_api/latest/_input_impl/video_types.py'
    assert hashlib.sha256(video_path.read_bytes()).hexdigest() == SOURCES['sim_api/latest/_input_impl/video_types.py']
    video_class = [node for node in ast.parse(video_path.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == 'VideoFromComponents']
    assert len(video_class) == 1
    namespace.update(VideoInput=object, VideoComponents=lambda **fields: SimpleNamespace(**fields),
                     VideoContainer=Enum('VideoContainer', {'AUTO': 'auto', 'MP4': 'mp4'}),
                     VideoCodec=Enum('VideoCodec', {'AUTO': 'auto', 'H264': 'h264'}), Fraction=Fraction)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *video_class], type_ignores=[])),
                 str(video_path), 'exec'), namespace)
    namespace['InputImpl'] = SimpleNamespace(VideoFromComponents=namespace['VideoFromComponents'])
    cases = []
    for shape in ([8, 11], [1, 8, 11], [3, 8, 11], [2, 1, 11]):
        values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
        values = ((values % 17) - 2) / 13
        mask = values.reshape(shape)
        count = 1 if len(shape) == 2 else shape[0]
        for maximum in (16, 2770000):
            for source_count in (count, count + 1):
                recorded = []

                async def upload(cls, video, *, wait_label):
                    assert wait_label == 'Uploading alpha video'
                    codec_rate = []

                    class StopAtCodec(Exception):
                        pass

                    class Container:
                        def __enter__(self):
                            return self

                        def __exit__(self, *args):
                            return False

                        def add_stream(self, codec, *, rate):
                            assert codec == 'h264'
                            codec_rate.append([rate.numerator, rate.denominator])
                            raise StopAtCodec

                    namespace['av'] = SimpleNamespace(open=lambda *args, **kwargs: Container())
                    try:
                        video.save_to(BytesIO(), format=namespace['VideoContainer'].MP4,
                                      codec=namespace['VideoCodec'].H264)
                    except StopAtCodec:
                        pass
                    assert len(codec_rate) == 1
                    components = video.get_components()
                    assert components.audio is None
                    recorded.append(dict(shape=list(components.images.shape),
                                         pixels=components.images.flatten().tolist(),
                                         frame_rate=[components.frame_rate.numerator, components.frame_rate.denominator],
                                         bit_depth=video.get_bit_depth(), audio=False, encode_frame_rate=codec_rate[0]))
                    return 'https://fixture.invalid/alpha.mp4'

                namespace['_MAX_PIXELS'] = maximum
                namespace['upload_video_to_simapi'] = upload
                case = dict(shape=list(shape), pixels=values.tolist(), maximum_pixels=maximum,
                            source_frame_count=source_count, frame_rate=[30000, 1001])
                try:
                    await namespace['_upload_mask_batch_as_video'](None, mask,
                        frame_rate=Fraction(30000, 1001), source_frame_count=source_count,
                        wait_label='Uploading alpha video')
                    assert len(recorded) == 1
                    case['output'] = recorded[0]
                except ValueError as error:
                    assert not recorded
                    case['error'] = str(error)
                cases.append(case)
    return dict(schema_version=1, sources=SOURCES, cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    options = parser.parse_args()
    result = asyncio.run(observations())
    if options.write:
        FIXTURE.write_text(json.dumps(result, indent=2) + '\n')
    else:
        assert json.loads(FIXTURE.read_text()) == result
    print(f'PASS: {len(result["cases"])} source alpha-video preparation cases (codec boundary only)')
