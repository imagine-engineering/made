"""Observe component-video audio trimming before encoding against pinned source."""
import argparse
import ast
import hashlib
import json
import math
from enum import Enum
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / 'projects/comfy/ComfyUI/comfy_api/latest/_input_impl/video_types.py'
SOURCE_SHA256 = 'b74216bc76178e9a21769e60cf4fd910183a19236cfa7fb352e74f2b13e482e2'
FIXTURE = Path(__file__).with_name('source_audio_plan.json')


def observations():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    body = [node for node in ast.parse(SOURCE.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == 'VideoFromComponents']
    assert len(body) == 1
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    namespace = dict(VideoInput=object, Fraction=Fraction, math=math,
                     VideoContainer=Enum('VideoContainer', {'AUTO': 'auto', 'MP4': 'mp4'}),
                     VideoCodec=Enum('VideoCodec', {'AUTO': 'auto', 'H264': 'h264'}))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[])),
                 str(SOURCE), 'exec'), namespace)
    cases = []
    for frame_count, rate in [(240, Fraction(30000, 1001)),
                              (240, Fraction(60000, 1001)),
                              (3, Fraction(4001, 2000))]:
        for channels in [1, 2, 6]:
            for available_samples in [100, 400000]:
                recorded = {}

                class Frame:
                    def reformat(self, **kwargs):
                        return self

                class Stream:
                    def encode(self, frame=None):
                        return ()

                class Container:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def add_stream(self, codec, **kwargs):
                        if codec == 'h264':
                            encoded_rate = kwargs['rate']
                            recorded['frame_rate'] = [encoded_rate.numerator, encoded_rate.denominator]
                        else:
                            assert codec == 'aac'
                            recorded['sample_rate'] = kwargs['rate']
                            recorded['layout'] = kwargs['layout']
                        return Stream()

                    def mux(self, packet):
                        assert packet == ()

                def audio_frame(array, *, format, layout):
                    assert format == 'fltp'
                    assert layout == recorded['layout']
                    assert list(array.shape[:1]) == [channels]
                    recorded['sample_count'] = array.shape[1]
                    return Frame()

                namespace['av'] = SimpleNamespace(
                    open=lambda *args, **kwargs: Container(),
                    VideoFrame=SimpleNamespace(from_ndarray=lambda *args, **kwargs: Frame()),
                    AudioFrame=SimpleNamespace(from_ndarray=audio_frame),
                )
                components = SimpleNamespace(images=torch.zeros(frame_count, 2, 2, 3), frame_rate=rate,
                    audio=dict(waveform=torch.zeros(1, channels, available_samples), sample_rate=48000))
                namespace['VideoFromComponents'](components).save_to(
                    BytesIO(), format=namespace['VideoContainer'].MP4)
                cases.append(dict(frame_count=frame_count, frame_rate=[rate.numerator, rate.denominator],
                    sample_rate=48000, channels=channels, available_samples=available_samples,
                    output=recorded))
    return dict(schema_version=1, source_sha256=SOURCE_SHA256, cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    options = parser.parse_args()
    result = observations()
    if options.write:
        FIXTURE.write_text(json.dumps(result, indent=2) + '\n')
    else:
        assert json.loads(FIXTURE.read_text()) == result
    print(f'PASS: {len(result["cases"])} source audio plans (recording codec boundary only)')
