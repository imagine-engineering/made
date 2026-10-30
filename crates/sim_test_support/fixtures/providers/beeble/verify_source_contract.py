"""Run pinned Beeble control flow with metadata-only media and recording uploads.

This development oracle stops before submit/poll. It does not encode media,
perform network calls, validate Pydantic coercions, or certify a component.
"""
import argparse
import ast
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / 'projects/sim/ComfyUI/sim_api_nodes/nodes_beeble.py'
VALIDATION = SOURCE.parent / 'util/validation_utils.py'
API = SOURCE.parent / 'apis/beeble.py'
FIXTURE = Path(__file__).with_name('conformance.fixture')
VALIDATION_SHA256 = '77554cc809f651f54943f73e6391b8899b3a52779d949edfc08166d8fdf86c66'
API_SHA256 = '19a0a82f85ee6bc1f514a0af5174029ee868c4650d0e5f35343a8bca1188e356'
SOURCE_SHA256 = 'a7d98292ba01564a4a8519ba3c8caa30d9e0aaf42bf16ca55e29944222896812'


class Tensor:
    def __init__(self, role, frames=1):
        self.role, self.shape = role, (frames, 4, 4)

    def dim(self):
        return 3

    def __getitem__(self, selection):
        return Tensor(self.role, min(self.shape[0], selection.stop))


class Video:
    def __init__(self, frames, role='source'):
        self.frames, self.role = frames, role

    def get_frame_count(self):
        return self.frames

    def get_frame_rate(self):
        return (30000, 1001)


class Submitted(Exception):
    pass


def source_functions():
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in ('_validate_inputs', '_upload_mask_as_image', '_upload_mask_batch_as_video',
                                  '_alpha_url', '_require_output_url')]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ('BeebleSwitchXImageEdit', 'BeebleSwitchXVideoEdit'):
            execute = copy.deepcopy(next(item for item in node.body if isinstance(item, ast.AsyncFunctionDef) and item.name == 'execute'))
            execute.name = 'execute_image' if 'Image' in node.name else 'execute_video'
            execute.decorator_list = []
            selected.append(execute)
    validation = ast.parse(VALIDATION.read_text())
    selected.extend(node for node in validation.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('validate_string', 'validate_video_frame_count'))
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    return compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])), str(SOURCE), 'exec')


async def observe(compiled, value):
    uploads = []
    request = None

    async def upload_image(cls, image, *, wait_label, **kwargs):
        role = {'Uploading keyframe': 'keyframe', 'Uploading alpha': 'alpha_image',
                'Uploading source': 'source_image', 'Uploading reference': 'reference'}[wait_label]
        uploads.append(role)
        name = 'alpha' if role in ('keyframe', 'alpha_image') else image.role
        return 'https://fixture.invalid/' + name

    async def upload_video(cls, video, *, wait_label):
        uploads.append('alpha_video' if wait_label == 'Uploading alpha video' else 'source_video')
        return 'https://fixture.invalid/' + video.role

    async def submit(cls, data):
        nonlocal request
        request = data
        raise Submitted()

    namespace = dict(_MAX_PIXELS=2770000, _MAX_FRAMES=240, _MAX_PROMPT_LEN=2000,
                     downscale_image_tensor=lambda image, maximum: image,
                     downscale_video_to_max_pixels=lambda video, maximum: video,
                     convert_mask_to_image=lambda mask: mask,
                     upload_image_to_simapi=upload_image, upload_video_to_simapi=upload_video,
                     CreateSwitchXRequest=lambda **fields: {key: item for key, item in fields.items() if item is not None},
                     Types=SimpleNamespace(VideoComponents=lambda **fields: fields),
                     InputImpl=SimpleNamespace(VideoFromComponents=lambda components: Video(components['images'].shape[0], 'alpha')),
                     _submit_and_poll=submit)
    exec(compiled, namespace)
    alpha_mode = {'alpha_mode': value['alpha_mode']}
    if value['mask_frame_count'] is not None:
        alpha_mode['alpha_keyframe' if value['alpha_mode'] == 'select' else 'alpha_mask'] = Tensor('alpha', value['mask_frame_count'])
    arguments = dict(cls=None, prompt=value['prompt'], alpha_mode=alpha_mode,
                     max_resolution=value['max_resolution'], seed=value['seed'],
                     reference_image=Tensor('reference') if value['reference_image'] else None)
    kind = value['generation_type']
    arguments['image' if kind == 'image' else 'video'] = Tensor('source') if kind == 'image' else Video(value['source_frame_count'])
    error = False
    try:
        await namespace['execute_' + kind](**arguments)
    except Submitted:
        pass
    except (ValueError, KeyError):
        error = True
    except Exception as exception:
        if '2000' not in str(exception):
            raise
        error = True
    return dict(input=value, expected_request=request, expected_uploads=uploads, expected_error=error)


async def generate():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Pinned Beeble source changed; review before refreshing the oracle')
    for path, digest in ((VALIDATION, VALIDATION_SHA256), (API, API_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Pinned supporting source changed: {path}')
    compiled = source_functions()
    cases = []
    for kind in ('image', 'video'):
        for mode in ('auto', 'fill', 'select', 'custom'):
            for reference in (False, True):
                value = dict(generation_type=kind, prompt=' \u001c A new look \u3000', reference_image=reference,
                             alpha_mode=mode, max_resolution='1080p' if reference else '720p', seed=7,
                             source_frame_count=30 if kind == 'video' else 1,
                             mask_frame_count=(30 if kind == 'video' else 3) if mode in ('select', 'custom') else None)
                cases.append(dict(name=f'{kind}-{mode}-{reference}', **await observe(compiled, value)))
    base = cases[0]['input']
    for name, fields in [
        ('reference-only', dict(prompt=' \u001c\u001f\u3000', reference_image=True)),
        ('missing-prompt-and-reference', dict(prompt=None)),
        ('unicode-limit', dict(prompt='\U0001f680' * 2000)),
        ('unicode-over-limit', dict(prompt='\U0001f680' * 2001)),
        ('video-limit', dict(generation_type='video', source_frame_count=240)),
        ('video-over-limit', dict(generation_type='video', source_frame_count=241)),
        ('custom-count-mismatch', dict(generation_type='video', alpha_mode='custom', source_frame_count=30, mask_frame_count=29)),
        ('missing-keyframe', dict(alpha_mode='select', mask_frame_count=None)),
        ('select-uses-first-mask', dict(generation_type='video', alpha_mode='select', source_frame_count=30, mask_frame_count=2)),
    ]:
        cases.append(dict(name=name, **await observe(compiled, {**base, **fields})))
    namespace = {}
    exec(compiled, namespace)
    responses = []
    for mode in ('auto', 'fill', 'select', 'custom'):
        for output in (None, {}, {'render': 'https://fixture.invalid/render'},
                       {'render': 'https://fixture.invalid/render', 'alpha': 'https://fixture.invalid/alpha'}):
            raw = dict(id='job-1', status='completed', progress=100, output=output)
            response = SimpleNamespace(**{**raw, 'output': None if output is None else
                                          SimpleNamespace(render=output.get('render'), alpha=output.get('alpha'))})
            try:
                render = namespace['_require_output_url'](response, 'render')
                alpha = namespace['_alpha_url'](response, mode)
                expected = [render, alpha]
            except RuntimeError:
                expected = None
            responses.append(dict(mode=mode, response=raw, expected_urls=expected))
    return dict(schema_version=1, namespace='zed.sim.provider.beeble', source_sha256=SOURCE_SHA256,
                validation_sha256=VALIDATION_SHA256, api_sha256=API_SHA256, cases=cases, responses=responses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    options = parser.parse_args()
    expected = json.dumps(asyncio.run(generate()), indent=2, ensure_ascii=False) + '\n'
    if options.write:
        FIXTURE.write_text(expected)
    elif FIXTURE.read_text() != expected:
        raise SystemExit('Beeble source conformance fixture differs')
    print('Verified 25 request/control-flow and 16 output cases from pinned source (metadata-only media).')
