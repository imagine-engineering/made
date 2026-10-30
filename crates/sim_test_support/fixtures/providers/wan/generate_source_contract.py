import argparse
import ast
import asyncio
import base64
import contextlib
import copy
import hashlib
import io
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import aiohttp
import numpy as np
import pydantic
from PIL import Image

SOURCE = Path('/home/asvegah/repos/sim/projects/comfy/ComfyUI')
FRONTEND = Path('/home/asvegah/repos/sim/projects/comfy/ComfyUI-Frontend')
ROOT = Path(__file__).resolve().parent
MP4 = (FRONTEND / 'browser_tests/assets/plain_video.mp4').read_bytes()
MP3 = (FRONTEND / 'src/scripts/metadata/__fixtures__/with_metadata.mp3').read_bytes()
PATHS = ['comfy_api_nodes/nodes_wan.py', 'comfy_api_nodes/apis/wan.py',
         'comfy_api_nodes/util/client.py', 'comfy_api_nodes/util/conversions.py',
         'comfy_api_nodes/util/download_helpers.py', 'comfy_api_nodes/util/_helpers.py',
         'comfy_api_nodes/util/validation_utils.py', 'comfy_api_nodes/util/upload_helpers.py',
         'comfy_api/latest/_input_impl/video_types.py']
operations = []


class Tensor:
    def __init__(self, slot='image', count=1, width=2, height=2, channels=3, array=None):
        self.slot = slot
        self.array = (np.tile(np.array([.5, .25, .125] + ([.375] if channels == 4 else []),
                                     dtype=np.float32), (count, height, width, 1))
                      if array is None else array)

    @property
    def shape(self): return self.array.shape
    @property
    def ndim(self): return self.array.ndim
    def __len__(self): return len(self.array)
    def __getitem__(self, index): return Tensor(self.slot + ':' + str(index), array=self.array[index])
    def __iter__(self):
        result = [self[index] for index in range(len(self.array))]
        operations.append({'kind': 'image_batch_items', 'input': serial(self), 'options': {}, 'output': serial(result)})
        return iter(result)
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.array
    def unsqueeze(self, dimension): return Tensor(self.slot, array=np.expand_dims(self.array, dimension))
    def squeeze(self, dimension=None): return Tensor(self.slot, array=np.squeeze(self.array, axis=dimension))
    def movedim(self, origin, target): return Tensor(self.slot, array=np.moveaxis(self.array, origin, target))


class Video:
    def __init__(self, slot='video', duration=5.0, duration_error=False, save_error=False):
        self.slot, self.duration = slot, duration
        self.duration_error, self.save_error = duration_error, save_error
    def get_duration(self):
        if self.duration_error: raise ValueError('source metadata unavailable')
        return self.duration
    def save_to(self, destination, **options):
        if self.save_error: raise ValueError('source codec owner unavailable')
        destination.write(MP4)


class OrderedMapping(dict):
    def __iter__(self):
        operations.append({'kind': 'mapping_items', 'input': serial(self), 'options': {},
                           'output': [[key, serial(value)] for key, value in self.items()]})
        return super().__iter__()


def serial(value):
    if isinstance(value, Tensor):
        return {'native_slot': value.slot, 'shape': list(value.shape),
                'pixel': value.array.reshape(-1)[0].item() if value.array.size else None,
                'channels': value.shape[-1] if value.ndim == 4 else None}
    if isinstance(value, Video):
        return {'native_slot': value.slot, 'duration': value.duration,
                'duration_error': value.duration_error, 'save_error': value.save_error}
    if hasattr(value, 'model_dump'): return value.model_dump(mode='json', by_alias=True)
    if isinstance(value, dict): return {key: serial(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)): return [serial(member) for member in value]
    return value


def strip(node):
    for member in ast.walk(node):
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            member.returns = None
            for argument in member.args.args + member.args.kwonlyargs: argument.annotation = None
    return ast.fix_missing_locations(node)


def run_module(namespace, path, names):
    tree = ast.parse((SOURCE / path).read_text())
    members = [copy.deepcopy(node) for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names
               or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    exec(compile(strip(ast.Module(body=members, type_ignores=[])), '<actual-Wan-' + path + '>', 'exec'), namespace)
    return tree


def runs(values):
    result = []
    for value in values:
        if result and result[-1]['value'] == value: result[-1]['repeat'] += 1
        else: result.append({'value': value, 'repeat': 1})
    return result


def audio(seconds=4.0, sample_rate=8000):
    return {'sample_rate': sample_rate,
            'waveform': Tensor('audio', array=np.zeros((1, 1, int(seconds * sample_rate)), dtype=np.float32))}


async def generate():
    logging.disable(logging.CRITICAL)
    requests, downloads, delays, progress, polls, uploads, codec_calls, poll_calls = [], [], [], [], [], [], [], []
    overrides, contracts, api_probes, scalar_probes, status_probes = {}, {}, [], [], []
    namespace = {'__builtins__': __builtins__, '__name__': 'actual_wan_source_probe',
                 'base64': base64, 'BytesIO': io.BytesIO, 'Image': Image, 'np': np, 'math': math,
                 'json': json, 'asyncio': asyncio, 'contextlib': contextlib, 'logging': logging,
                 'dataclass': dataclass, 're': re, 'torch': SimpleNamespace(Tensor=Tensor,
                  from_numpy=lambda value: Tensor('download', array=value)),
                 'IO': SimpleNamespace(ComfyNode=type('ComfyNode', (), {}), NodeOutput=lambda *values: values),
                 'time': SimpleNamespace(monotonic=lambda: 0.0), 'is_processing_interrupted': lambda: False,
                 'uuid': SimpleNamespace(uuid4=lambda: '00000000-0000-0000-0000-000000000001'),
                 'mimetype_to_extension': lambda value: value.split('/')[-1]}
    hidden = SimpleNamespace(auth_token_comfy_org='capability-token', api_key_comfy_org='capability-api-key', comfy_usage_source=None)
    namespace['IO'].ComfyNode.hidden = hidden
    namespace['get_deploy_environment'] = lambda: 'host'
    run_module(namespace, PATHS[5], ['get_auth_header', 'get_usage_source', 'get_comfy_api_headers'])
    auth = []
    for token, key in [('capability-token', 'capability-api-key'), (None, 'capability-api-key'), (None, None)]:
        hidden.auth_token_comfy_org, hidden.api_key_comfy_org = token, key
        auth.append({'token': bool(token), 'key': bool(key), 'headers': namespace['get_comfy_api_headers'](namespace['IO'].ComfyNode)})
    hidden.auth_token_comfy_org, hidden.api_key_comfy_org = 'capability-token', 'capability-api-key'
    exec(compile((SOURCE / PATHS[1]).read_text(), '<actual-Wan-api-models>', 'exec'), namespace)
    actual_models = {name: value for name, value in namespace.items() if isinstance(value, type) and hasattr(value, 'model_validate')}
    request_models = [name for name in actual_models if (name.endswith(('Field', 'Request')) or name == 'Wan27MediaItem') and name not in ['TaskCreationOutputField', 'ImageTaskStatusOutputField', 'VideoTaskStatusOutputField']]
    def constructor(name):
        def call(**kwargs):
            record = {'kind': 'api_construct', 'input': serial(kwargs), 'options': {'model': name}}
            operations.append(record)
            try:
                result = actual_models[name](**kwargs)
                record['output'] = result.model_dump(mode='json', exclude_none=name.endswith('Request'))
            except Exception as error:
                record['error_type'] = type(error).__name__
                raise
            return result
        return call
    for name in request_models: namespace[name] = constructor(name)
    video_tree = ast.parse((SOURCE / PATHS[8]).read_text())
    video_class = next(node for node in video_tree.body if isinstance(node, ast.ClassDef) and node.name == 'VideoFromFile')
    wrapped = ast.ClassDef(name='VideoFromFile', bases=[], keywords=[], decorator_list=[], body=[copy.deepcopy(member) for member in video_class.body if isinstance(member, ast.FunctionDef) and member.name in ['__init__', 'get_stream_source']])
    namespace['io'] = io
    exec(compile(strip(ast.Module(body=[wrapped], type_ignores=[])), '<actual-Wan-VideoFromFile-wrapper>', 'exec'), namespace)
    namespace['InputImpl'] = SimpleNamespace(VideoFromFile=namespace['VideoFromFile'])
    namespace['Types'] = SimpleNamespace(VideoContainer=SimpleNamespace(MP4=SimpleNamespace(value='mp4')), VideoCodec=SimpleNamespace(H264='h264'))
    run_module(namespace, PATHS[3], ['tensor_to_pil', 'tensor_to_base64_string', 'tensor_to_bytesio', 'pil_to_bytesio', '_compute_downscale_dims', 'downscale_image_tensor', 'bytesio_to_image_tensor', 'audio_to_base64_string', 'audio_tensor_to_contiguous_ndarray'])
    source_image = namespace['tensor_to_base64_string']
    response_png = base64.b64decode(source_image(Tensor()))
    def encode_audio(array, sample_rate, container_format='mp4', codec_name='aac'):
        codec_calls.append({'shape': list(array.shape), 'sample_rate': sample_rate, 'container': container_format, 'codec': codec_name,
                            'encoded_input_sha256': hashlib.sha256(MP3).hexdigest(), 'gate': 'injected existing MP3 bytes; no PyAV encoding'})
        return io.BytesIO(MP3)
    namespace['audio_ndarray_to_bytesio'] = encode_audio
    run_module(namespace, PATHS[6], ['get_image_dimensions', 'validate_image_dimensions', 'validate_image_aspect_ratio', '_assert_ratio_bounds', '_ratio_from_tuple', 'get_number_of_images', 'validate_audio_duration', 'validate_video_duration', 'validate_string'])
    def instrument(name, argument_names):
        actual = namespace[name]
        def function(*args, **kwargs):
            record = {'kind': name, 'input': serial(args[0]), 'options': {**{key: serial(value) for key, value in zip(argument_names, args[1:])}, **serial(kwargs)}}
            operations.append(record)
            try: record['output'] = serial(actual(*args, **kwargs))
            except Exception as error:
                record['error_type'] = type(error).__name__
                raise
            return record['output']
        namespace[name] = function
    for name, names in [('get_number_of_images', []), ('tensor_to_base64_string', ['total_pixels', 'mime_type']), ('audio_to_base64_string', ['container_format', 'codec_name']), ('validate_string', []), ('validate_audio_duration', ['min_duration', 'max_duration']), ('validate_video_duration', ['min_duration', 'max_duration']), ('validate_image_dimensions', []), ('validate_image_aspect_ratio', ['min_ratio', 'max_ratio'])]: instrument(name, names)
    upload_tree = run_module(namespace, PATHS[7], ['upload_image_to_comfyapi', 'upload_images_to_comfyapi', 'upload_audio_to_comfyapi', 'upload_video_to_comfyapi'])
    async def file_owner(cls, data, name, mime_type=None, wait_label='Uploading', *args):
        encoded = data.getvalue()
        record = {'file_name': name, 'content_type': mime_type, 'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest(),
                  'wait_label': wait_label, 'classification': 'actual wrapper/PIL or injected codec-boundary bytes; no signed HTTP'}
        uploads.append(record)
        return 'https://fixture.invalid/upload/' + str(len(uploads)) + '/' + name
    namespace['upload_file_to_comfyapi'] = file_owner
    for name in ['upload_image_to_comfyapi', 'upload_audio_to_comfyapi', 'upload_video_to_comfyapi']:
        actual = namespace[name]
        def make_upload(name, actual):
            async def call(cls, value=None, **kwargs):
                if value is None and 'image' in kwargs: value = kwargs.pop('image')
                record = {'kind': name, 'input': serial(value), 'options': serial(kwargs)}
                operations.append(record)
                try: record['output'] = await actual(cls, value, **kwargs)
                except Exception as error:
                    record['error_type'] = type(error).__name__
                    raise
                return record['output']
            return call
        namespace[name] = make_upload(name, actual)
    client = run_module(namespace, PATHS[2], ['COMPLETED_STATUSES', 'FAILED_STATUSES', 'QUEUED_STATUSES', '_PollUIState', 'sync_op', 'poll_op', 'poll_op_raw', '_normalize_statuses', '_normalize_status_value', '_validate_or_raise', '_wrap_model_extractor'])
    class Interrupted(Exception): pass
    class NetworkError(Exception): pass
    class ApiError(Exception): pass
    async def delay(seconds, *args): delays.append(seconds)
    namespace.update(ProcessingInterrupted=Interrupted, LocalNetworkError=NetworkError, ApiServerError=ApiError, sleep_with_interrupt=delay,
                     _display_time_progress=lambda *args, **kwargs: progress.append(kwargs),
                     utils=SimpleNamespace(ProgressBar=lambda total: SimpleNamespace(update_absolute=lambda *args, **kwargs: None)),
                     ApiEndpoint=lambda path, method='GET', **kwargs: SimpleNamespace(path=path, method=method, headers={}))
    initial = {'output': {'task_id': 'task-1', 'task_status': 'PENDING'}, 'request_id': 'request-1'}
    image_final = {'output': {'task_id': 'task-1', 'task_status': 'SUCCEEDED', 'results': [{'url': 'https://fixture.invalid/image.png'}, {'url': 'https://fixture.invalid/ignored.png'}]}, 'request_id': 'request-1'}
    video_final = {'output': {'task_id': 'task-1', 'task_status': 'SUCCEEDED', 'video_url': 'https://fixture.invalid/video.mp4'}, 'request_id': 'request-1'}
    async def raw_sync(cls, endpoint, data=None, **options):
        body = data.model_dump(mode='json', exclude_none=True) if data else None
        headers = namespace['get_comfy_api_headers'](cls)
        if endpoint.method != 'GET': headers = {**headers, 'Content-Type': 'application/json'}
        requests.append({'path': endpoint.path, 'method': endpoint.method, 'body': body,
                         'json_body_hex': aiohttp.payload.JsonPayload(body)._value.hex() if body is not None else '',
                         'headers': headers, 'header_order': list(headers), 'options': options})
        if endpoint.method == 'POST': return copy.deepcopy(overrides.get('generate', initial))
        sequence = overrides.get('polls', [image_final if active_node in ['WanTextToImageApi', 'WanImageToImageApi'] else video_final])
        value = copy.deepcopy(sequence[min(len(polls), len(sequence) - 1)])
        polls.append(value)
        return value
    namespace['sync_op_raw'] = raw_sync
    actual_poll = namespace['poll_op']
    async def poll_call(cls, endpoint, **options):
        poll_calls.append({'path': endpoint.path, 'interval': options.get('poll_interval', 5.0),
                           'estimated_duration': options.get('estimated_duration'), 'response_model': options['response_model'].__name__})
        return await actual_poll(cls, endpoint, **options)
    namespace['poll_op'] = poll_call
    async def download_bytes(url, destination, **options):
        downloads.append({'url': url, 'options': options})
        parsed = urlparse(url)
        if not parsed.scheme and not parsed.netloc and options.get('cls') is None: raise ValueError("For relative 'cloud' paths, the `cls` parameter is required.")
        destination.write(response_png if active_node in ['WanTextToImageApi', 'WanImageToImageApi'] else MP4)
        destination.seek(0)
    namespace['download_url_to_bytesio'] = download_bytes
    run_module(namespace, PATHS[4], ['download_url_to_image_tensor', 'download_url_to_video_output'])
    tree = ast.parse((SOURCE / PATHS[0]).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'RES_IN_PARENS' for target in node.targets): selected.append(copy.deepcopy(node))
        if not isinstance(node, ast.ClassDef): continue
        methods = [member for member in node.body if isinstance(member, ast.AsyncFunctionDef) and member.name == 'execute']
        if not methods: continue
        method = copy.deepcopy(methods[0])
        schema = next(call for call in ast.walk(node) if isinstance(call, ast.Call) and ast.unparse(call.func) == 'IO.Schema')
        fields = {keyword.arg: keyword.value for keyword in schema.keywords}
        defaults = {argument.arg: ast.literal_eval(default) for argument, default in zip(method.args.args[-len(method.args.defaults):], method.args.defaults)} if method.args.defaults else {}
        contracts[node.name] = {'arguments': [argument.arg for argument in method.args.args if argument.arg != 'cls'], 'defaults': defaults,
                                'schema': ast.unparse(schema), 'hidden': ast.unparse(fields['hidden']), 'outputs': ast.unparse(fields['outputs']), 'price_badge': ast.unparse(fields['price_badge'])}
        selected.append(ast.ClassDef(name=node.name, bases=[ast.Attribute(value=ast.Name(id='IO', ctx=ast.Load()), attr='ComfyNode', ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]))
    exec(compile(strip(ast.Module(body=selected, type_ignores=[])), '<actual-Wan-14-registered-execute-methods>', 'exec'), namespace)
    model = {'model': 'wan2.7-t2v', 'prompt': ' source prompt ', 'negative_prompt': '', 'resolution': '720P', 'ratio': '16:9', 'duration': 5}
    bases = {
        'WanTextToImageApi': {'model': 'wan2.5-t2i-preview', 'prompt': ' source prompt '},
        'WanImageToImageApi': {'model': 'wan2.5-i2i-preview', 'image': Tensor('image', 2), 'prompt': ' source prompt '},
        'WanTextToVideoApi': {'model': 'wan2.5-t2v-preview', 'prompt': ' source prompt '},
        'WanImageToVideoApi': {'model': 'wan2.5-i2v-preview', 'image': Tensor(), 'prompt': ' source prompt '},
        'WanReferenceVideoApi': {'model': 'wan2.6-r2v', 'prompt': ' source prompt ', 'negative_prompt': '', 'reference_videos': OrderedMapping(z=Video('z'), a=Video('a')), 'size': '720p: 1:1 (960x960)', 'duration': 5, 'seed': 0, 'shot_type': 'single', 'watermark': False},
        'Wan2TextToVideoApi': {'model': model, 'seed': 0, 'prompt_extend': True, 'watermark': False},
        'Wan2ImageToVideoApi': {'model': model, 'first_frame': Tensor('first_frame'), 'seed': 0, 'prompt_extend': True, 'watermark': False},
        'Wan2VideoContinuationApi': {'model': model, 'first_clip': Video('first_clip')},
        'Wan2VideoEditApi': {'model': {**model, 'duration': 'auto', 'reference_images': OrderedMapping(z=Tensor('z'), a=Tensor('a'))}, 'video': Video(), 'seed': 0, 'audio_setting': 'auto', 'watermark': False},
        'Wan2ReferenceVideoApi': {'model': {**model, 'reference_videos': OrderedMapping(z=Video('z'), a=Video('a')), 'reference_images': OrderedMapping(image=Tensor('image'))}, 'seed': 0, 'watermark': False},
        'HappyHorseTextToVideoApi': {'model': model, 'seed': 0, 'watermark': False},
        'HappyHorseImageToVideoApi': {'model': model, 'first_frame': Tensor('first_frame', width=300, height=300), 'seed': 0, 'watermark': False},
        'HappyHorseVideoEditApi': {'model': {**model, 'reference_images': OrderedMapping(z=Tensor('z'), a=Tensor('a'))}, 'video': Video(), 'seed': 0, 'watermark': False},
        'HappyHorseReferenceVideoApi': {'model': {**model, 'reference_images': OrderedMapping(z=Tensor('z', width=400, height=400), a=Tensor('a', width=400, height=400))}, 'seed': 0, 'watermark': False},
    }
    labels = {'Wan2TextToVideoApi': 'wan2.7-t2v', 'Wan2ImageToVideoApi': 'wan2.7-i2v',
              'Wan2VideoContinuationApi': 'wan2.7-i2v', 'Wan2VideoEditApi': 'wan2.7-videoedit',
              'Wan2ReferenceVideoApi': 'wan2.7-r2v', 'HappyHorseTextToVideoApi': 'happyhorse-1.1-t2v',
              'HappyHorseImageToVideoApi': 'happyhorse-1.1-i2v', 'HappyHorseVideoEditApi': 'happyhorse-1.0-video-edit',
              'HappyHorseReferenceVideoApi': 'happyhorse-1.1-r2v'}
    for node, label in labels.items(): bases[node]['model'] = {**bases[node]['model'], 'model': label}
    cases, errors, response_errors = [], [], []
    active_node = ''
    async def observe(node, arguments, failure=False, override=None):
        nonlocal active_node
        active_node = node
        for values in [requests, downloads, delays, progress, polls, operations, uploads, codec_calls, poll_calls]: values.clear()
        overrides.clear(); overrides.update(override or {})
        record = {'input': {'node': node, 'arguments': serial(arguments)}, 'override': override or {}}
        try:
            output = await namespace[node].execute(**copy.deepcopy(arguments))
        except Exception as error:
            if not failure: raise
            record.update(error_type=type(error).__name__, message=re.sub(r'0x[0-9a-fA-F]+', '<address>', str(error)))
            last_status = namespace['_normalize_status_value']((polls[-1].get('output') or {}).get('task_status')) if polls else None
            record['failure_phase'] = ('before_generation' if not requests else 'generation_response' if 'generate' in overrides else 'download_wrapper' if downloads else 'after_generation_before_poll' if not poll_calls else 'result_selection' if last_status in namespace['COMPLETED_STATUSES'] else 'polling')
            (response_errors if override else errors).append(record)
        else:
            if failure: raise ValueError('Expected source failure ' + node)
            result = output[0]
            record['output'] = ({'kind': 'IMAGE', 'shape': list(result.shape), 'pixels': result.array.reshape(-1, result.shape[-1])[0].tolist()}
                                if isinstance(result, Tensor) else {'kind': 'VIDEO', 'bytes': len(result.get_stream_source().getvalue()), 'sha256': hashlib.sha256(result.get_stream_source().getvalue()).hexdigest()})
            cases.append(record)
        record.update(operations=copy.deepcopy(operations), request_runs=runs(copy.deepcopy(requests)), downloads=copy.deepcopy(downloads), delay_runs=runs(delays), progress=copy.deepcopy(progress), poll_runs=runs(copy.deepcopy(polls)), uploads=copy.deepcopy(uploads), codec_calls=copy.deepcopy(codec_calls), poll_calls=copy.deepcopy(poll_calls))
        for response_kind, response_model, value in [('Generation', 'TaskCreationResponse', overrides.get('generate', initial))] + [('ImagePoll' if node in ['WanTextToImageApi', 'WanImageToImageApi'] else 'VideoPoll', 'ImageTaskStatusResponse' if node in ['WanTextToImageApi', 'WanImageToImageApi'] else 'VideoTaskStatusResponse', value) for value in polls]:
            probe = {'kind': response_kind, 'input': value}
            try: probe['output'] = actual_models[response_model].model_validate(value).model_dump(mode='json')
            except Exception as error: probe['error_type'] = type(error).__name__
            if probe not in api_probes: api_probes.append(probe)
        generation = overrides.get('generate', initial)
        for value in [None, 1024, generation.get('code'), generation.get('message')] + [((value.get('output') or {}).get('results') or [{}])[0].get('url') for value in polls]:
            probe = {'input': value, 'output': str(value)}
            if probe not in scalar_probes: scalar_probes.append(probe)
        for value in [(value.get('output') or {}).get('task_status') for value in polls]:
            normalized = namespace['_normalize_status_value'](value)
            probe = {'input': value, 'output': normalized, 'completed': normalized in namespace['COMPLETED_STATUSES'], 'failed': normalized in namespace['FAILED_STATUSES'], 'queued': normalized in namespace['QUEUED_STATUSES']}
            if probe not in status_probes: status_probes.append(probe)
    for node, base in bases.items():
        await observe(node, base)
        if isinstance(base['model'], dict):
            for missing in ['model', 'prompt']:
                changed = {key: value for key, value in base['model'].items() if key != missing}
                await observe(node, {**base, 'model': changed}, True)
        for seed in [1, 2147483647, -1, 2147483648, '3', 'invalid']:
            await observe(node, {**base, 'seed': seed}, seed in [-1, 2147483648, 'invalid'])
        prompt_container = base['model'] if isinstance(base['model'], dict) else base
        for prompt in ['', ' ', None, 123, '\u001csource\u001f']:
            arguments = {**base, 'model': {**base['model'], 'prompt': prompt}} if isinstance(base['model'], dict) else {**base, 'prompt': prompt}
            required_nonempty = node in ['Wan2TextToVideoApi', 'Wan2VideoEditApi', 'Wan2ReferenceVideoApi', 'HappyHorseTextToVideoApi', 'HappyHorseVideoEditApi', 'HappyHorseReferenceVideoApi']
            allows_none = node in ['Wan2ImageToVideoApi', 'Wan2VideoContinuationApi', 'HappyHorseImageToVideoApi']
            await observe(node, arguments, prompt == 123 or prompt is None and not allows_none or prompt == '' and required_nonempty)
        if node in ['WanTextToVideoApi', 'WanImageToVideoApi', 'Wan2TextToVideoApi', 'Wan2ImageToVideoApi']:
            for duration in [4.0, 2.0, 1.0, 29.000125, 60.0, 60.00025]:
                low, high = ((3, 29) if node in ['WanTextToVideoApi', 'WanImageToVideoApi'] else (1.5, 60) if node == 'Wan2TextToVideoApi' else (2, 30))
                await observe(node, {**base, 'audio': audio(duration)}, duration + 1 / 8000 < low or duration - 1 / 8000 > high)
        if node == 'WanImageToImageApi':
            for count in [0, 1, 3]: await observe(node, {**base, 'image': Tensor('image', count)}, count not in (1, 2))
            await observe(node, {**base, 'image': Tensor('image', 2, channels=4)})
        if node == 'WanImageToVideoApi':
            for count in [0, 2]: await observe(node, {**base, 'image': Tensor('image', count)}, True)
        if node == 'WanTextToVideoApi':
            for changes in [{'size': 'invalid'}, {'size': '480p: 1:1 (640x640)', 'model': 'wan2.6-t2v'}, {'duration': 15}, {'duration': 15, 'model': 'wan2.6-t2v'}, {'size': 'custom (960 × 512)', 'duration': 10}]:
                await observe(node, {**base, **changes}, changes.get('size') == 'invalid' or changes.get('model') == 'wan2.6-t2v' and '480p' in changes.get('size', '') or changes == {'duration': 15})
            await observe(node, {**base, 'duration': '5'}, True)
        if node == 'WanImageToVideoApi':
            await observe(node, {**base, 'resolution': '480P', 'model': 'wan2.6-i2v'}, True)
            await observe(node, {**base, 'duration': 15}, True)
            await observe(node, {**base, 'duration': '5'}, True)
        if node == 'WanReferenceVideoApi':
            await observe(node, {**base, 'reference_videos': OrderedMapping(z=Video('z'), a=Video('a', 1))}, True)
            await observe(node, {**base, 'reference_videos': OrderedMapping(z=Video('z', duration_error=True), a=Video('a'))})
            await observe(node, {**base, 'reference_videos': OrderedMapping()})
        if node == 'Wan2ImageToVideoApi': await observe(node, {**base, 'last_frame': Tensor('last_frame', channels=4), 'audio': audio(4)})
        if node == 'Wan2VideoContinuationApi':
            await observe(node, {**base, 'last_frame': Tensor('last_frame', channels=4), 'prompt': 'ignored', 'negative_prompt': 'ignored'})
        if node in ['Wan2VideoContinuationApi', 'Wan2VideoEditApi', 'HappyHorseVideoEditApi']:
            field = 'first_clip' if node == 'Wan2VideoContinuationApi' else 'video'
            for value in [Video(field, 1), Video(field, 61), Video(field, duration_error=True), Video(field, save_error=True)]: await observe(node, {**base, field: value}, value.duration in [1, 61] or value.save_error)
        if node == 'Wan2VideoEditApi':
            for duration in ['auto', '7', '-2', 'bad']:
                await observe(node, {**base, 'model': {**base['model'], 'duration': duration}}, duration == 'bad')
        if node == 'Wan2ReferenceVideoApi':
            for count in [0, 5, 6]:
                images = OrderedMapping((str(index), Tensor(str(index))) for index in range(count))
                await observe(node, {**base, 'model': {**model, 'reference_images': images}}, count in [0, 6])
        if node == 'HappyHorseReferenceVideoApi':
            await observe(node, {**base, 'model': model}, True)
            await observe(node, {**base, 'model': {**model, 'reference_images': OrderedMapping(z=Tensor('z', width=400, height=400), a=Tensor('a', width=399, height=400))}}, True)
        if node == 'HappyHorseImageToVideoApi':
            for width, height in [(299, 300), (300, 299), (750, 300), (751, 300)]: await observe(node, {**base, 'first_frame': Tensor('first_frame', width=width, height=height)}, width < 300 or height < 300 or width / height > 2.5)
        final = image_final if node in ['WanTextToImageApi', 'WanImageToImageApi'] else video_final
        for override in [{'generate': {'request_id': 'request-1', 'code': 'source-code', 'message': 'source-message'}}, {'generate': {}}, {'generate': {'output': {'task_id': '', 'task_status': 'PENDING'}, 'request_id': 'request-1'}}]:
            await observe(node, base, override['generate'].get('output') is None, override)
        for status in ['RUNNING', ' QUEUED ', ' FAILED ', 'weird', '']:
            pending = copy.deepcopy(final); pending['output']['task_status'] = status
            await observe(node, base, 'FAILED' in status, {'polls': [pending, final]})
        for status in [' COMPLETE ', 'cancelled', 'in_queue']:
            pending = copy.deepcopy(final); pending['output']['task_status'] = status
            await observe(node, base, status == 'cancelled', {'polls': [pending, final]})
        for missing in [{'request_id': 'request-1'}, {'output': {'task_id': 'task-1', 'task_status': 'SUCCEEDED'}, 'request_id': 'request-1'}]:
            await observe(node, base, True, {'polls': [missing]})
        if node in ['WanTextToImageApi', 'WanImageToImageApi']:
            for results, failure in [([], True), ([{}], True), ([{'url': '/cloud/image.png'}], True)]:
                response = copy.deepcopy(final); response['output']['results'] = results
                await observe(node, base, failure, {'polls': [response]})
        else:
            response = copy.deepcopy(final); response['output']['video_url'] = '/cloud/video.mp4'
            await observe(node, base, True, {'polls': [response]})
    first = bases['WanTextToImageApi']
    queued = copy.deepcopy(image_final); queued['output']['task_status'] = 'QUEUED'
    await observe('WanTextToImageApi', first, False, {'polls': [queued] * 501 + [image_final]})
    duration_probes = []
    estimate = next(keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'poll_op' for keyword in node.keywords if keyword.arg == 'estimated_duration' and isinstance(keyword.value, ast.BinOp))
    for duration in [5, 10, 15, '5', None]:
        probe = {'input': duration}
        try: probe['output'] = eval(compile(ast.Expression(body=copy.deepcopy(estimate)), '<actual-Wan-duration-estimate>', 'eval'), {'duration': duration})
        except Exception as error: probe['error_type'] = type(error).__name__
        duration_probes.append(probe)
    size_probes = []
    pattern = namespace['RES_IN_PARENS']
    for size in ['720p: 1:1 (960x960)', '480p: 1:1 (640x640)', 'custom (960 × 512)', 'invalid']:
        probe = {'input': size}
        try: probe['output'] = list(pattern.search(size).groups())
        except Exception as error: probe['error_type'] = type(error).__name__
        size_probes.append(probe)
    edit_duration_probes = []
    expression = next(node.value for method in ast.walk(tree) if isinstance(method, ast.AsyncFunctionDef) and method.name == 'execute' for node in method.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'duration' for target in node.targets) and isinstance(node.value, ast.IfExp))
    for duration in ['auto', '7', '-2', 'bad']:
        probe = {'input': duration}
        try: probe['output'] = eval(compile(ast.Expression(body=copy.deepcopy(expression)), '<actual-Wan-edit-duration>', 'eval'), {'model': {'duration': duration}})
        except Exception as error: probe['error_type'] = type(error).__name__
        edit_duration_probes.append(probe)
    return {'schema_version': 1, 'namespace': 'zed.sim.provider.wan', 'scope': 'actual14nodeexecute/models/helpers/PIL/wrappers and compressed lossless source traces; injected existing codec bytes, no native allocation or clock',
            'versions': {'python': sys.version.split()[0], 'aiohttp': aiohttp.__version__, 'numpy': np.__version__, 'pillow': Image.__version__, 'pydantic': pydantic.__version__},
            'source_sha256': {path: hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() for path in PATHS},
            'contracts': contracts, 'auth_headers': auth, 'cases': cases, 'ordered_errors': errors, 'response_errors': response_errors,
            'response_api_probes': api_probes, 'scalar_probes': scalar_probes, 'status_probes': status_probes, 'duration_probes': duration_probes, 'size_probes': size_probes, 'edit_duration_probes': edit_duration_probes,
            'initial_response': initial, 'image_response': image_final, 'video_response': video_final,
            'response_image_hex': response_png.hex(),
            'encoded_png_probes': {'rgb_2x2': base64.b64decode(source_image(Tensor())).hex(),
                                    'rgba_2x2': base64.b64decode(source_image(Tensor(channels=4))).hex(),
                                    'rgb_400x400': base64.b64decode(source_image(Tensor(width=400, height=400))).hex()},
            'helper_defaults': {name: {argument.arg: ast.unparse(default) for argument, default in zip(member.args.kwonlyargs, member.args.kw_defaults) if default is not None} for name in ['sync_op', 'poll_op', 'upload_image_to_comfyapi', 'upload_audio_to_comfyapi', 'upload_video_to_comfyapi'] for member in list(client.body) + list(upload_tree.body) if isinstance(member, ast.AsyncFunctionDef) and member.name == name}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    data = asyncio.run(generate()); encoded = json.dumps(data, indent=2, ensure_ascii=False) + '\n'
    target = ROOT / 'source-contract.json'
    if args.check:
        if target.read_text() != encoded: raise ValueError('Wan actual source changed')
    else: target.write_text(encoded)
    print('Wan', len(data['cases']), 'success', len(data['ordered_errors']), 'ordered source errors', len(data['response_errors']), 'response failures')
