"""Execute pinned upload DTO/helper bodies and capture signed PUT header policy."""
import argparse
import ast
import asyncio
import contextlib
import copy
import hashlib
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pydantic
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / 'projects/sim/ComfyUI/sim_api_nodes/util/upload_helpers.py'
ANTHROPIC = ROOT / 'projects/sim/ComfyUI/sim_api_nodes/nodes_anthropic.py'
DESTINATION = Path(__file__).with_name('source_storage_upload.json')

class HeaderCapture(Exception):
    def __init__(self, headers, skip_auto_headers, data):
        self.headers = headers
        self.skip_auto_headers = sorted(skip_auto_headers)
        self.data = data

async def capture():
    namespace = {'__builtins__': __builtins__, 'BaseModel': BaseModel,
                 'Field': Field, 'BytesIO': BytesIO, 'contextlib': contextlib,
                 'HeaderCapture': HeaderCapture}
    selected = []
    for original in ast.parse(SOURCE.read_text()).body:
        if getattr(original, 'name', None) not in {'UploadRequest', 'UploadResponse',
                'upload_file_to_simapi', 'upload_file'}:
            continue
        node = copy.deepcopy(original)
        if isinstance(node, ast.AsyncFunctionDef):
            node.returns = None
            for argument in [*node.args.args, *node.args.kwonlyargs]:
                argument.annotation = None
        if node.name == 'upload_file':
            captured = []
            for statement in node.body:
                captured.append(statement)
                if isinstance(statement, ast.If) and isinstance(statement.test, ast.Name) and statement.test.id == 'content_type':
                    captured.append(ast.Raise(exc=ast.Call(func=ast.Name(id='HeaderCapture', ctx=ast.Load()), args=[ast.Name(id=name, ctx=ast.Load()) for name in ['headers', 'skip_auto_headers', 'data']], keywords=[])))
                    break
            assert isinstance(captured[-1], ast.Raise)
            node.body = captured
        selected.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(SOURCE), 'exec'), namespace)
    observations = []
    async def sync(cls, endpoint, data, response_model, **options):
        observations.append({'method': endpoint.method, 'path': endpoint.path,
                             'body': data.model_dump(exclude_none=True), 'options': options})
        return response_model.model_validate({'upload_url': 'https://fixture.invalid/signed',
                                             'download_url': 'https://fixture.invalid/download'})
    namespace.update(sync_op=sync, ApiEndpoint=lambda path, method: SimpleNamespace(path=path, method=method))
    cases = []
    for content_type, data in [(mime, b'\x00\xffsource\r\n') for mime in [None, 'image/png', 'audio/mp4', 'video/mp4']] + [(None, b'')]:
        observations.clear()
        try:
            await namespace['upload_file_to_simapi'](object, BytesIO(data), 'source.png', content_type)
        except HeaderCapture as captured:
            assert captured.data == data
            cases.append({'declared_content_type': content_type, 'file_name': 'source.png',
                          'bytes_hex': data.hex(), 'storage': observations[0],
                          'signed_put_headers': captured.headers,
                          'skip_auto_headers': captured.skip_auto_headers})
        else:
            raise AssertionError('actual signed PUT header boundary was not captured')
    anthropic_namespace = {'__builtins__': __builtins__,
        'AnthropicImageContent': lambda **values: values,
        'AnthropicImageSourceUrl': lambda **values: values}
    anthropic_nodes = []
    for original in ast.parse(ANTHROPIC.read_text()).body:
        if isinstance(original, ast.Assign) and any(isinstance(target, ast.Name) and target.id in {'CLAUDE_MAX_IMAGES', 'ANTHROPIC_IMAGE_MAX_PIXELS'} for target in original.targets):
            anthropic_nodes.append(copy.deepcopy(original))
        if getattr(original, 'name', None) == '_build_image_content_blocks':
            node = copy.deepcopy(original)
            node.returns = None
            for argument in node.args.args:
                argument.annotation = None
            anthropic_nodes.append(node)
    async def capture_images(cls, images, **options):
        observations.append(options)
        return ['https://fixture.invalid/image']
    anthropic_namespace['upload_images_to_simapi'] = capture_images
    exec(compile(ast.fix_missing_locations(ast.Module(body=anthropic_nodes, type_ignores=[])), str(ANTHROPIC), 'exec'), anthropic_namespace)
    observations.clear()
    await anthropic_namespace['_build_image_content_blocks'](object, [object()])
    anthropic_upload = {'source_sha256': hashlib.sha256(ANTHROPIC.read_bytes()).hexdigest(), 'options': observations[0]}
    assert 'mime_type' not in anthropic_upload['options']
    return {'anthropic_upload': anthropic_upload, 'schema_version': 1, 'pydantic_version': pydantic.__version__, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': cases}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    result = json.dumps(asyncio.run(capture()), indent=2) + '\n'
    if arguments.check:
        if DESTINATION.read_text() != result:
            raise SystemExit('immutable actual source upload fixture differs')
    else:
        DESTINATION.write_text(result)
    print('verified actual source storage/signed upload policy')
