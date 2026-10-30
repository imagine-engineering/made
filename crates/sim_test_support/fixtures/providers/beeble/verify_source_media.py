"""Compare Beeble PNG preparation with pinned source, without network calls."""
import argparse
import ast
import asyncio
import hashlib
import json
import math
import time
import uuid
import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / 'projects/sim/ComfyUI'
FIXTURE = Path(__file__).with_name('source_media.json')
SOURCES = {
    'sim/utils.py': '8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb',
    'sim_api_nodes/util/conversions.py': '632476df7c448efcc80acc045b45a3f1c6cf44540c7883bb380615040cc9d7a2',
    'sim_api_nodes/util/upload_helpers.py': 'aa761b97035139e84250579e58b85497fb64f6f40137ab8089757fbe2dde2c39',
    'sim_api_nodes/nodes_beeble.py': 'a7d98292ba01564a4a8519ba3c8caa30d9e0aaf42bf16ca55e29944222896812',
}


def source_namespace():
    namespace = dict(np=np, torch=torch, Image=Image, math=math, time=time, uuid=uuid,
                     BytesIO=BytesIO, mimetype_to_extension=lambda mime: 'png')
    for relative, names in [
        ('sim/utils.py', {'lanczos', 'common_upscale'}),
        ('sim_api_nodes/util/conversions.py', {'_compute_downscale_dims', 'downscale_image_tensor',
         'tensor_to_pil', 'tensor_to_bytesio', 'pil_to_bytesio', 'convert_mask_to_image'}),
        ('sim_api_nodes/util/upload_helpers.py', {'upload_images_to_simapi', 'upload_image_to_simapi'}),
        ('sim_api_nodes/nodes_beeble.py', {'_upload_mask_as_image'}),
    ]:
        path = SOURCE / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCES[relative], relative
        selected = [node for node in ast.parse(path.read_text()).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
        assert {node.name for node in selected} == names
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])),
                     str(path), 'exec'), namespace)
    return namespace


async def source_results():
    namespace = source_namespace()
    cases = []
    shapes = [[1, 2, 3, 3], [2, 8, 11, 3], [1, 8, 11, 4], [1, 8, 11, 1],
              [2, 8, 11, 1], [1, 1, 11, 3], [1, 11, 1, 3], [1, 1, 1, 3], [2, 8, 11, 4]]
    scenarios = [(role, shape) for role in ('source', 'reference') for shape in shapes]
    scenarios.extend(('mask', shape) for shape in ([8, 11], [1, 8, 11], [2, 8, 11], [2, 1, 11]))
    palette = [-0.25, 0.0, 0.125, 0.5, 0.75, 1.0, 1.25]
    for role, shape in scenarios:
        pixels = [palette[index % len(palette)] for index in range(math.prod(shape))]
        tensor = torch.tensor(pixels, dtype=torch.float32).reshape(shape)
        for maximum_pixels in (4, 16, 2770000):
            observed = []

            async def record_upload(cls, stream, filename, mime_type, *args):
                assert mime_type == 'image/png'
                image = Image.open(stream)
                observed.append(dict(width=image.width, height=image.height,
                                     mode=image.mode, pixels=list(image.tobytes())))
                return 'https://fixture.invalid/upload'

            namespace['upload_file_to_simapi'] = record_upload
            namespace['_MAX_PIXELS'] = maximum_pixels
            case = dict(role=role, shape=shape, pixels=pixels, maximum_pixels=maximum_pixels)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    if role == 'mask':
                        await namespace['_upload_mask_as_image'](None, tensor, wait_label='Uploading alpha')
                    else:
                        image = namespace['downscale_image_tensor'](tensor, maximum_pixels) if role == 'source' else tensor
                        await namespace['upload_image_to_simapi'](None, image, mime_type='image/png',
                            total_pixels=None if role == 'source' else maximum_pixels)
                assert len(observed) == 1
                case['output'] = observed[0]
            except (TypeError, ValueError, IndexError) as error:
                assert not observed
                case['error'] = type(error).__name__
            cases.append(case)
    return dict(schema_version=1, sources=SOURCES, cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    options = parser.parse_args()
    expected = asyncio.run(source_results())
    if options.write:
        FIXTURE.write_text(json.dumps(expected, indent=2) + '\n')
    else:
        assert json.loads(FIXTURE.read_text()) == expected
    successes = sum('output' in case for case in expected['cases'])
    print(f'PASS: {len(expected["cases"])} Beeble PNG source cases ({successes} images, {len(expected["cases"]) - successes} errors)')
