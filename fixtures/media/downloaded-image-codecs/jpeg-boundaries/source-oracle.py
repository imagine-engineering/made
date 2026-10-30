import argparse
import ast
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, features

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--generate', action='store_true')
arguments = parser.parse_args()
source = arguments.source.read_bytes()
source_hash = hashlib.sha256(source).hexdigest()
assert source_hash == '632476df7c448efcc80acc045b45a3f1c6cf44540c7883bb380615040cc9d7a2'
function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'bytesio_to_image_tensor')
namespace = {'torch': torch, 'np': np, 'Image': Image, 'BytesIO': io.BytesIO}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(arguments.source), 'exec'), namespace)
torch.set_num_threads(1)
root = Path(__file__).parent
sizes = sorted(set([(size, 5) for size in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33]] + [(9, size) for size in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33]] + [(size, size) for size in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33]]))
profiles = [('444', 'RGB', {'subsampling': 0}), ('422', 'RGB', {'subsampling': 1}), ('420', 'RGB', {'subsampling': 2}), ('progressive420', 'RGB', {'subsampling': 2, 'progressive': True}), ('gray', 'L', {}), ('cmyk', 'CMYK', {})]
cases = []
for width, height in sizes:
    for name, mode, options in profiles:
        filename = f'{name}-{width}x{height}'
        image = Image.new(mode, (width, height))
        pixels = [((x * 31 + y * 17 + x * y * 13) % 256, (x * 7 + y * 53 + x * y * 3) % 256, (x * 73 + y * 11 + x * y * 19) % 256, (x * 19 + y * 29) % 256) for y in range(height) for x in range(width)]
        image.putdata([pixel[0] if mode == 'L' else pixel[:3] if mode == 'RGB' else pixel for pixel in pixels])
        encoded_path = root / (filename + '.jpg')
        if arguments.generate:
            image.save(encoded_path, 'JPEG', quality=91, **options)
        encoded = encoded_path.read_bytes()
        tensor = namespace['bytesio_to_image_tensor'](io.BytesIO(encoded))
        assert list(tensor.shape) == [1, height, width, 4]
        rgba = Image.open(io.BytesIO(encoded)).convert('RGBA').tobytes()
        tensor_bytes = tensor.numpy().astype('<f4').tobytes()
        if arguments.generate:
            (root / (filename + '.rgba')).write_bytes(rgba)
            (root / (filename + '.f32le')).write_bytes(tensor_bytes)
        assert (root / (filename + '.rgba')).read_bytes() == rgba
        assert (root / (filename + '.f32le')).read_bytes() == tensor_bytes
        cases.append({'name': filename, 'encoded': filename + '.jpg', 'encoded_sha256': hashlib.sha256(encoded).hexdigest(), 'rgba_sha256': hashlib.sha256(rgba).hexdigest(), 'f32le_sha256': hashlib.sha256(tensor_bytes).hexdigest(), 'shape': list(tensor.shape), 'options': {'quality': 91, **options}})
malformed = [b'\xff\xd8', b'\xff\xd8\xff\xdb\x00\x01', b'\xff\xd8\xff\xc0\x00\x11\x08\x00\x00\x00\x00\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01']
failures = []
for index, encoded in enumerate(malformed):
    try:
        namespace['bytesio_to_image_tensor'](io.BytesIO(encoded))
        raise AssertionError('malformed source unexpectedly accepted')
    except (OSError, ValueError) as error:
        failures.append({'encoded_hex': encoded.hex(), 'error_type': type(error).__name__, 'message': re.sub(r'0x[0-9a-f]+', '0xADDRESS', str(error))})
manifest = {'source_sha256': source_hash, 'python': sys.version.split()[0], 'pillow': Image.__version__, 'libjpeg_turbo': features.version_feature('libjpeg_turbo'), 'torch': torch.__version__, 'numpy': np.__version__, 'cases': cases, 'malformed': failures}
if arguments.generate:
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
else:
    assert json.loads((root / 'manifest.json').read_text()) == manifest
print(json.dumps({'cases': len(cases), 'malformed': failures, 'source_sha256': source_hash}))
