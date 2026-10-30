import argparse
import ast
import hashlib
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, features

SOURCE_SHA256 = '632476df7c448efcc80acc045b45a3f1c6cf44540c7883bb380615040cc9d7a2'
parser = argparse.ArgumentParser()
parser.add_argument('--generate', action='store_true')
parser.add_argument('--source', type=Path, required=True)
arguments = parser.parse_args()
source = arguments.source.read_bytes()
assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'bytesio_to_image_tensor')
namespace = {'torch': torch, 'np': np, 'Image': Image, 'BytesIO': io.BytesIO}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(arguments.source), 'exec'), namespace)
torch.set_num_threads(1)
root = Path(__file__).parent
width, height = 9, 5
rgb = Image.new('RGB', (width, height))
rgb.putdata([((x * 31 + y * 17) % 256, (x * 7 + y * 53) % 256, (x * 73 + y * 11) % 256) for y in range(height) for x in range(width)])
gray = Image.new('L', (width, height))
gray.putdata([(x * 31 + y * 17) % 256 for y in range(height) for x in range(width)])
cmyk = Image.new('CMYK', (width, height))
cmyk.putdata([((x * 31 + y * 17) % 256, (x * 7 + y * 53) % 256, (x * 73 + y * 11) % 256, (x * 19 + y * 29) % 256) for y in range(height) for x in range(width)])
rgba = rgb.convert('RGBA')
rgba.putdata([(*pixel, [0, 1, 32, 128, 254, 255][index % 6]) for index, pixel in enumerate(rgb.get_flattened_data())])
orientation = Image.Exif()
orientation[274] = 6
cases = [
    ('jpeg-rgb-444', rgb, 'JPEG', {'quality': 91, 'subsampling': 0}),
    ('jpeg-rgb-420', rgb, 'JPEG', {'quality': 91, 'subsampling': 2}),
    ('jpeg-progressive', rgb, 'JPEG', {'quality': 91, 'subsampling': 2, 'progressive': True}),
    ('jpeg-oriented', rgb, 'JPEG', {'quality': 91, 'subsampling': 0, 'exif': orientation}),
    ('jpeg-grayscale', gray, 'JPEG', {'quality': 91}),
    ('jpeg-cmyk', cmyk, 'JPEG', {'quality': 91}),
    ('webp-rgb-lossless', rgb, 'WEBP', {'lossless': True}),
    ('webp-rgb-lossy', rgb, 'WEBP', {'quality': 83}),
    ('webp-grayscale', gray, 'WEBP', {'lossless': True}),
    ('webp-alpha-lossless', rgba, 'WEBP', {'lossless': True, 'exact': True}),
    ('webp-alpha-lossy', rgba, 'WEBP', {'quality': 83, 'exact': True}),
    ('webp-oriented', rgba, 'WEBP', {'lossless': True, 'exact': True, 'exif': orientation}),
]
results = []
for name, image, codec, options in cases:
    encoded_path = root / (name + ('.jpg' if codec == 'JPEG' else '.webp'))
    rgba_path = root / (name + '.rgba')
    tensor_path = root / (name + '.f32le')
    if arguments.generate:
        image.save(encoded_path, format=codec, **options)
    encoded = encoded_path.read_bytes()
    decoded = Image.open(io.BytesIO(encoded))
    tensor = namespace['bytesio_to_image_tensor'](io.BytesIO(encoded))
    assert list(tensor.shape) == [1, height, width, 4]
    pixels = decoded.convert('RGBA').tobytes()
    tensor_bytes = tensor.numpy().astype('<f4').tobytes()
    assert tensor_bytes == b''.join(struct.pack('<f', np.float32(value) / np.float32(255)) for value in pixels)
    if arguments.generate:
        rgba_path.write_bytes(pixels)
        tensor_path.write_bytes(tensor_bytes)
    assert rgba_path.read_bytes() == pixels
    assert tensor_path.read_bytes() == tensor_bytes
    results.append({'name': name, 'encoded': encoded_path.name, 'encoded_sha256': hashlib.sha256(encoded).hexdigest(), 'rgba_sha256': hashlib.sha256(pixels).hexdigest(), 'f32le_sha256': hashlib.sha256(tensor_bytes).hexdigest(), 'shape': list(tensor.shape), 'decoded_mode': decoded.mode, 'exif_orientation': decoded.getexif().get(274)})
manifest = {'source_sha256': SOURCE_SHA256, 'python': sys.version.split()[0], 'pillow': Image.__version__, 'libjpeg_turbo': features.version_feature('libjpeg_turbo'), 'libwebp': features.version_module('webp'), 'torch': torch.__version__, 'numpy': np.__version__, 'cases': results}
if arguments.generate:
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
else:
    assert json.loads((root / 'manifest.json').read_text()) == manifest
print(json.dumps(manifest, indent=2))
