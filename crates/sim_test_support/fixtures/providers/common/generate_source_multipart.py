"""Capture exact aiohttp FormData bytes without invoking native encoding."""
import argparse
import asyncio
import hashlib
import json
from io import BytesIO
from pathlib import Path

import aiohttp

DESTINATION = Path(__file__).with_name('source_multipart.json')
CASES = [
    [('prompt', None, 'text/plain; charset=utf-8', '☁ source'.encode()),
     ('files', 'one.bin', 'application/octet-stream', b'\x00\xff\r\n'),
     ('files', 'two.bin', 'application/octet-stream', b'next')],
    [('prompt', None, 'text/plain; charset=utf-8', b''),
     ('file', 'empty.bin', 'application/octet-stream', b'')],
    [('file', 'image.png', 'image/png', b'\x89PNG\r\n\x1a\n')],
]

URLENCODED_CASES = [('quote-plus', [('field', '~ + /?&= 🌊')]),
 ('ordered-repeat-empty', [('same', 'first'), ('same', 'second'), ('empty', '')]),
 ('unicode-space', [('unicode', 'é漢字'), ('spaces', ' a b ')]),
 ('all-ascii',
  [('ascii',
    '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n'
    '\x0b'
    '\x0c'
    '\r'
    '\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c'
    '\x1d'
    '\x1e'
    '\x1f '
    '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f')]),
 ('empty-form', []),
 ('empty-names', [('', ''), ('', 'two')]),
 ('escaped-names', [('a&=+ []~', 'value')]),
 ('literal-percent', [('field', '%20+*')])]

async def capture_urlencoded():
    captured = []
    for name, fields in URLENCODED_CASES:
        form = aiohttp.FormData(fields)
        payload = form()
        class Writer:
            def __init__(self):
                self.data = bytearray()
            async def write(self, data):
                self.data.extend(data)
        writer = Writer()
        await payload.write(writer)
        captured.append({'name': name,
                         'fields': [{'name': key, 'value': value} for key, value in fields],
                         'body_utf8': writer.data.decode('utf-8'),
                         'content_type': payload.headers['Content-Type']})
    return {'cases': captured}

async def capture():
    captured = []
    for fields in CASES:
        form = aiohttp.FormData()
        for name, filename, content_type, data in fields:
            value = data.decode('utf-8') if filename is None else BytesIO(data)
            form.add_field(name, value, filename=filename, content_type=content_type)
        payload = form()
        payload._boundary = b'sim-source-boundary'
        class Writer:
            def __init__(self):
                self.data = bytearray()
            async def write(self, data):
                self.data.extend(data)
        writer = Writer()
        await payload.write(writer)
        captured.append({'fields': [{'name': name, 'file_name': filename,
                                    'media_type': content_type, 'bytes_hex': data.hex()}
                                   for name, filename, content_type, data in fields],
                         'body_hex': writer.data.hex()})
    return {'schema_version': 1, 'aiohttp_version': aiohttp.__version__,
            'formdata_sha256': hashlib.sha256(Path(aiohttp.formdata.__file__).read_bytes()).hexdigest(),
            'multipart_sha256': hashlib.sha256(Path(aiohttp.multipart.__file__).read_bytes()).hexdigest(),
            'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'boundary': 'sim-source-boundary', 'cases': captured,
            'urlencoded': await capture_urlencoded()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    result = json.dumps(asyncio.run(capture()), indent=2, ensure_ascii=False) + '\n'
    if arguments.check:
        if DESTINATION.read_text() != result:
            raise SystemExit('immutable source multipart fixture differs')
    else:
        DESTINATION.write_text(result)
    print('actual aiohttp FormData source fixture verified' if arguments.check else 'captured actual aiohttp FormData source fixture')
