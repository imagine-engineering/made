"""Render frame-placement timelines from the native execution test report."""
import colorsys
import json
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

rows = json.loads((Path(tempfile.gettempdir()) / 'sim-wandancer-padding-native-review.json').read_text())
selected = [2, 3, 4, 5, 6, 8, 9, 11, 15, 16]
image = Image.new('RGB', (1280, 1020), '#111823')
draw = ImageDraw.Draw(image)
draw.font = ImageFont.load_default(size=15)
draw.text((28, 18), 'WanDancer padding - actual native frame placement and audio slices', fill='white')
draw.text((28, 42), 'Numbers identify source frames; dark cells are padding. The lower strip is the valid-frame mask.', fill='#b6c2d3')
for ordinal, index in enumerate(selected):
    row = rows[index]
    case = row['case']
    values = row['actual_keyframes']
    mask = row['actual_mask']
    y = 82 + ordinal * 92
    draw.text((28, y), f"{case['images']} images | segment {case['segment_index']} | length {case['segment_length']} | input {case['audio_frames']} samples at {case['sample_rate']} Hz | slice {row['actual_audio_length']} samples", fill='white')
    cell = min(90, 1190 // max(len(values), 1))
    for position, (value, valid) in enumerate(zip(values, mask)):
        x = 28 + position * (cell + 3)
        color = '#273242'
        if value:
            color = tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb((value * 0.137) % 1, 0.55, 0.95))
        draw.rectangle((x, y + 23, x + cell, y + 53), fill=color)
        draw.text((x + 8, y + 31), str(int(value)) if value else '-', fill='#111823' if value else '#9aabbe')
        draw.rectangle((x, y + 57, x + cell, y + 66), fill='#e9f0f8' if valid else '#273242')
output = Path(__file__).with_name('native-padding-review.png')
image.save(output)
print(output)
