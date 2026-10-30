import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

source_path = Path('projects/sim/ComfyUI/sim_extras/nodes_sdpose.py')
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == "d9b38524b1ee4b09b0ba4373537fb53526f08a6b2d8f714bb80225379342ee21"
source = source_path.read_text()
module = ast.parse(source)
nodes = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {'get_face_bboxes', 'SDPoseFaceBBoxes'}]
assert len(nodes) == 2
namespace = {'np': np, 'io': SimpleNamespace(ComfyNode=object, NodeOutput=lambda value: value)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), 'exec'), namespace)

def person(points, scores=None):
    points = (points * (70 // len(points) + 1))[:70]
    return {'face_keypoints_2d': [value for index, (x, y) in enumerate(points) for value in (x, y, 1.0 if scores is None else scores[index % len(scores)])]}

def frame(width, height, people):
    return {'canvas_width': width, 'canvas_height': height, 'people': people}

rectangle = person([(20, 25), (40, 25), (40, 35), (20, 35)])
edge = person([(-1, 2), (8, 2), (8, 16), (1, 16)], [0.0, 0.01, 1.0, 0.0])
rounding = person([(101.0000001, 83.1), (148.3, 83.1), (148.3, 120.300001), (101.0000001, 120.300001)])
small = person([(1.5, 1.5), (10.25, 1.5), (10.25, 5.75), (1.5, 5.75)])
scenarios = [
    ('rectangle', [frame(100, 80, [rectangle])]),
    ('zero_score_and_negative_points', [frame(100, 80, [edge])]),
    ('float32_normalization', [frame(997, 743, [rounding])]),
    ('small_canvas', [frame(13, 7, [small])]),
    ('degenerate', [frame(100, 80, [person([(12.25, 40.0)])])]),
    ('no_people', [frame(100, 80, [])]),
    ('no_frames', []),
    ('multiple_frames_and_people', [frame(100, 80, [rectangle, edge]), frame(997, 743, [rounding]), frame(13, 7, [])]),
]
cases = []
for name, frames in scenarios:
    for scale in [1.0, 1.5, 10.0]:
        for force_square in [False, True]:
            output = namespace['SDPoseFaceBBoxes'].execute(frames, scale, force_square)
            cases.append({'id': f'{name}-scale-{scale}-square-{force_square}', 'frames': frames, 'scale': scale, 'force_square': force_square, 'expected': output})
result = {'source_path': str(source_path), 'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(), 'oracle': 'Pinned source AST execution with NumPy', 'numpy_version': np.__version__, 'cases': cases}
fixture = json.loads(Path(__file__).with_name('fixture.json').read_text())
assert result == fixture['face_reference']
print('Validated', len(cases), 'pinned-source face reference cases')
