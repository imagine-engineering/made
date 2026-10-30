import argparse
import ast
import asyncio
import contextlib
import inspect
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

def generation_reference(root):
    path = root / 'projects/sim/ComfyUI/sim_extras/nodes_textgen.py'
    rows = json.loads((root / 'crates/sim_test_support/parity/catalogs/backend-node-contracts.json').read_text())['contracts']
    rows = [r for r in rows if r['feature_id'] in ['COMFY-NODE-0671', 'COMFY-NODE-0672']]
    assert all((r['source']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest() for r in rows))
    class Clip:

        def __init__(self):
            self.calls = []

        def tokenize(self, prompt, **kwargs):
            self.calls.append({'method': 'tokenize', 'prompt': prompt, **kwargs})
            return 'captured-tokens'

        def generate(self, tokens, **kwargs):
            self.calls.append({'method': 'generate', 'tokens': tokens, **kwargs})
            return 'captured-generated-ids'

        def decode(self, ids):
            self.calls.append({'method': 'decode', 'ids': ids})
            return 'captured-decoded-text'
    namespace = {'io': SimpleNamespace(ComfyNode=object, NodeOutput=lambda text: text)}
    selected = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name in ['TextGenerate', 'TextGenerateLTX2Prompt'] or (isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id.startswith('LTX2_') for t in n.targets)))]
    exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body + selected, type_ignores=[]), str(path), 'exec'), namespace)
    cases = []
    for name in ['TextGenerate', 'TextGenerateLTX2Prompt']:
        for image in [None, 'source-image-marker']:
            for sampling_mode in [{'sampling_mode': 'off'}, {'sampling_mode': 'on', 'temperature': 0.7, 'top_k': 64, 'top_p': 0.95, 'min_p': 0.05, 'repetition_penalty': 1.05, 'seed': 0}, {'sampling_mode': 'on', 'temperature': 2.0, 'top_k': 1000, 'top_p': 0.0, 'min_p': 1.0, 'repetition_penalty': 0.0, 'presence_penalty': 5.0, 'seed': 2 ** 64 - 1}]:
                for template in [True, False]:
                    clip = Clip()
                    inputs = {'prompt': 'Walk through the rain', 'max_length': 17, 'sampling_mode': sampling_mode, 'image': image, 'video': 'video-marker', 'audio': 'audio-marker', 'thinking': True, 'use_default_template': template}
                    output = namespace[name].execute(clip, **inputs)
                    cases.append({'class_type': name, 'inputs': inputs, 'calls': clip.calls, 'output': output})
    fixture = {'provenance': 'Pinned source TextGenerate AST with call-recording CLIP; choreography and parameter reference only, no model output claim', 'nodes': rows, 'cases': cases, 'ltx_system_prompts': {key: value for key, value in namespace.items() if key.startswith('LTX2_')}}
    return fixture

def transform_reference(root):
    rows = json.loads((root / 'crates/sim_test_support/parity/catalogs/backend-node-contracts.json').read_text())['contracts']
    rows = [r for r in rows if r['feature_id'] in [f'COMFY-NODE-{n:04}' for n in [649, 673, 674, 701]]]
    texts = ['', 'AbC HELLO world', 'Straße ﬃ İı ΟΣ ΣΟΣ', '\x1c\x1d\x1e\x1f\t  mixed\n\xa0\u2003\u202f', '👩\u200d💻 café é\n终点', '\u200bzero width\u200b', 'A\x00B', 'é' * 10001]
    cases = []
    for row in rows:
        path = root / 'projects/sim/ComfyUI' / row['source']['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['source']['sha256']
        node = next((n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == row['source']['symbol']['symbol']))
        function = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '_process'))
        function.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        for text in texts:
            lengths = [1, 2, 5, 77, 10000] if row['node_identifier'] == 'TruncateText' else [None]
            for length in lengths:
                inputs = {'texts': text}
                parameters = {'max_length': length} if length is not None else {}
                inputs.update(parameters)
                output = namespace['_process'](None, text, **parameters)
                cases.append({'class_type': row['node_identifier'], 'inputs': inputs, 'output': output})
    fixture = {'provenance': 'Pinned source text _process AST, Python Unicode scalar behavior; registered-node and runtime list mapping require separate validation', 'nodes': [{'feature_id': r['feature_id'], 'class_type': r['node_identifier'], 'source': r['source'], 'schema': r['schema']['portable']} for r in rows], 'cases': cases}
    return fixture


def mapping_reference(root):
    path = root / 'projects/sim/ComfyUI/execution.py'
    function = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.AsyncFunctionDef) and node.name == '_async_map_node_over_list')
    namespace = dict(asyncio=asyncio, inspect=inspect, is_class=inspect.isclass, CurrentNodeContext=lambda *args: contextlib.nullcontext(), ExecutionBlocker=type('ExecutionBlocker', (), {}), _ComfyNodeInternal=type('_ComfyNodeInternal', (), {}))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    class Transform:
        def execute(self, texts, max_length):
            return texts[:max_length]
    inputs = {'texts': ['ABCDE', 'é👩\u200d💻末', 'last'], 'max_length': [1, 3]}
    outputs = asyncio.run(namespace['_async_map_node_over_list']('prompt', 'node', Transform(), inputs, 'execute'))
    return {'provenance': 'Pinned canonical source execution mapper applied to truncation; no model output claim', 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'inputs': inputs, 'outputs': outputs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[5])
    options = parser.parse_args()
    fixture = json.loads(Path(__file__).with_name('fixture.json').read_text())
    assert generation_reference(options.repository) == fixture['source_generation']
    assert transform_reference(options.repository) == fixture['source_transforms']
    assert mapping_reference(options.repository) == fixture['source_mapping']
    print('Verified repeat-last mapping, 24 source generation/template calls and 64 Unicode/truncation cases without modifying fixtures')


if __name__ == '__main__':
    main()
