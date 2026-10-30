import ast, csv, hashlib, json
from pathlib import Path
from types import SimpleNamespace
import torch
root = Path(__file__).resolve().parents[5]
source = root / 'projects/sim/ComfyUI'
identifiers = {'COMFY-NODE-' + str(i).zfill(4) for i in [149, 150, 151, 153, 154, 155, 156, 157, 158]}
namespace = {'torch': torch, 'io': SimpleNamespace(NodeOutput=lambda *values: values), 'IO': SimpleNamespace(NodeOutput=lambda *values: values), 'sim': SimpleNamespace(model_management=SimpleNamespace(intermediate_device=lambda: 'cpu', intermediate_dtype=lambda: torch.float32))}
cases = []
sources = {}
for row in csv.DictReader((root / '.agents/specs/sim-parity/catalogs/backend-nodes.csv').open()):
    if row['feature_id'] not in identifiers:
        continue
    path = source / row['source_file']
    sources[row['source_file']] = hashlib.sha256(path.read_bytes()).hexdigest()
    original = next((n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == row['class_name']))
    method = next((n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'execute'))
    selected = ast.ClassDef(name=original.name, bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=ast.parse('from __future__ import annotations').body + [selected], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    names = {n.arg for n in method.args.args + method.args.kwonlyargs} - {'cls'}
    if 'seconds' in names:
        inputs = [dict(seconds=seconds, batch_size=batch) for seconds in [1.0, 1.02, 1.06, 2.5] for batch in [1, 2]]
    else:
        inputs = []
        for width, height in [(64, 64), (65, 67), (96, 128)]:
            for batch in [1, 2]:
                for length in [1, 4, 5, 9] if 'length' in names else [None]:
                    values = dict(width=width, height=height, batch_size=batch)
                    if length is not None:
                        values['length'] = length
                    inputs.append(values)
    for values in inputs:
        output = namespace[original.name].execute(**values)[0]
        samples = output['samples']
        assert samples.dtype == torch.float32 and torch.count_nonzero(samples).item() == 0
        cases.append(dict(feature_id=row['feature_id'], class_type=row['node_identifier'], inputs=values, shape=list(samples.shape), metadata={key: value for key, value in output.items() if key != 'samples'}, dtype=str(samples.dtype), all_zero=True))
result = dict(provenance='Pinned source execute methods with actual CPU float32 PyTorch allocations; adapters supply only NodeOutput containers and the CPU intermediate device/dtype.', pytorch_version=torch.__version__, source_sha256=sources, cases=cases)
expected = json.loads(Path(__file__).with_name('source_empty_latents.json').read_text())
assert result['source_sha256'] == expected['source_sha256']
assert result['cases'] == expected['cases']
print(len(cases), 'actual-source empty latent cases')
