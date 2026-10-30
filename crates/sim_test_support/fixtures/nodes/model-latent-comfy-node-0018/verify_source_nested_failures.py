import ast, json, hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
root = Path(__file__).resolve().parents[5] / 'projects/sim/ComfyUI'
namespace = {'torch': torch}
sources = {}
for relative, names in [('sim/nested_tensor.py', {'NestedTensor'}), ('sim_extras/nodes_latent.py', {'reshape_latent_to'}), ('sim_extras/nodes_post_processing.py', {'batch_latents'}), ('sim/utils.py', {'common_upscale'})]:
    path = root / relative
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body + nodes, type_ignores=[]), str(path), 'exec'), namespace)
namespace['sim'] = SimpleNamespace(utils=SimpleNamespace(common_upscale=namespace['common_upscale']))
cases = []
for side in [2, 3]:
    first = namespace['NestedTensor']([torch.zeros([1, 16, 2, 2, 2]), torch.zeros([1, 64, 4])])
    second = namespace['NestedTensor']([torch.zeros([1, 16, 2, side, side]), torch.zeros([1, 64, 4])])
    try:
        namespace['batch_latents']([{'samples': first}, {'samples': second}])
        raise AssertionError('Source unexpectedly accepted nested batching')
    except (TypeError, AttributeError) as error:
        cases.append(dict(second_side=side, error=type(error).__name__, message=str(error)))
result = dict(provenance='Actual pinned NestedTensor and batch_latents/reshape_latent_to/common_upscale with real PyTorch tensors.', pytorch_version=torch.__version__, source_sha256=sources, cases=cases)
expected = json.loads(Path(__file__).with_name('source_nested_failures.json').read_text())
assert result['source_sha256'] == expected['source_sha256']
assert result['cases'] == expected['cases']
print(json.dumps(cases))
