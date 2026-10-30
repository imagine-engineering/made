import ast, hashlib, json, math
from pathlib import Path
from types import SimpleNamespace
import torch
root = Path(__file__).resolve().parents[5] / 'projects/sim/ComfyUI'
namespace = {'torch': torch}
source_hashes = {}
for relative, names in [('sim/utils.py', {'common_upscale'}), ('sim_extras/nodes_latent.py', {'reshape_latent_to'}), ('sim_extras/nodes_post_processing.py', {'batch_latents'})]:
    path = root / relative
    source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    selected = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body + selected, type_ignores=[]), str(path), 'exec'), namespace)
namespace['sim'] = SimpleNamespace(utils=SimpleNamespace(common_upscale=namespace['common_upscale']))
shapes = [[1, 4, 2, 3], [2, 4, 2, 3], [1, 4, 3, 2], [1, 4, 5, 3], [1, 8, 2, 3], [1, 4, 2, 2, 3], [2, 4, 2, 3, 2], [1, 64, 5]]
cases = []
for first_shape in shapes:
    for second_shape in shapes:
        first_samples = torch.arange(math.prod(first_shape), dtype=torch.float32).reshape(first_shape) / 32
        second_samples = torch.arange(math.prod(second_shape), dtype=torch.float32).reshape(second_shape) / 16
        mask_shape = [1, 2, 3] if len(first_shape) != 3 else [1, 1, 3]
        mask = torch.ones(mask_shape)
        first = dict(samples=first_samples, noise_mask=mask, batch_index=[11 + i for i in range(first_shape[0])], downscale_ratio_spacial=8)
        second = dict(samples=second_samples, downscale_ratio_spacial=16)
        case = dict(first_shape=first_shape, second_shape=second_shape, mask_shape=mask_shape)
        try:
            output = namespace['batch_latents']([first, second])
            assert output['noise_mask'] is mask
            assert output['downscale_ratio_spacial'] == 8
            assert first['batch_index'] == [11 + i for i in range(first_shape[0])]
            case.update(output_shape=list(output['samples'].shape), output_values=output['samples'].flatten().tolist(), batch_indices=output['batch_index'], original_mask_retained=True)
        except Exception as error:
            case.update(error=type(error).__name__, message=str(error))
        cases.append(case)
result = dict(provenance='Actual pinned batch_latents, reshape_latent_to, and common_upscale functions executed with real CPU float32 PyTorch tensors; includes bilinear center-crop and temporal flatten/restore.', pytorch_version=torch.__version__, source_sha256=source_hashes, cases=cases)
expected = json.loads(Path(__file__).with_name('source_batch_latents.json').read_text())
assert result['source_sha256'] == expected['source_sha256']
assert result['cases'] == expected['cases']
print(len(cases), 'actual source cases;', sum(('error' not in case for case in cases)), 'successes;', sum(('error' in case for case in cases)), 'source exceptions')
