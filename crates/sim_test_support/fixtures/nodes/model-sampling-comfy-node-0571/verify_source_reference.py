import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--repository", type=Path)
arguments = parser.parse_args()
ROOT = arguments.repository or Path(__file__).resolve().parents[5]
FIXTURE = Path(__file__).with_name("fixture.json")
SOURCE = ROOT / 'projects/sim/ComfyUI'
IDENTIFIERS = {f'COMFY-NODE-{number:04}' for number in [571,572,573,574,575,576,577,578,580,581]}
contracts = json.loads((ROOT/'crates/sim_test_support/parity/catalogs/backend-node-contracts.json').read_text())['contracts']
rows = [row for row in contracts if row['feature_id'] in IDENTIFIERS]
assert len(rows) == 10

def capture(name, parameters):
    parameters = dict(parameters)
    if 'noise_scaler' in parameters:
        scaler = parameters.pop('noise_scaler')
        parameters['noise_scaler'] = None if scaler is None else [scaler(value) for value in [0.0,0.25,1.0,2.0]]
    return {'algorithm':name,'parameters':parameters}

def sample_lcm_upscale():
    raise AssertionError('This probe captures source node parameters only, not sampler execution')

namespace = {'io':SimpleNamespace(ComfyNode=object,NodeOutput=lambda value:value),
    'sim':SimpleNamespace(samplers=SimpleNamespace(ksampler=capture,
        KSAMPLER=lambda function,extra_options:capture(function.__name__,extra_options))),
    'sample_lcm_upscale':sample_lcm_upscale}
source_hashes = {}
for relative in sorted({row['source']['path'] for row in rows}):
    path=SOURCE/relative
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(row['source']['sha256']==digest for row in rows if row['source']['path']==relative)
    source_hashes[relative]=digest
    wanted={row['node_identifier'] for row in rows if row['source']['path']==relative}
    selected=[node for node in ast.parse(path.read_text()).body if isinstance(node,ast.ClassDef) and node.name in wanted]
    assert len(selected)==len(wanted)
    exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+selected,type_ignores=[]),str(path),'exec'),namespace)

cases=[]
for row in rows:
    fields=row['schema']['portable']['inputs']
    def literal(value):
        if value['kind']=='preserved_expression':
            expression=json.loads(value['source'])
            assert expression['kind']=='unary' and expression['operator']=='USub' and expression['operand']['kind']=='literal'
            return -expression['operand']['value']
        return float(value['value']) if value['kind']=='finite_decimal' else value['value']
    defaults={field['name']:literal(field['default']) if field['default'] is not None else literal(field['choices'][0]) for field in fields}
    variants=[('defaults',defaults)]
    for field in fields:
        candidates=[literal(value) for value in field['choices']]
        candidates += [literal(field[key]) for key in ['minimum','maximum'] if field[key] is not None]
        for index,value in enumerate(candidates):
            inputs=dict(defaults);inputs[field['name']]=value
            variants.append((f"{field['name']}-{index}",inputs))
    choices=[field for field in fields if field['choices']]
    for values in itertools.product(*[[literal(value) for value in field['choices']] for field in choices]):
        inputs=dict(defaults);inputs.update({field['name']:value for field,value in zip(choices,values)})
        variants.append(('choices-'+str(values),inputs))
    if row['node_identifier']=='SamplerER_SDE':
        for solver in ['ER-SDE','Reverse-time SDE','ODE']:
            for eta in [0.0,0.5,2.0]:
                variants.append((f'{solver}-{eta}',dict(defaults,solver_type=solver,eta=eta,s_noise=3.0)))
    seen=set()
    for label,inputs in variants:
        key=json.dumps(inputs,sort_keys=True)
        if key in seen:continue
        seen.add(key)
        output=namespace[row['node_identifier']].execute(**inputs)
        cases.append({'class_type':row['node_identifier'],'case':label,'inputs':inputs,'output':output})
fixture={'nodes':[{'feature_id':row['feature_id'],'class_type':row['node_identifier'],'source':row['source'],'schema':row['schema']['portable']} for row in rows],
    'source_reference':{'provenance':'Pinned source node AST with sampler-construction capture; no numerical sampler-execution claim', 'source_sha256':source_hashes,'cases':cases}}
assert fixture == json.loads(FIXTURE.read_text()), "Source sampler constructor references changed"
print(f'PASS: {len(cases)} pinned source constructor projections and {len(rows)} exact portable contracts')
