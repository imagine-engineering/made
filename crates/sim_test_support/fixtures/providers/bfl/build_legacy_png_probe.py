#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--source-directory', type=Path, required=True)
parser.add_argument('--recipe-directory', type=Path, required=True)
parser.add_argument('--target-directory', type=Path, required=True)
parser.add_argument('--execute', action='store_true')
arguments = parser.parse_args()
source = arguments.source_directory.resolve(strict=True)
recipe = arguments.recipe_directory.resolve(strict=True)
projection = recipe / 'generated_projection.rs'
manifest_path = recipe / 'manifest-template.json'
manifest = json.loads(manifest_path.read_text())
if manifest['identifier'] != 'zed.sim.provider.bfl' or len(manifest['nodes']) != 10:
    raise SystemExit('canonical complete BFL manifest is required')
binding = manifest['provider_binding']
if binding['implementation_namespace'] != manifest['identifier'] or len(binding['bindings']) != 10:
    raise SystemExit('exact complete namespace binding set is required')
if {node['id'] for node in manifest['nodes']} != {claim['node_id'] for claim in binding['bindings']}:
    raise SystemExit('node/binding set mismatch')
if len({claim['node_id'] for claim in binding['bindings']}) != 10:
    raise SystemExit('duplicate binding')
if not any(claim['node_id'] == 'FluxProUltraImageNode' and claim['feature_id'] == 'COMFY-NODE-0188' for claim in binding['bindings']):
    raise SystemExit('exact Ultra Source claim is required')
inputs = {str(path): digest(path) for path in [projection, manifest_path, *[source/name for name in ['Cargo.toml','Cargo.lock','guest.rs','rebuild_fixture.rs']]]}
target = arguments.target_directory.resolve()
module = target/'wasm32-unknown-unknown/release/sim_provider_bfl_legacy_png_process_probe_component.wasm'
artifact = target/'unsigned-bfl-legacy-png-process-probe.wasm'
commands = [
 ['cargo','build','--locked','--offline','--manifest-path',str(source/'Cargo.toml'),'--target','wasm32-unknown-unknown','--release','--lib'],
 ['cargo','run','--locked','--offline','--manifest-path',str(source/'Cargo.toml'),'--bin','rebuild_fixture','--',str(module),str(artifact)],
]
if not arguments.execute:
    print(json.dumps({'state':'PREPARED_UNRUN','inputs':inputs,'commands':commands,'qualification':'ComponentEncoder.validate(true) plus scheduled real host preflight/process are required; magic or a recipe is not acceptance.'},indent=2))
    raise SystemExit(0)
# Coordinator must run this driver under its reviewed created-PGID/resource supervisor.
target.mkdir(parents=True, exist_ok=False)
result_path = target/'build-results.json'
environment = os.environ.copy()
environment.update({'CARGO_TARGET_DIR':str(target),'CARGO_BUILD_JOBS':'1','CARGO_INCREMENTAL':'0','SIM_BFL_PNG_GUEST_PROJECTION_RS':str(projection)})
results = {'state':'ACTIVE','inputs':inputs,'commands':[],'acceptance':'UNRUN'}
with result_path.open('x') as output:
    json.dump(results,output,indent=2)
for index, command in enumerate(commands):
    entry={'argv':command,'state':'ACTIVE','started_monotonic':time.monotonic()}
    results['commands'].append(entry)
    result_path.write_text(json.dumps(results,indent=2)+'\n')
    try:
        with (target/f'cargo-{index}.log').open('xb') as output:
            completed=subprocess.run(command,env=environment,stdout=output,stderr=subprocess.STDOUT,timeout=3600,check=False)
        entry.update({'exit_code':completed.returncode,'duration_seconds':time.monotonic()-entry['started_monotonic'],'state':'PASS' if completed.returncode==0 else 'FAIL'})
    except BaseException as error:
        entry.update({'state':'ERROR','error':repr(error),'duration_seconds':time.monotonic()-entry['started_monotonic']})
        results['state']='ERROR'
        result_path.write_text(json.dumps(results,indent=2)+'\n')
        raise
    result_path.write_text(json.dumps(results,indent=2)+'\n')
    if completed.returncode:
        results['state']='FAIL'
        result_path.write_text(json.dumps(results,indent=2)+'\n')
        raise SystemExit(completed.returncode)
if inputs != {path:digest(Path(path)) for path in inputs}:
    results['state']='FAIL_INPUT_CHANGED'
    result_path.write_text(json.dumps(results,indent=2)+'\n')
    raise SystemExit('immutable guest recipe/source changed during build')
results.update({'state':'BUILD_PASS_HOST_PROCESS_UNRUN','artifact':{'path':str(artifact),'sha256':digest(artifact),'bytes':artifact.stat().st_size},'acceptance':'Actual ComponentEncoder.validate(true) ran; genuine signed host preflight/private process still UNRUN.'})
result_path.write_text(json.dumps(results,indent=2)+'\n')
