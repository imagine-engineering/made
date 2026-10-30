"""Execute pinned BFL request AST with explicit canonical-media boundary probes."""
import argparse
import ast
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE_ROOT = Path(__file__).resolve().parents[5] / 'projects/sim/ComfyUI'
DESTINATION = Path(__file__).with_name('source-contract.json')
SOURCES = ['sim_api_nodes/nodes_bfl.py', 'sim_api_nodes/apis/bfl.py', 'sim_api_nodes/util/client.py', 'sim_api_nodes/util/_helpers.py', 'sim_api_nodes/util/validation_utils.py', 'sim_api_nodes/util/download_helpers.py', 'sim_api_nodes/util/conversions.py']

class MediaProbe:
    def __init__(self, encoded='aW1hZ2U=', shape=(1, 256, 256, 3)):
        self.encoded, self.shape = encoded, shape
    def __getitem__(self, key):
        return MediaProbe(self.encoded, self.shape[1:] if isinstance(key, int) else self.shape)

class RequestProbe:
    defaults = {}
    def __init__(self, **fields):
        self.fields = {**self.defaults, **fields}
    def model_dump(self, exclude_none=True):
        return {key: value for key, value in self.fields.items() if value is not None}

def field(default=None, **unused):
    return default

def strip_annotations(node):
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            child.returns = None
            for argument in [*child.args.posonlyargs, *child.args.args, *child.args.kwonlyargs]:
                argument.annotation = None
        if isinstance(child, ast.AnnAssign):
            child.annotation = ast.Name(id='object', ctx=ast.Load())
    return ast.fix_missing_locations(node)

async def generate():
    requests = []
    async def sync_op(cls, endpoint, response_model, data):
        requests.append({'path': endpoint.path, 'method': endpoint.method, 'body': data.model_dump(exclude_none=True)})
        return SimpleNamespace(polling_url='https://poll.fixture.invalid/task', cost=3.75)
    async def poll_op(*arguments, **options):
        return SimpleNamespace(result={'sample': 'https://download.fixture.invalid/image.png'})
    async def download(url):
        return url
    namespace = {'__builtins__': __builtins__, 'classmethod': classmethod, 'BaseModel': RequestProbe, 'Field': field,
                 'IO': SimpleNamespace(ComfyNode=object), 'torch': SimpleNamespace(Tensor=MediaProbe),
                 'ApiEndpoint': lambda path, method='GET': SimpleNamespace(path=path, method=method),
                 'sync_op': sync_op, 'poll_op': poll_op, 'download_url_to_image_tensor': download,
                 'tensor_to_base64_string': lambda tensor, **options: tensor.encoded,
                 'resize_mask_to_image': lambda mask, image: mask,
                 'convert_mask_to_image': lambda mask: mask,
                 'get_number_of_images': lambda tensor: tensor.shape[0] if len(tensor.shape)==4 else 1,
                 'validate_string': lambda *arguments, **options: None,
                 'validate_aspect_ratio_string': lambda *arguments, **options: None,
                 'validate_image_dimensions': lambda *arguments, **options: None,
                 'BFLStatus': SimpleNamespace(ready='Ready', pending='Pending', request_moderated='Request Moderated', content_moderated='Content Moderated', error='Error', task_not_found='Task not found'),
                 'BFLFluxStatusResponse': object, 'BFLFluxProGenerateResponse': object}
    namespace['IO'].NodeOutput = lambda value: value
    validation = ast.parse((SOURCE_ROOT / SOURCES[4]).read_text())
    names = {'get_image_dimensions', 'validate_image_dimensions', 'validate_aspect_ratio_string', 'validate_string', '_ratio_from_tuple', '_assert_ratio_bounds', '_parse_aspect_ratio_string'}
    functions = [copy.deepcopy(node) for node in validation.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(strip_annotations(ast.Module(body=functions, type_ignores=[])), '<pinned-bfl-validation>', 'exec'), namespace)
    models = ast.parse((SOURCE_ROOT / SOURCES[1]).read_text())
    for node in models.body:
        if isinstance(node, ast.ClassDef) and node.name.endswith('Request'):
            defaults = {}
            for member in node.body:
                if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                    defaults[member.target.id] = eval(compile(ast.Expression(member.value), '<pinned-request-default>', 'eval'), namespace)
            namespace[node.name] = type(node.name, (RequestProbe,), {'defaults': defaults})
    tree = ast.parse((SOURCE_ROOT / SOURCES[0]).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '_FLUX2_MODEL_ENDPOINTS' for target in node.targets):
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name.startswith('Flux'):
            node = copy.deepcopy(node)
            node.body = [member for member in node.body if isinstance(member, ast.Assign) or isinstance(member, ast.AsyncFunctionDef) and member.name=='execute']
            selected.append(node)
    exec(compile(strip_annotations(ast.Module(body=selected, type_ignores=[])), '<pinned-bfl-execute>', 'exec'), namespace)
    base = {'prompt':'  cloud ☁  ', 'seed':18446744073709551615}
    cases = []
    def candidate(node, **extra):
        return {'node':node, 'prompt':base['prompt'], 'seed':base['seed'], 'width':1024, 'height':768, 'aspect_ratio':'16:9', 'guidance':3.15, 'steps':50, 'prompt_upsampling':False, 'raw':False, 'image_prompt_strength':0.125, 'top':0,'bottom':32,'left':64,'right':96,'dilate_pixels':10,'model':'Flux.2 [pro]', 'images':[], **extra}
    media=MediaProbe()
    mask=MediaProbe('bWFzaw==', (1,256,256))
    async def observe(input, kwargs):
        requests.clear()
        await namespace[input['node']].execute(**kwargs)
        if len(requests)!=1: raise ValueError('expected one pinned source request')
        cases.append({'input':input, 'request':requests[0]})
    for node in ['FluxProUltraImageNode','FluxKontextProImageNode','FluxKontextMaxImageNode']:
        value=candidate(node)
        kwargs={'prompt':value['prompt'], 'aspect_ratio':value['aspect_ratio'], 'seed':value['seed'], 'prompt_upsampling':False}
        if node=='FluxProUltraImageNode': kwargs.update(raw=False, image_prompt=media, image_prompt_strength=value['image_prompt_strength']); value['images']=['aW1hZ2U=']
        else: kwargs.update(guidance=value['guidance'],steps=50,input_image=media); value['images']=['aW1hZ2U=']
        await observe(value,kwargs)
    for node in ['FluxProExpandNode','FluxProFillNode']:
        value=candidate(node, guidance=60.0,images=['aW1hZ2U='])
        kwargs={'image':media,'prompt':value['prompt'],'prompt_upsampling':False,'steps':50,'guidance':60.0,'seed':value['seed']}
        if node=='FluxProExpandNode': kwargs.update(top=0,bottom=32,left=64,right=96)
        else: kwargs['mask']=mask;value['mask']='bWFzaw=='
        await observe(value,kwargs)
    value=candidate('FluxEraseNode',seed=2147483647,images=['aW1hZ2U='],mask='bWFzaw==')
    await observe(value,{'image':media,'mask':mask,'dilate_pixels':10,'seed':value['seed']})
    value=candidate('FluxVTONode',prompt='',images=['aW1hZ2U=','aW1hZ2U='])
    await observe(value,{'person':media,'garment':media,'prompt':'','seed':value['seed']})
    for node in ['Flux2ProImageNode','Flux2MaxImageNode']:
        value=candidate(node,images=['aW1hZ2U=']*9)
        await observe(value,{'prompt':value['prompt'],'width':1024,'height':768,'seed':value['seed'],'prompt_upsampling':False,'images':MediaProbe(shape=(9,256,256,3))})
    for model in ['Flux.2 [pro]','Flux.2 [max]']:
        value=candidate('Flux2ImageNode',model=model,images=['aW1hZ2U=']*8)
        await observe(value,{'prompt':value['prompt'],'seed':value['seed'],'model':{'model':model,'width':1024,'height':768,'images':{'image_1':MediaProbe(shape=(8,256,256,3))}}})
    return {'schema_version':1,'namespace':'zed.sim.provider.bfl','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES}, 'scope':'Actual pinned node.execute request AST and request-model defaults; actual scalar/aspect/dimension validation helpers; canonical image encoding/mask arithmetic, transport, and image decoding are explicit unexecuted boundaries in this development oracle.', 'cases':cases}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);args=parser.parse_args();SOURCE_ROOT=args.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode()
    if args.check:
        if DESTINATION.read_bytes()!=encoded: raise SystemExit('BFL source contract differs')
    else: DESTINATION.write_bytes(encoded)
