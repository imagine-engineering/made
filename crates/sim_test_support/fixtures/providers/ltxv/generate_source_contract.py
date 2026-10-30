#!/usr/bin/env python3
import argparse, ast, asyncio, copy, hashlib, itertools, json, re
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

FOLDER=Path(__file__).resolve().parent
SOURCE_ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
SOURCES=['sim_api_nodes/nodes_ltxv.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/_helpers.py']
class TensorProbe:
    def __init__(self,count=1): self.shape=(count,720,720,3);self.ndim=4

def serial(value):
    if isinstance(value,TensorProbe):return {'native_slot':'image','shape':list(value.shape)}
    if isinstance(value,dict):return {key:serial(member) for key,member in value.items()}
    if isinstance(value,(list,tuple)):return [serial(member) for member in value]
    return value

def strip(module):
    for node in ast.walk(module):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            node.returns=None
            for argument in node.args.args+node.args.kwonlyargs:argument.annotation=None
    return ast.fix_missing_locations(module)

async def generate():
    tree=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());requests=[];media=[];contracts={}
    raw=(FOLDER/'response-video.mp4').read_bytes()
    def video_file(buffer):return {'canonical_video_from_file':{'bytes':len(buffer.getvalue()),'sha256':hashlib.sha256(buffer.getvalue()).hexdigest()}}
    namespace={'__builtins__':__builtins__,'BytesIO':BytesIO,'classmethod':classmethod,'torch':SimpleNamespace(Tensor=TensorProbe),'IO':SimpleNamespace(NodeOutput=lambda *values:values),'InputImpl':SimpleNamespace(VideoFromFile=video_file),'ApiEndpoint':lambda path,method:SimpleNamespace(path=path,method=method)}
    from pydantic import BaseModel,Field
    namespace.update(BaseModel=BaseModel,Field=Field)
    async def sync(cls,endpoint,data,**options):
        requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(mode='json',exclude_none=True),'options':options});return raw
    async def upload(cls,image,**options):
        media.append({'image':serial(image),'operation':'upload_images','max_images':options.get('max_images',8),'mime_type':options.get('mime_type'),'wait_label':'Uploading','show_batch_index':True,'total_pixels':4194304});return ['https://storage.fixture.invalid/source-image.png']
    namespace.update(sync_op_raw=sync,upload_images_to_simapi=upload)
    helpers=ast.parse((SOURCE_ROOT/SOURCES[2]).read_text());selection=[copy.deepcopy(node) for node in helpers.body if isinstance(node,ast.FunctionDef) and node.name in {'validate_string','get_number_of_images'}];exec(compile(strip(ast.Module(body=selection,type_ignores=[])),'<source-validation>','exec'),namespace)
    selected=[];expression=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='MODELS_MAP':selected.append(copy.deepcopy(node))
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='PRICE_BADGE':expression=ast.literal_eval(next(keyword.value for keyword in node.value.keywords if keyword.arg=='expr'))
        if isinstance(node,ast.ClassDef) and node.name=='ExecuteTaskRequest':selected.append(copy.deepcopy(node))
        if isinstance(node,ast.ClassDef) and node.name in {'ImageToVideoNode','TextToVideoNode'}:
            method=next(copy.deepcopy(member) for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute');schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords};identifier=ast.literal_eval(fields['node_id']);defaults={argument.arg:ast.literal_eval(value) for argument,value in zip(method.args.args[-len(method.args.defaults):],method.args.defaults)}
            inputs={}
            for call in fields['inputs'].elts:
                options={keyword.arg:ast.literal_eval(keyword.value) for keyword in call.keywords if keyword.arg in {'options','default','optional'} and not isinstance(keyword.value,ast.Call)}
                inputs[ast.literal_eval(call.args[0])]=options
            inputs['model']['options']=['LTX-2 (Pro)','LTX-2 (Fast)']
            contracts[identifier]={'symbol':node.name,'arguments':[argument.arg for argument in method.args.args if argument.arg!='cls'],'defaults':defaults,'hidden':ast.unparse(fields['hidden']),'outputs':ast.unparse(fields['outputs']),'widgets':inputs}
            selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=[method],decorator_list=[]))
    exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<source-execute>','exec'),namespace)
    cases=[];errors=[]
    async def observe(node,arguments,failure=False):
        requests.clear();media.clear()
        try:result=await namespace[contracts[node]['symbol']].execute(**copy.deepcopy(arguments))
        except Exception as error:
            if not failure:raise
            errors.append({'input':{'node':node,'arguments':serial(arguments)},'message':str(error),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media)})
        else:
            if failure:raise ValueError('Expected source failure')
            cases.append({'input':{'node':node,'arguments':serial(arguments)},'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'outputs':serial(result)})
    baseline={'model':'LTX-2 (Fast)','prompt':'  source  ','duration':8,'resolution':'1920x1080'}
    for node in contracts:
        base={**baseline,**({'image':TensorProbe()} if node=='LtxvApiImageToVideo' else {})}
        await observe(node,base)
        for model,duration,resolution,fps,audio in itertools.product(namespace['MODELS_MAP'],[6,8,10,12,14,16,18,20],['1920x1080','2560x1440','3840x2160'],[25,50],[False,True]):
            arguments={**base,'model':model,'duration':duration,'resolution':resolution,'fps':fps,'generate_audio':audio};await observe(node,arguments,duration>10 and (model!='LTX-2 (Fast)' or resolution!='1920x1080' or fps!=25))
        for override in [{'prompt':''},{'prompt':' '},{'prompt':'x'*10001},{'model':'unknown'}]:await observe(node,{**base,**override},True)
        await observe(node,{**base,'fps':None,'generate_audio':None})
        await observe(node,{**base,'fps':'25','generate_audio':'true'})
        if node=='LtxvApiImageToVideo':
            for count in [0,2]:await observe(node,{**base,'image':TensorProbe(count)},True)
    rate_text=re.search(r'\$prices\s*:=\s*(\{.*?\});',expression,re.S).group(1);rates=json.loads(rate_text);prices=[]
    for model,resolution,duration in itertools.product(namespace['MODELS_MAP'],['1920x1080','2560x1440','3840x2160'],[6,8,10,12,14,16,18,20]):prices.append({'model':model,'resolution':resolution,'duration':duration,'maximum_microunits':int(Decimal(str(rates[model.lower()][resolution]))*duration*1000000)})
    client=ast.parse((SOURCE_ROOT/SOURCES[1]).read_text());raw_method=next(node for node in client.body if isinstance(node,ast.AsyncFunctionDef) and node.name=='sync_op_raw');client_defaults={argument.arg:ast.literal_eval(default) for argument,default in zip(raw_method.args.kwonlyargs,raw_method.args.kw_defaults) if default is not None and isinstance(default,ast.Constant)}
    return {'schema_version':1,'namespace':'zed.sim.provider.ltxv','scope':'Actual pinned source AST/API/validation request branches; canonical upload/video allocation are injected owners. Price rates are extracted from the exact source JSONata literal and declared duration multiplication, not proof of the JSONata engine.','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'contracts':contracts,'models':namespace['MODELS_MAP'],'request_api_schema':namespace['ExecuteTaskRequest'].model_json_schema(),'cases':cases,'preflight_errors':errors,'price_expression':expression,'price_rates':rates,'price_cases':prices,'client_defaults':client_defaults,'response_sha256':hashlib.sha256(raw).hexdigest(),'response_bytes':len(raw),'response_provenance':json.loads((FOLDER/'media-provenance.json').read_text())}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');arguments=parser.parse_args();SOURCE_ROOT=arguments.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode();destination=FOLDER/'source-contract.json'
    if arguments.check:
        if destination.read_bytes()!=encoded:raise SystemExit('LTXV source oracle differs')
    else:destination.write_bytes(encoded)
