"""Execute pinned HitPaw request/preflight AST while keeping native pixels at owner boundaries."""
import argparse, ast, asyncio, copy, hashlib, json, logging, math
from pathlib import Path
from types import SimpleNamespace
SOURCE_ROOT=Path(__file__).resolve().parents[5]/'projects/sim/ComfyUI'
DESTINATION=Path(__file__).with_name('source-contract.json')
SOURCES=['sim_api_nodes/nodes_hitpaw.py','sim_api_nodes/apis/hitpaw.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/download_helpers.py','sim_api_nodes/util/_helpers.py']
class Request:
    defaults={}
    def __init__(self,**fields): self.__dict__.update(copy.deepcopy(self.defaults));self.__dict__.update(fields)
    def model_dump(self,exclude_none=True):
        def encode(value):
            if isinstance(value,Request): return value.model_dump(exclude_none)
            if isinstance(value,list): return [encode(member) for member in value]
            return value
        return {key:encode(value) for key,value in vars(self).items() if not exclude_none or value is not None}
class Image:
    def __init__(self,slot,width,height,count=1): self.slot=slot;self.shape=(count,height,width,3)
class Video:
    def __init__(self,slot,width,height,duration=1): self.slot=slot;self.width=width;self.height=height;self.duration=duration
    def get_duration(self): return self.duration
    def get_dimensions(self): return self.width,self.height

def strip(node):
    for member in ast.walk(node):
        if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
            member.returns=None
            for argument in [*member.args.posonlyargs,*member.args.args,*member.args.kwonlyargs]: argument.annotation=None
    return ast.fix_missing_locations(node)
def input_json(value):
    if isinstance(value,Image): return {'native_image':value.slot,'width':value.shape[2],'height':value.shape[1],'batch_count':value.shape[0]}
    if isinstance(value,Video): return {'native_video':value.slot,'width':value.width,'height':value.height,'duration':value.duration}
    if isinstance(value,dict): return {key:input_json(member) for key,member in value.items()}
    return value
async def generate():
    requests=[];media=[];response_mode=None
    namespace={'__builtins__':__builtins__,'classmethod':classmethod,'math':math,'logging':logging,'IO':SimpleNamespace(NodeOutput=lambda *values:values),'ApiEndpoint':lambda path,method='GET':SimpleNamespace(path=path,method=method)}
    api=ast.parse((SOURCE_ROOT/SOURCES[1]).read_text())
    for node in api.body:
        if isinstance(node,ast.ClassDef) and any(ast.unparse(base)=='BaseModel' for base in node.bases):
            defaults={member.target.id:ast.literal_eval(member.value.args[0]) if member.value.args else None for member in node.body if isinstance(member,ast.AnnAssign)}
            defaults={key:None if value is Ellipsis else value for key,value in defaults.items()}
            namespace[node.name]=type(node.name,(Request,),{'defaults':defaults})
    async def sync(cls,endpoint,data,**options):
        requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(),'wait_label':options['wait_label'],'final_label_on_success':options['final_label_on_success']})
        coins=None if response_mode=='missing-coins' else 123
        response=namespace['TaskCreateResponse'](code=400 if response_mode=='creation-error' else 200,message='source creation failure' if response_mode=='creation-error' else 'success',data=namespace['TaskCreateDataResponse'](job_id='source-job-42',consume_coins=coins));requests[-1]['response']=response.model_dump();return response
    async def poll(cls,endpoint,data,**options):
        requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(),'poll_interval':options['poll_interval'],'price_usd':options['price_extractor'](None)})
        return namespace['TaskStatusResponse'](code=200,message='success',data=SimpleNamespace(job_id='source-job-42',status='completed',res_url='https://fixture.invalid/result'))
    def downscale(image,total_pixels): media.append({'operation':'downscale_image','slot':image.slot,'total_pixels':total_pixels});return Image(image.slot+'-downscaled',1,1)
    async def upload_image(cls,image,**options): media.append({'operation':'upload_image','slot':image.slot,**options});return 'https://storage.fixture.invalid/image'
    async def upload_video(cls,video): media.append({'operation':'upload_video','slot':video.slot,'container':'mp4','codec':'h264'});return 'https://storage.fixture.invalid/video'
    async def download(url,kind): return {'native_materializer':kind,'url':url}
    namespace.update(sync_op=sync,poll_op=poll,downscale_image_tensor=downscale,upload_image_to_simapi=upload_image,upload_video_to_simapi=upload_video,download_url_to_image_tensor=lambda url:download(url,'image'),download_url_to_video_output=lambda url:download(url,'video'))
    validation=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text());selected=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef) and node.name in {'get_image_dimensions','validate_video_duration'}];exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<source-validation>','exec'),namespace)
    nodes=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());selected=[];contracts={}
    for node in nodes.body:
        if isinstance(node,ast.Assign): selected.append(copy.deepcopy(node))
        elif isinstance(node,ast.ClassDef):
            methods=[copy.deepcopy(member) for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute']
            if not methods: continue
            selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=methods,decorator_list=[]));execute=methods[0]
            schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords}
            contracts[node.name]={'arguments':[argument.arg for argument in execute.args.args if argument.arg!='cls'],'defaults':{},'hidden':ast.unparse(fields['hidden']),'outputs':ast.unparse(fields['outputs'])}
    exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<source-node-execute>','exec'),namespace)
    cases=[];errors=[];response_errors=[]
    async def observe(node,arguments,error=False,mode=None):
        nonlocal response_mode
        requests.clear();media.clear();response_mode=mode
        try: result=await namespace[node].execute(**copy.deepcopy(arguments))
        except Exception as failure:
            if not error: raise
            observed={'input':{'node':node,'arguments':input_json(arguments)},'message':str(failure),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'response_mode':mode}
            (response_errors if mode else errors).append(observed)
        else:
            if error: raise ValueError('Expected source error')
            cases.append({'input':{'node':node,'arguments':input_json(arguments)},'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'outputs':list(result)})
    for model in ['generative_portrait','generative']:
        for factor in [1,2,4]: await observe('HitPawGeneralImageEnhance',dict(model=model,image=Image('small',1024,768,2),upscale_factor=factor,auto_downscale=False))
    for width,height,factor in [(2000,1500,4),(4000,4000,4),(12000,10000,4),(4000,4000,2),(10000,10000,1)]:
        await observe('HitPawGeneralImageEnhance',dict(model='generative',image=Image('large',width,height),upscale_factor=factor,auto_downscale=True))
    await observe('HitPawGeneralImageEnhance',dict(model='generative',image=Image('large',4000,4000),upscale_factor=4,auto_downscale=False),True)
    video=dict(model={'model':'General Restore Model (2x)','resolution':'original'},video=Video('video',1919,1081,3))
    for model in namespace['VIDEO_MODELS_MODELS_MAP']:
        for resolution in namespace['RESOLUTIONS_LIMITED'] if model in namespace['LIMITED_RESOLUTION_MODELS'] else namespace['RESOLUTIONS_FULL']:
            await observe('HitPawVideoEnhance',{**video,'model':{'model':model,'resolution':resolution},'video':Video('small-video',640,360,0.49995)})
    for width,height,resolution in [(720,720,'4K/UHD'),(720,720,'8K'),(360,640,'1080p'),(1919,1081,'2K/QHD')]:
        await observe('HitPawVideoEnhance',{**video,'model':{**video['model'],'resolution':resolution},'video':Video('dimensions',width,height,3600.00005)})
    for width,height,resolution in [(1080,1080,'720p'),(1920,1080,'720p')]:
        await observe('HitPawVideoEnhance',{**video,'model':{**video['model'],'resolution':resolution},'video':Video('too-large',width,height)},True)
    for duration in [0.4998,3600.0002]: await observe('HitPawVideoEnhance',{**video,'video':Video('duration',640,360,duration)},True)
    image=dict(model='generative',image=Image('source',100,100),upscale_factor=2,auto_downscale=False)
    for mode in ['creation-error','missing-coins']:
        await observe('HitPawGeneralImageEnhance',image,True,mode)
        await observe('HitPawVideoEnhance',video,True,mode)
    client=ast.parse((SOURCE_ROOT/SOURCES[2]).read_text());polling={ast.unparse(node.targets[0]):ast.literal_eval(node.value) for node in client.body if isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) in {'COMPLETED_STATUSES','FAILED_STATUSES','QUEUED_STATUSES'}}
    return {'schema_version':1,'namespace':'zed.sim.provider.hitpaw','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'scope':'Actual source request/preflight/model-resolution selection and task errors. Native Lanczos pixels/upload/materialization are explicit structural boundaries, not fabricated outputs.','contracts':contracts,'model_map':namespace['VIDEO_MODELS_MODELS_MAP'],'target_map':namespace['RESOLUTION_TARGET_MAP'],'square_map':namespace['RESOLUTION_SQUARE_MAP'],'limited_models':sorted(namespace['LIMITED_RESOLUTION_MODELS']),'cases':cases,'preflight_errors':errors,'response_errors':response_errors,'polling':polling}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');args=parser.parse_args();SOURCE_ROOT=args.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode()
    if args.check:
        if DESTINATION.read_bytes()!=encoded: raise SystemExit('HitPaw source contract differs')
    else: DESTINATION.write_bytes(encoded)
