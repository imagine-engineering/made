"""Execute pinned Grok request, preflight, pricing and polling AST at native boundaries."""
import argparse, ast, asyncio, copy, hashlib, json, logging
from pathlib import Path
from types import SimpleNamespace
SOURCE_ROOT = Path(__file__).resolve().parents[5] / 'projects/sim/ComfyUI'
DESTINATION = Path(__file__).with_name('source-contract.json')
SOURCES = ['sim_api_nodes/nodes_grok.py', 'sim_api_nodes/apis/grok.py', 'sim_api_nodes/util/client.py', 'sim_api_nodes/util/validation_utils.py', 'sim_api_nodes/util/upload_helpers.py', 'sim_api_nodes/util/conversions.py', 'sim_api_nodes/util/download_helpers.py', 'sim_api_nodes/util/_helpers.py']
class Request:
    defaults = {}
    def __init__(self, **fields): self.__dict__.update(copy.deepcopy(self.defaults)); self.__dict__.update(fields)
    def model_dump(self, exclude_none=True):
        def encode(value):
            if isinstance(value, Request): return value.model_dump(exclude_none)
            if isinstance(value, list): return [encode(member) for member in value]
            if isinstance(value, dict): return {key: encode(member) for key, member in value.items()}
            return value
        return {key: encode(value) for key, value in vars(self).items() if not exclude_none or value is not None}
class Image:
    def __init__(self, slot, count=1, batched=True): self.slot=slot; self.shape=(count,2,2,3) if batched else (2,2,3); self.ndim=len(self.shape)
    def __getitem__(self, index):
        if index >= self.shape[0]: raise IndexError(index)
        return Image(f'{self.slot}/{index}', batched=False)
    def __iter__(self): return (self[index] for index in range(self.shape[0]))
class Video:
    def __init__(self, slot, duration=3, size=1024): self.slot=slot; self.duration=duration; self.size=size
    def get_duration(self): return self.duration
    def get_stream_source(self): return self

def strip(node):
    for member in ast.walk(node):
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            member.returns=None
            for argument in [*member.args.posonlyargs,*member.args.args,*member.args.kwonlyargs]: argument.annotation=None
    return ast.fix_missing_locations(node)
def input_json(value):
    if isinstance(value,Image): return {'native_image':value.slot,'image_count':value.shape[0] if value.ndim==4 else 1,'batched':value.ndim==4}
    if isinstance(value,Video): return {'native_video':value.slot,'duration':value.duration,'source_size':value.size}
    if isinstance(value,Request): return value.model_dump()
    if isinstance(value,dict): return {key:input_json(member) for key,member in value.items()}
    if isinstance(value,list): return [input_json(member) for member in value]
    return value
async def generate():
    requests=[]; media=[]; downloads=[]; contracts={}
    namespace={'__builtins__':__builtins__, 'classmethod':classmethod, 'torch':SimpleNamespace(Tensor=Image,cat=lambda values:{'native_materializer':'image_batch','members':values}), 'IO':SimpleNamespace(NodeOutput=lambda *values:values), 'logging':logging, 'ApiEndpoint':lambda path,method='GET':SimpleNamespace(path=path,method=method)}
    api=ast.parse((SOURCE_ROOT/SOURCES[1]).read_text())
    for node in api.body:
        if isinstance(node,ast.ClassDef):
            defaults={member.target.id:ast.literal_eval(member.value.args[0] if member.value.args else next((keyword.value for keyword in member.value.keywords if keyword.arg=='default'),ast.Constant(None))) for member in node.body if isinstance(member,ast.AnnAssign)}
            defaults={key:None if value is Ellipsis else value for key,value in defaults.items()}
            namespace[node.name]=type(node.name,(Request,),{'defaults':defaults})
    def image_response(): return namespace['ImageGenerationResponse'](data=[namespace['ImageResponseObject'](url='https://fixture.invalid/image.png')],usage=SimpleNamespace(cost_in_usd_ticks=123456789))
    async def sync(cls,endpoint,data,**options):
        requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(),'price_extractor':getattr(options.get('price_extractor'),'__name__',None)})
        return image_response() if endpoint.path.startswith('/proxy/xai/v1/images/') else namespace['VideoGenerationResponse'](request_id='source-request-42')
    async def poll(cls,endpoint,**options):
        response=namespace['VideoStatusResponse'](status=None,video=namespace['VideoResponseObject'](url='https://fixture.invalid/video.mp4',duration=6),usage=SimpleNamespace(cost_in_usd_ticks=123456789))
        requests.append({'path':endpoint.path,'method':endpoint.method,'status':options['status_extractor'](response),'price_extractor':options['price_extractor'].__name__})
        return response
    def inline(payload): media.append({'operation':'inline_png','slot':payload.slot}); return 'canonical-png-boundary:'+payload.slot
    async def upload_images(cls,images,**options):
        flattened=[]
        for image in images: flattened.extend(list(image) if image.ndim==4 else [image])
        flattened=flattened[:options['max_images']]
        media.append({'operation':'upload_images','slots':[image.slot for image in flattened],**options})
        return [f'https://storage.fixture.invalid/{image.slot}' for image in flattened]
    async def upload_video(cls,video): media.append({'operation':'upload_video','slot':video.slot,'mime_type':'video/mp4','codec':'h264'}); return f'https://storage.fixture.invalid/{video.slot}.mp4'
    async def download_image(url): downloads.append({'kind':'image','url':url}); return {'native_materializer':'image','url':url}
    async def download_video(url): downloads.append({'kind':'video','url':url}); return {'native_materializer':'video','url':url}
    namespace.update(sync_op=sync,poll_op=poll,tensor_to_base64_string=inline,upload_images_to_simapi=upload_images,upload_video_to_simapi=upload_video,download_url_to_image_tensor=download_image,download_url_to_video_output=download_video,get_fs_object_size=lambda stream:stream.size)
    validation=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text())
    selected=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef) and node.name in {'validate_string','validate_video_duration','get_number_of_images'}]
    exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<source-validation>','exec'),namespace)
    nodes=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text()); selected=[]
    for node in nodes.body:
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='_GROK_VIDEO_MODEL_API_IDS': selected.append(copy.deepcopy(node))
        elif isinstance(node,ast.FunctionDef) and node.name in {'_extract_grok_price','_extract_grok_video_price'}: selected.append(copy.deepcopy(node))
        elif isinstance(node,ast.ClassDef):
            methods=[copy.deepcopy(member) for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute']
            if not methods: continue
            selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=methods,decorator_list=[])); execute=methods[0]
            defaults={argument.arg:ast.literal_eval(default) for argument,default in zip(execute.args.args[-len(execute.args.defaults):],execute.args.defaults)} if execute.args.defaults else {}
            contracts[node.name]={'arguments':[argument.arg for argument in execute.args.args if argument.arg!='cls'],'defaults':defaults}
            schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema'); fields={keyword.arg:keyword.value for keyword in schema.keywords}
            contracts[node.name].update(node_identifier=ast.literal_eval(fields['node_id']),hidden=ast.unparse(fields['hidden']),outputs=ast.unparse(fields['outputs']),deprecated=ast.literal_eval(fields.get('is_deprecated',ast.Constant(False))))
    exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<source-node-execute>','exec'),namespace)
    cases=[]; errors=[]
    async def observe(node,arguments,error=False):
        requests.clear();media.clear();downloads.clear()
        try: result=await namespace[node].execute(**copy.deepcopy(arguments))
        except Exception as failure:
            if not error: raise
            errors.append({'input':{'node':node,'arguments':input_json(arguments)},'message':str(failure),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media)})
        else:
            if error: raise ValueError('Expected source failure')
            cases.append({'input':{'node':node,'arguments':input_json(arguments)},'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'downloads':copy.deepcopy(downloads),'outputs':input_json(list(result))})
    generation=dict(model='grok-imagine-image-quality',prompt=' source prompt ',aspect_ratio='16:9',number_of_images=2,seed=2147483647)
    await observe('GrokImageNode',generation)
    edit=dict(model='grok-imagine-image',image=Image('legacy',3),prompt=' edit ',resolution='2K',number_of_images=2,seed=42,aspect_ratio='16:9')
    await observe('GrokImageEditNode',edit)
    v2=dict(prompt=' v2 ',model=dict(model='grok-imagine-image-quality',resolution='1K',number_of_images=1,images={'image_2':Image('second',2),'image_1':Image('first',batched=False)},aspect_ratio='4:3'),seed=42)
    await observe('GrokImageEditNodeV2',v2)
    await observe('GrokImageEditNodeV2',{**v2,'model':{**v2['model'],'model':'grok-imagine-image-pro','images':{'image_1':Image('pro')},'aspect_ratio':'auto'}})
    video=dict(model='grok-imagine-video',prompt=' video ',resolution='720p',aspect_ratio='auto',duration=6,seed=42)
    await observe('GrokVideoNode',video)
    await observe('GrokVideoNode',{**video,'model':'grok-imagine-video-1.5','resolution':'1080p','image':Image('first-frame')})
    video_edit=dict(model='grok-imagine-video',prompt=' change ',video=Video('edit',8.70005,50*1024*1024),seed=42)
    await observe('GrokVideoEditNode',video_edit)
    reference=dict(prompt=' reference ',model=dict(model='grok-imagine-video',resolution='480p',aspect_ratio='16:9',duration=10,reference_images={'reference_2':Image('second',6),'reference_1':Image('first',3)}),seed=42)
    await observe('GrokVideoReferenceNode',reference)
    extension=dict(prompt=' continue ',video=Video('extend',15.00005),model=dict(model='grok-imagine-video',duration=8),seed=2147483647)
    await observe('GrokVideoExtendNode',extension)
    await observe('GrokImageNode',{**generation,'prompt':' '},True)
    await observe('GrokImageEditNode',{**edit,'model':'grok-imagine-image-pro','image':Image('two',2)},True)
    await observe('GrokImageEditNode',{**edit,'image':Image('four',4)},True)
    await observe('GrokImageEditNode',{**edit,'image':Image('one')},True)
    await observe('GrokImageEditNodeV2',{**v2,'model':{**v2['model'],'images':{}}},True)
    await observe('GrokVideoNode',{**video,'model':'grok-imagine-video-1.5'},True)
    await observe('GrokVideoNode',{**video,'resolution':'1080p'},True)
    await observe('GrokVideoNode',{**video,'image':Image('two',2)},True)
    await observe('GrokVideoEditNode',{**video_edit,'video':Video('long',8.7002)},True)
    await observe('GrokVideoEditNode',{**video_edit,'video':Video('large',3,50*1024*1024+1)},True)
    await observe('GrokVideoExtendNode',{**extension,'video':Video('short',1.9998)},True)
    await observe('GrokVideoExtendNode',{**extension,'video':Video('large',3,50*1024*1024+1)},True)
    prices=[]
    for ticks in [None,0,123456789,10000000000]:
        response=SimpleNamespace(usage=None if ticks is None else SimpleNamespace(cost_in_usd_ticks=ticks)); prices.append({'cost_in_usd_ticks':ticks,'image_usd':namespace['_extract_grok_price'](response),'video_usd':namespace['_extract_grok_video_price'](response)})
    client=ast.parse((SOURCE_ROOT/SOURCES[2]).read_text()); polling={}
    for node in client.body:
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) in {'COMPLETED_STATUSES','FAILED_STATUSES','QUEUED_STATUSES'}: polling[ast.unparse(node.targets[0])]=ast.literal_eval(node.value)
        elif isinstance(node,ast.AsyncFunctionDef) and node.name in {'poll_op_raw','sync_op_raw'}:
            polling[node.name]={argument.arg:ast.literal_eval(default) for argument,default in zip(node.args.kwonlyargs,node.args.kw_defaults) if isinstance(default,ast.Constant)}
    return {'schema_version':1,'namespace':'zed.sim.provider.grok','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'scope':'Actual request/preflight/price AST and source polling defaults; native media encoding/upload/materialization are explicit structural boundaries only.','contracts':contracts,'cases':cases,'preflight_errors':errors,'price_cases':prices,'polling':polling}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');args=parser.parse_args();SOURCE_ROOT=args.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode()
    if args.check:
        if DESTINATION.read_bytes()!=encoded: raise SystemExit('Grok source contract differs')
    else: DESTINATION.write_bytes(encoded)
