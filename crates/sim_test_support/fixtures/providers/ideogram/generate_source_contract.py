"""Execute pinned Ideogram AST; native tensor operations remain explicit owner probes."""
import argparse, ast, asyncio, contextlib, copy, hashlib, io, json
from pathlib import Path
from types import SimpleNamespace
SOURCE_ROOT=Path(__file__).resolve().parents[5]/'projects/sim/ComfyUI'
DESTINATION=Path(__file__).with_name('source-contract.json')
SOURCES=['sim_api_nodes/nodes_ideogram.py','sim_api_nodes/apis/ideogram.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/download_helpers.py','sim_api_nodes/util/_helpers.py']
class NativeProbe:
    def __init__(self,slot,shape): self.slot=slot;self.shape=shape
    def squeeze(self): return self
    def cpu(self): return self
    def numpy(self): return self
    def __mul__(self,factor): return self
    def astype(self,dtype): return self
    def __rsub__(self,value): return NativeProbe(self.slot+'-inverted',self.shape)
def strip(node):
    for member in ast.walk(node):
        if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
            member.returns=None
            for argument in [*member.args.posonlyargs,*member.args.args,*member.args.kwonlyargs]: argument.annotation=None
    return ast.fix_missing_locations(node)
def encode_input(value):
    if isinstance(value,NativeProbe): return {'native_slot':value.slot,'shape':value.shape}
    if isinstance(value,dict): return {key:encode_input(member) for key,member in value.items()}
    return value
async def generate():
    requests=[];native=[];downloads=[];response_data=[SimpleNamespace(url='https://fixture.invalid/first.png'),SimpleNamespace(url=None),SimpleNamespace(url='https://fixture.invalid/second.png')]
    class EncodingProbe:
        def __init__(self,array): self.array=array
        def save(self,buffer,format):
            native.append({'operation':'png_encode','slot':self.array.slot,'format':format})
            # This marker is only a source call probe, never a media fixture or decoded tensor.
            buffer.write(('structural-owner:'+self.array.slot).encode())
    namespace={'__builtins__':__builtins__,'classmethod':classmethod,'BytesIO':io.BytesIO,'Image':SimpleNamespace(fromarray=EncodingProbe),'np':SimpleNamespace(uint8='uint8'),'IO':SimpleNamespace(NodeOutput=lambda *values:values),'torch':SimpleNamespace(cat=lambda values,dim:{'canonical_rgb_batch':values,'dim':dim}),'ApiEndpoint':lambda path,method='GET':SimpleNamespace(path=path,method=method)}
    api=ast.parse((SOURCE_ROOT/SOURCES[1]).read_text())
    exec(compile(api,'<pinned-source-api>','exec'),namespace)
    client=ast.parse((SOURCE_ROOT/SOURCES[2]).read_text())
    multipart=next(copy.deepcopy(node) for node in ast.walk(client) if isinstance(node,ast.If) and ast.unparse(node.test)=="cfg.content_type == 'multipart/form-data' and method != 'GET'")
    multipart.orelse=[]
    multipart_code=compile(ast.fix_missing_locations(ast.Module(body=[multipart],type_ignores=[])),'<source-client-multipart>','exec')
    class FormProbe:
        def __init__(self,default_to_multipart): self.fields=[];assert default_to_multipart
        def add_field(self,name,value,filename=None,content_type=None):
            self.fields.append({'name':name,'value':value if isinstance(value,str) else None,'file_name':filename,'media_type':content_type})
    async def sync(cls,endpoint,data,**options):
        files=options.pop('files',None) or {}
        requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(exclude_none=True),'form_order':list(data.model_dump(exclude_none=True)),'files':[{'name':name,'file_name':file.name,'native_slot':file.getvalue().decode().removeprefix('structural-owner:')} for name,file in files.items()],**{key:value for key,value in options.items() if key!='response_model'}})
        if options.get('content_type')=='multipart/form-data':
            state={'cfg':SimpleNamespace(content_type='multipart/form-data',multipart_parser=None,data=data.model_dump(exclude_none=True),files=files),'method':endpoint.method,'payload_headers':{},'payload_kw':{},'aiohttp':SimpleNamespace(FormData=FormProbe),'BytesIO':io.BytesIO,'contextlib':contextlib}
            exec(multipart_code,state)
            requests[-1]['wire_fields']=state['payload_kw']['data'].fields
        return SimpleNamespace(data=response_data)
    def resize(mask,image,allow_gradient):
        native.append({'operation':'resize_mask','mask':mask.slot,'image':image.slot,'upscale_method':'nearest-exact','crop':'disabled','allow_gradient':allow_gradient,'threshold':'>0.5','invert_after':True})
        return NativeProbe(mask.slot+'-resized',(image.shape[0],image.shape[1],image.shape[2]))
    async def download(url): downloads.append({'url':url});return SimpleNamespace(url=url)
    def decode(image,mode): downloads[-1]['mode']=mode;return {'canonical_materializer':mode,'url':image.url}
    namespace.update(sync_op=sync,resize_mask_to_image=resize,download_url_as_bytesio=download,bytesio_to_image_tensor=decode)
    validation=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text());exec(compile(strip(ast.Module(body=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef) and node.name=='validate_string'],type_ignores=[])),'<source-validation>','exec'),namespace)
    tree=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());selected=[];contracts={}
    for node in tree.body:
        if isinstance(node,ast.Assign) or isinstance(node,ast.AsyncFunctionDef) and node.name=='download_and_process_images': selected.append(copy.deepcopy(node))
        elif isinstance(node,ast.ClassDef) and node.name in {'IdeogramV3','IdeogramV4'}:
            method=next(copy.deepcopy(member) for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute')
            selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=[method],decorator_list=[]))
            schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords}
            defaults={argument.arg:ast.literal_eval(value) for argument,value in zip(method.args.args[-len(method.args.defaults):],method.args.defaults)} if method.args.defaults else {}
            contracts[node.name]={'arguments':[argument.arg for argument in method.args.args if argument.arg!='cls'],'defaults':defaults,'hidden':ast.unparse(fields['hidden']),'outputs':ast.unparse(fields['outputs'])}
    exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<source-node-execute>','exec'),namespace)
    cases=[];errors=[];response_errors=[]
    async def observe(node,arguments,error=False,data=None):
        nonlocal response_data
        requests.clear();native.clear();downloads.clear();response_data=data if data is not None else [SimpleNamespace(url='https://fixture.invalid/first.png'),SimpleNamespace(url=None),SimpleNamespace(url='https://fixture.invalid/second.png')]
        try: result=await namespace[node].execute(**copy.deepcopy(arguments))
        except Exception as failure:
            if not error: raise
            observed={'input':{'node':node,'arguments':encode_input(arguments)},'message':str(failure),'requests':copy.deepcopy(requests),'native':copy.deepcopy(native)}
            (response_errors if data is not None else errors).append(observed)
        else:
            if error: raise ValueError('Expected source error')
            cases.append({'input':{'node':node,'arguments':encode_input(arguments)},'requests':copy.deepcopy(requests),'native':copy.deepcopy(native),'downloads':copy.deepcopy(downloads),'outputs':list(result)})
    await observe('IdeogramV3',{'prompt':''})
    for ratio in namespace['V3_RATIO_MAP']: await observe('IdeogramV3',{'prompt':'source prompt','aspect_ratio':ratio})
    await observe('IdeogramV3',{'prompt':'source prompt','aspect_ratio':'unknown'})
    for resolution in namespace['V3_RESOLUTIONS'][1:]: await observe('IdeogramV3',{'prompt':'source prompt','resolution':resolution,'aspect_ratio':'3:1'})
    for speed in ['DEFAULT','TURBO','QUALITY','BALANCED']:
        for magic in ['AUTO','ON','OFF']: await observe('IdeogramV3',{'prompt':'source prompt','rendering_speed':speed,'magic_prompt_option':magic,'seed':2147483647,'num_images':8})
    image=NativeProbe('image',(1,2,3,3));mask=NativeProbe('mask',(1,1,1));character=NativeProbe('character',(1,4,5,3));character_mask=NativeProbe('character-mask',(1,2,2))
    for editing in [False,True]:
        for reference in [False,True]:
            for refmask in [False,True] if reference else [False]:
                arguments={'prompt':'source prompt','resolution':'512x1536','aspect_ratio':'9:16','magic_prompt_option':'ON','seed':42,'num_images':3}
                if editing: arguments.update(image=image,mask=mask)
                if reference: arguments.update(character_image=character)
                if refmask: arguments.update(character_mask=character_mask)
                await observe('IdeogramV3',arguments)
    for arguments in [{'prompt':'source','image':image},{'prompt':'source','mask':mask},{'prompt':'source','character_mask':character_mask},{'prompt':'source','image':image,'character_image':character,'character_mask':character_mask}]: await observe('IdeogramV3',arguments,True)
    resolutions=next(ast.literal_eval(keyword.value) for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='IdeogramV4' for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Combo.Input' and call.args and ast.literal_eval(call.args[0])=='resolution' for keyword in call.keywords if keyword.arg=='options')
    for resolution in resolutions:
        for speed in ['DEFAULT','TURBO','QUALITY']: await observe('IdeogramV4',dict(prompt='  source prompt  ',resolution=resolution,rendering_speed=speed,seed=2147483647))
    for prompt in ['', '   ']: await observe('IdeogramV4',dict(prompt=prompt,resolution='Auto',rendering_speed='DEFAULT',seed=0),True)
    for node,arguments in [('IdeogramV3',{'prompt':'source'}),('IdeogramV4',dict(prompt='source',resolution='Auto',rendering_speed='DEFAULT',seed=0))]:
        for data in [[],[SimpleNamespace(url=None)],[SimpleNamespace(url='')]]: await observe(node,arguments,True,data)
    schema_cases=[]
    for created in ['2026-10-01T00:00:00Z','not-a-date',42,True,None]:
        payload={'created':created,'data':[{'url':'https://fixture.invalid/image','seed':'42'}]}
        try: parsed=namespace['IdeogramGenerateResponse'].model_validate(payload)
        except Exception: schema_cases.append({'payload':payload,'accepted':False})
        else: schema_cases.append({'payload':payload,'accepted':True,'parsed':parsed.model_dump(mode='json',exclude_none=True)})
    resize_source=next(ast.unparse(node) for node in ast.parse((SOURCE_ROOT/SOURCES[4]).read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='resize_mask_to_image')
    return {'schema_version':1,'namespace':'zed.sim.provider.ideogram','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'scope':'Actual source request, file ordering, defaults, hidden auth and failures. Native tensor/pixel preparation and RGB materialization are structural owner probes, never fabricated outputs.','contracts':contracts,'ratio_map':namespace['V3_RATIO_MAP'],'v3_resolutions':namespace['V3_RESOLUTIONS'],'v4_resolutions':resolutions,'resize_owner_source':resize_source,'cases':cases,'preflight_errors':errors,'response_errors':response_errors,'api_schema_probe':schema_cases}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');args=parser.parse_args();SOURCE_ROOT=args.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode()
    if args.check:
        if DESTINATION.read_bytes()!=encoded: raise SystemExit('Ideogram source contract differs')
    else: DESTINATION.write_bytes(encoded)
