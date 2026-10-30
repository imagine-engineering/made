"""Execute pinned Bytedance request AST with explicit canonical media probes."""
import argparse, ast, asyncio, base64, copy, hashlib, json, logging, math, re
from pathlib import Path
from types import SimpleNamespace
SOURCE_ROOT=Path(__file__).resolve().parents[5]/'projects/sim/ComfyUI'
DESTINATION=Path(__file__).with_name('source-contract.json')
SOURCES=['sim_api_nodes/nodes_bytedance.py','sim_api_nodes/nodes_bytedance_llm.py','sim_api_nodes/apis/bytedance.py','sim_api_nodes/apis/bytedance_llm.py','sim_api_nodes/util/client.py','sim_api_nodes/util/_helpers.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/download_helpers.py']
class Media:
    def __init__(self,name,kind='image',count=1,width=640,height=640,duration=2.5):
        self.name,self.kind,self.count,self.width,self.height,self.duration=name,kind,count,width,height,duration
        self.shape=(count,height,width,4)
    def __iter__(self):return iter([Media(self.name,self.kind,1,self.width,self.height,self.duration) for unused in range(self.count)])
    def get_duration(self):return self.duration
    def get_dimensions(self):return self.width,self.height
    def get_frame_rate(self):return 30
class Request:
    defaults={}
    def __init__(self,**fields):self.__dict__.update(copy.deepcopy(self.defaults));self.__dict__.update(fields)
    def model_dump(self,exclude_none=True):
        def encode(value):
            if isinstance(value,Request):return value.model_dump(exclude_none=exclude_none)
            if isinstance(value,list):return [encode(member) for member in value]
            if isinstance(value,dict):return {key:encode(member) for key,member in value.items() if not exclude_none or member is not None}
            return value
        return {key:encode(value) for key,value in vars(self).items() if not exclude_none or value is not None}
def field(value=None,default=None,default_factory=None,**unused):return default_factory() if default_factory else default if default is not None else value
def strip(node):
    for child in ast.walk(node):
        if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)):
            child.returns=None
            for argument in [*child.args.posonlyargs,*child.args.args,*child.args.kwonlyargs]:argument.annotation=None
    return ast.fix_missing_locations(node)
def json_input(value):
    if isinstance(value,Media):return {'media':value.name,'kind':value.kind,'count':value.count,'width':value.width,'height':value.height,'duration':value.duration,'fps':30}
    if isinstance(value,dict):return {key:json_input(member) for key,member in value.items()}
    if isinstance(value,list):return [json_input(member) for member in value]
    return value
async def generate():
    requests=[];uploads=[];polls=[];preparation=[]
    async def upload_many(cls,media,max_images=8,mime_type=None,wait_label='Uploading',**kwargs):
        media=media if isinstance(media,list) else [media]
        flattened=[member for batch in media for member in batch]
        result=[]
        for member in flattened[:max_images]:
            url='https://storage.fixture.invalid/'+member.name
            uploads.append({'slot':member.name,'kind':member.kind,'mime_type':mime_type,'wait_label':wait_label,'url':url});result.append(url)
        return result
    async def upload_one(cls,media,wait_label='Uploading',**kwargs):return (await upload_many(cls,media,max_images=1,wait_label=wait_label,**kwargs))[0]
    async def upload_video(cls,media,wait_label='Uploading',**kwargs):
        url='https://storage.fixture.invalid/'+media.name;uploads.append({'slot':media.name,'kind':'video','mime_type':'video/mp4','wait_label':wait_label,'url':url});return url
    async def virtual_image(cls,media,wait_label='Uploading image'):
        preparation.append({'slot':media.name,'owner':'canonical normalized image hash and virtual-library protocol','wait_label':wait_label});uploads.append({'slot':media.name,'kind':'image','mime_type':None,'wait_label':wait_label,'url':'asset://'+media.name});return 'asset://'+media.name
    async def virtual_video(cls,media,wait_label='Uploading video'):
        preparation.append({'slot':media.name,'owner':'canonical MP4/H264 encoding, hash and virtual-library protocol','wait_label':wait_label});uploads.append({'slot':media.name,'kind':'video','mime_type':'video/mp4','wait_label':wait_label,'url':'asset://'+media.name});return 'asset://'+media.name
    async def sync(cls,endpoint,data=None,response_model=None,**options):
        requests.append({'method':endpoint.method,'path':endpoint.path,'body':data.model_dump() if data else None})
        if '/assets/' in endpoint.path:return SimpleNamespace(status='Active',asset_type='Image',error=None)
        return SimpleNamespace(id='task-42',asset_id='asset-42',model='fixture',created=0,data=[{'url':'https://download.fixture.invalid/image.png'}],error=None,audio='canonical-encoded-audio-boundary',code=None,message=None,output=[SimpleNamespace(type='message',content=[SimpleNamespace(type='output_text',text='source text',refusal=None)])],usage=None)
    async def poll(cls,endpoint,**options):
        polls.append({'path':endpoint.path,'method':endpoint.method,'interval':options.get('poll_interval',5),'maximum_attempts':options.get('max_poll_attempts',480),'completed_statuses':options.get('completed_statuses'),'failed_statuses':options.get('failed_statuses'),'estimated_duration':options.get('estimated_duration')})
        return SimpleNamespace(content=SimpleNamespace(video_url='https://download.fixture.invalid/video.mp4'),group_id='resolved-group',status='Active',usage=None)
    async def download(url):return url
    namespace={'__builtins__':__builtins__,'__name__':'source_oracle','classmethod':classmethod,'Field':field,'BaseModel':Request,'math':math,'re':re,'logging':logging,
      'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'ApiEndpoint':lambda path,method='GET':SimpleNamespace(path=path,method=method),'sync_op':sync,'poll_op':poll,
      'upload_images_to_simapi':upload_many,'upload_image_to_simapi':upload_one,'upload_video_to_simapi':upload_video,'download_url_to_image_tensor':download,'download_url_to_video_output':download,
      'get_number_of_images':lambda media:media.count,'validate_image_aspect_ratio':lambda *args,**kwargs:None,'validate_image_dimensions':lambda *args,**kwargs:None,'validate_video_dimensions':lambda *args,**kwargs:None,
      'image_tensor_pair_to_batch':lambda first,last:[first,last],'PromptServer':SimpleNamespace(instance=SimpleNamespace(send_progress_text=lambda *args:None)),
      'torch':SimpleNamespace(cat=lambda images:images),'base64':SimpleNamespace(b64decode=lambda value:value),'audio_bytes_to_audio_input':lambda value:'canonical audio decode boundary',
      'downscale_image_tensor_by_max_side':lambda image,**kwargs:image,'upscale_image_tensor_to_min_pixels':lambda image,*args:image,'tensor_to_base64_string':lambda image,**kwargs:'canonical-image-base64-boundary'}
    # Request fields and table constants are taken from the actual pinned API AST.
    for filename in SOURCES[2:4]:
        for node in ast.parse((SOURCE_ROOT/filename).read_text()).body:
            if isinstance(node,ast.ClassDef):
                defaults={}
                for member in node.body:
                    if isinstance(member,ast.AnnAssign) and isinstance(member.target,ast.Name):defaults[member.target.id]=eval(compile(ast.Expression(member.value),'<source-field>','eval'),namespace)
                namespace[node.name]=type(node.name,(Request,),{'defaults':defaults})
            elif isinstance(node,(ast.Assign,ast.AnnAssign)):
                exec(compile(strip(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),'<source-table>','exec'),namespace)
            elif isinstance(node,ast.FunctionDef) and node.name=='seedance2_price_per_1k_tokens':exec(compile(strip(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),'<source-price>','exec'),namespace)
    validation=ast.parse((SOURCE_ROOT/SOURCES[6]).read_text())
    functions=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef) and node.name in ['validate_string','validate_video_duration']]
    exec(compile(strip(ast.Module(body=functions,type_ignores=[])),'<source-validation>','exec'),namespace)
    for filename in SOURCES[:2]:
        tree=ast.parse((SOURCE_ROOT/filename).read_text());selected=[]
        for node in tree.body:
            if isinstance(node,(ast.Assign,ast.AnnAssign)):
                name=ast.unparse(node.targets[0] if isinstance(node,ast.Assign) else node.target)
                if name.startswith('PRICE_BADGE'):continue
                selected.append(copy.deepcopy(node))
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and not node.name.startswith('_seedream_model_inputs') and not node.name.startswith('_seedance2_') and node.name not in ['sim_entrypoint','_seed_model_inputs']:
                selected.append(copy.deepcopy(node))
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='_seedance2_price_extractor':selected.append(copy.deepcopy(node))
            elif isinstance(node,ast.ClassDef):
                members=[copy.deepcopy(member) for member in node.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name=='execute']
                if members:selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=members,decorator_list=[]))
        exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<source-node>','exec'),namespace)
    for value in namespace.values():
        if isinstance(value,type) and value.__name__.startswith('ByteDance'):value.hidden=SimpleNamespace(unique_id='oracle')
    namespace['_resize_to_exact']=lambda image,width,height:Media(image.name,image.kind,image.count,width,height,image.duration)
    namespace['_seedance2_target_dims']=lambda resolution,ratio,image:(1280,720)
    namespace['_seedance_virtual_library_upload_image_asset']=virtual_image;namespace['_seedance_virtual_library_upload_video_asset']=virtual_video
    cases=[];errors=[]
    async def observe(node,arguments,error=False):
        requests.clear();uploads.clear();polls.clear();preparation.clear();original=json_input(arguments)
        try: result=await namespace[node].execute(**copy.deepcopy(arguments))
        except Exception as failure:
            if not error:raise
            errors.append({'input':{'node':node,'arguments':original},'message':str(failure),'requests':copy.deepcopy(requests),'uploads':copy.deepcopy(uploads)})
        else:
            if error:raise ValueError('Expected pinned source failure')
            cases.append({'input':{'node':node,'arguments':original},'requests':copy.deepcopy(requests),'uploads':copy.deepcopy(uploads),'polls':copy.deepcopy(polls),'preparation_boundaries':copy.deepcopy(preparation),'outputs':json_input(list(result))})
    image=Media('image');first=Media('first_frame');last=Media('last_frame');video=Media('video','video')
    await observe('ByteDanceImageNode',dict(model='seedream-3-0-t2i-250415',prompt='  blue sky ☁  ',size_preset='Custom',width=1024,height=768,seed=42,guidance_scale=4.5,watermark=False))
    for model in ['seedream-4-0-250828','seedream 5.0 lite']:
        await observe('ByteDanceSeedreamNode',dict(model=model,prompt='  blue sky ☁  ',image=image,size_preset='Custom',width=2048,height=2048,sequential_image_generation='auto',max_images=2,seed=42,watermark=False,fail_on_partial=True))
    for model in ['seedream 5.0 pro','seedream-4-5-251128']:
        await observe('ByteDanceSeedreamNodeV2',dict(prompt='  blue sky ☁  ',model={'model':model,'size_preset':'Custom','width':2048,'height':2048,'max_images':1,'images':{'image_1':image},'fail_on_partial':False},seed=42,watermark=True))
    legacy=dict(model='seedance-1-5-pro-251215',prompt='  travel ☁  ',resolution='720p',aspect_ratio='16:9',duration=5,seed=42,camera_fixed=True,watermark=False,generate_audio=True)
    await observe('ByteDanceTextToVideoNode',legacy)
    await observe('ByteDanceImageToVideoNode',{**legacy,'image':image})
    await observe('ByteDanceFirstLastFrameNode',{**legacy,'first_frame':first,'last_frame':last})
    await observe('ByteDanceImageReferenceNode',{key:value for key,value in {**legacy,'images':image}.items() if key not in ['camera_fixed','generate_audio']})
    modern={'model':'Seedance 2.0','prompt':'Asset1 glides ☁','resolution':'720p','ratio':'16:9','duration':7,'generate_audio':True}
    await observe('ByteDance2TextToVideoNode',dict(model=modern,seed=42,watermark=False))
    await observe('ByteDance2FirstLastFrameNode',dict(model=modern,seed=42,watermark=False,first_frame=first,last_frame=last,first_frame_asset_id='',last_frame_asset_id=''))
    await observe('ByteDance2FirstLastFrameNode',dict(model=modern,seed=42,watermark=False,first_frame=None,last_frame=None,first_frame_asset_id='  existing-image  ',last_frame_asset_id=''))
    await observe('ByteDance2ReferenceNode',dict(model={**modern,'reference_images':{'image_1':image},'reference_videos':{},'reference_audios':{},'reference_assets':{},'auto_downscale':True,'auto_upscale':False},seed=42,watermark=False))
    await observe('ByteDance2ReferenceNode',dict(model={**modern,'reference_images':{},'reference_videos':{},'reference_audios':{},'reference_assets':{'asset_1':'  existing-image  '},'auto_downscale':True,'auto_upscale':False},seed=42,watermark=False))
    await observe('ByteDanceCreateImageAsset',dict(image=image,group_id=' existing-group '))
    await observe('ByteDanceCreateVideoAsset',dict(video=video,group_id=' existing-group '))
    for mode in ['text only','preset voice','image reference']:
        options={'reference_mode':mode}
        if mode=='preset voice':options['preset_voice']='Vivi (Female, multilingual)'
        if mode=='image reference':options['reference_image']=image
        await observe('ByteDanceSeedAudioNode',dict(text_prompt='Hello world',reference_mode=options,sample_rate='24000',speech_rate=0,loudness_rate=0,pitch_rate=0,seed=42))
    await observe('ByteDanceSeedNode',dict(prompt='  explain ☁  ',model={'model':'Seed 2.0 Pro','temperature':0.75,'images':{'image_1':image},'videos':{}},seed=42,system_prompt='Guide kindly'))
    await observe('ByteDanceImageNode',dict(model='seedream-3-0-t2i-250415',prompt=' ',size_preset='Custom',width=1024,height=768,seed=42,guidance_scale=4.5,watermark=False),True)
    await observe('ByteDanceTextToVideoNode',{**legacy,'prompt':'bad --duration 4'},True)
    await observe('ByteDance2FirstLastFrameNode',dict(model=modern,seed=42,watermark=False,first_frame=None,last_frame=None,first_frame_asset_id='',last_frame_asset_id=''),True)
    await observe('ByteDance2ReferenceNode',dict(model=modern,seed=42,watermark=False),True)
    await observe('ByteDanceSeedAudioNode',dict(text_prompt='@Audio1 hello',reference_mode={'reference_mode':'text only'},sample_rate='24000',speech_rate=0,loudness_rate=0,pitch_rate=0,seed=42),True)
    await observe('ByteDanceSeedNode',dict(prompt=' ',model={'model':'Seed 2.0 Pro','temperature':1.0},seed=42,system_prompt=''),True)
    constants={key:namespace[key] for key in ['SEEDREAM_MODELS','SEEDREAM_PRESETS','SEEDANCE_MODELS','SEED_MODELS','RECOMMENDED_PRESETS','RECOMMENDED_PRESETS_SEEDREAM_4','VIDEO_TASKS_EXECUTION_TIME','SEED_AUDIO_VOICE_MAP','SEEDANCE2_REF_VIDEO_PIXEL_LIMITS']}
    argument_names={}
    for filename in SOURCES[:2]:
        for node in ast.parse((SOURCE_ROOT/filename).read_text()).body:
            if isinstance(node,ast.ClassDef):
                for member in node.body:
                    if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute':argument_names[node.name]=[argument.arg for argument in member.args.args if argument.arg!='cls']
    constants['argument_names']=argument_names
    constants['seedance2_prices']=[{'model':key[0],'has_video':key[1],'resolution':key[2],'usd_per_1k':value} for key,value in namespace['SEEDANCE2_PRICE_PER_1K_TOKENS'].items()]
    return {'schema_version':1,'namespace':'zed.sim.provider.bytedance','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'scope':'Actual pinned request/node/helper AST; canonical image/audio/video transformations, encoding, normalized media hashing, virtual library and materialization are explicit media probe boundaries. No live requests or decoded native media claimed.','constants':constants,'cases':cases,'preflight_errors':errors}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');args=parser.parse_args();SOURCE_ROOT=args.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode()
    if args.check:
        if DESTINATION.read_bytes()!=encoded:raise SystemExit('Bytedance source contract differs')
    else:DESTINATION.write_bytes(encoded)
