"""Execute pinned Kling AST/API and expose native owners as structural probes."""
import argparse, ast, asyncio, base64, copy, hashlib, json, logging, math, re, unicodedata
from pathlib import Path
from types import SimpleNamespace
SOURCE_ROOT=Path(__file__).resolve().parents[5]/'projects/sim/ComfyUI'
FOLDER=Path(__file__).parent
SOURCES=['sim_api_nodes/nodes_kling.py','sim_api_nodes/apis/kling.py','sim_api_nodes/apis/__init__.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/download_helpers.py','sim_api_nodes/util/_helpers.py']
class TensorProbe:
    def __init__(self,slot,shape=(1,720,720,3)):self.slot=slot;self.shape=shape;self.ndim=len(shape)
    def __iter__(self):return iter([TensorProbe(self.slot+'-'+str(index),self.shape[1:]) for index in range(self.shape[0])])
class VideoProbe:
    def __init__(self,slot='video',width=720,height=720,duration=5):self.slot=slot;self.width=width;self.height=height;self.duration=duration
    def get_dimensions(self):return self.width,self.height
    def get_duration(self):return self.duration
def strip(node):
    for member in ast.walk(node):
        if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
            member.returns=None
            for argument in [*member.args.posonlyargs,*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
    return ast.fix_missing_locations(node)
def serial(value):
    if isinstance(value,TensorProbe):return {'native_slot':value.slot,'shape':list(value.shape)}
    if isinstance(value,VideoProbe):return {'native_video':value.slot,'width':value.width,'height':value.height,'duration':value.duration}
    if hasattr(value,'model_dump'):return value.model_dump(mode='json',exclude_none=True)
    if isinstance(value,dict):return {key:serial(member) for key,member in value.items()}
    if isinstance(value,(tuple,list)):return [serial(member) for member in value]
    return value
async def generate():
    requests=[];media=[];response_mode=None
    def concatenate(tensors,dim=0):
        if not tensors:raise RuntimeError('torch.cat(): expected a non-empty list of Tensors')
        return {'canonical_image_batch':serial(tensors),'dim':dim}
    namespace={'__builtins__':__builtins__,'classmethod':classmethod,'logging':logging,'math':math,'re':re,'torch':SimpleNamespace(Tensor=TensorProbe,cat=concatenate),'IO':SimpleNamespace(NodeOutput=lambda *values:values),'ApiEndpoint':lambda path,method='GET',query_params=None:SimpleNamespace(path=path,method=method,query_params=query_params)}
    tree=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());legacy=ast.parse((SOURCE_ROOT/SOURCES[2]).read_text());classes={node.name:node for node in legacy.body if isinstance(node,ast.ClassDef)}
    required={alias.name for node in tree.body if isinstance(node,ast.ImportFrom) and node.module=='sim_api_nodes.apis' for alias in node.names}
    while True:
        expanded=required|{member.id for name in required for member in ast.walk(classes[name]) if isinstance(member,ast.Name) and member.id in classes}
        if expanded==required:break
        required=expanded
    selection=[copy.deepcopy(node) for node in legacy.body if isinstance(node,(ast.Import,ast.ImportFrom)) or isinstance(node,ast.ClassDef) and node.name in required]
    exec(compile(ast.Module(body=selection,type_ignores=[]),'<source-legacy-api>','exec'),namespace)
    exec(compile(ast.parse((SOURCE_ROOT/SOURCES[1]).read_text()),'<source-modern-api>','exec'),namespace)
    def raw_response(model,status):
        if model.__name__=='Kling3TurboCreateResponse':return {'code':0,'data':{'id':'source-task-42'},'message':'success'}
        if model.__name__=='Kling3TurboQueryResponse':return {'code':0,'data':[{'id':'source-task-42','status':status,'outputs':[{'type':'image','url':'https://fixture.invalid/unused.png'},{'type':'video','url':'https://fixture.invalid/result.mp4'}]}]}
        return {'code':0,'message':'success','data':{'task_id':'source-task-42','task_status':status,'task_result':{'videos':[{'url':'https://fixture.invalid/result.mp4','id':'source-video-42','duration':'5'}],'images':[{'url':'https://fixture.invalid/first.png','index':0},{'url':'https://fixture.invalid/second.png','index':1}]}}}
    async def sync(cls,endpoint,data,response_model,**options):
        raw=raw_response(response_model,'submitted')
        if response_mode=='creation-error':raw['code']=400;raw['message']='source rejected'
        if response_mode=='missing-id':raw['data'].pop('id',None);raw['data'].pop('task_id',None)
        requests.append({'phase':'create','path':endpoint.path,'method':endpoint.method,'query_params':endpoint.query_params,'body':data.model_dump(mode='json',exclude_none=True),'response_model':response_model.__name__,'options':options,'response':copy.deepcopy(raw)})
        return response_model.model_validate(raw)
    async def poll(cls,endpoint,response_model,status_extractor,**options):
        raw=raw_response(response_model,'succeed')
        if response_mode=='missing-result':
            if isinstance(raw['data'],list):raw['data'][0]['outputs']=[]
            else:raw['data']['task_result']={'videos':[],'images':[]}
        response=response_model.model_validate(raw)
        requests.append({'phase':'poll','path':endpoint.path,'method':endpoint.method,'query_params':endpoint.query_params,'response_model':response_model.__name__,'options':options,'status':status_extractor(response),'response':copy.deepcopy(raw)})
        return response
    def inline(image,**options):media.append({'operation':'inline_image','image':serial(image),'mime_type':'image/png','total_pixels':4194304,**options});return base64.b64encode((FOLDER/'downloaded-image.png').read_bytes()).decode()
    async def images(cls,image,**options):media.append({'operation':'upload_images','image':serial(image),**options});return ['https://storage.fixture.invalid/'+image.slot+'/'+str(index) for index in range(namespace['get_number_of_images'](image))]
    async def image(cls,image,**options):media.append({'operation':'upload_image','image':serial(image),**options});return 'https://storage.fixture.invalid/'+image.slot
    async def video(cls,video,**options):media.append({'operation':'upload_video','video':serial(video),'container':'mp4','codec':'h264',**options});return 'https://storage.fixture.invalid/'+video.slot
    async def audio(cls,audio,**options):media.append({'operation':'upload_audio','audio':serial(audio),**options});return 'https://storage.fixture.invalid/audio'
    async def download(url,kind):return {'canonical_media_owner':kind,'url':url}
    namespace.update(sync_op=sync,poll_op=poll,tensor_to_base64_string=inline,upload_images_to_simapi=images,upload_image_to_simapi=image,upload_video_to_simapi=video,upload_audio_to_simapi=audio,download_url_to_image_tensor=lambda url:download(url,'RGBA'),download_url_to_video_output=lambda url:download(url,'VIDEO'))
    validation=ast.parse((SOURCE_ROOT/SOURCES[4]).read_text());exec(compile(strip(ast.Module(body=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef)],type_ignores=[])),'<source-validation>','exec'),namespace)
    selected=[];contracts={}
    for node in tree.body:
        if isinstance(node,ast.Assign):selected.append(copy.deepcopy(node))
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name not in {'_generate_storyboard_inputs','sim_entrypoint'}:selected.append(copy.deepcopy(node))
        elif isinstance(node,ast.ClassDef):
            method=next((copy.deepcopy(member) for member in node.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name=='execute'),None)
            if method is None:continue
            schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords};identifier=ast.literal_eval(fields['node_id'])
            defaults={argument.arg:ast.literal_eval(value) for argument,value in zip(method.args.args[-len(method.args.defaults):],method.args.defaults)} if method.args.defaults else {}
            contracts[identifier]={'symbol':node.name,'arguments':[argument.arg for argument in method.args.args if argument.arg!='cls'],'defaults':defaults,'hidden':ast.unparse(fields.get('hidden',ast.List(elts=[],ctx=ast.Load()))),'outputs':ast.unparse(fields['outputs'])}
            selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=[method],decorator_list=[]))
    exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<source-execute>','exec'),namespace)
    cases=[];errors=[];response_errors=[]
    async def observe(node,arguments,error=False,mode=None):
        nonlocal response_mode
        requests.clear();media.clear();response_mode=mode
        try:
            result=namespace[contracts[node]['symbol']].execute(**copy.deepcopy(arguments))
            if hasattr(result,'__await__'):result=await result
        except Exception as failure:
            if not error:raise
            record={'input':{'node':node,'arguments':serial(arguments)},'message':str(failure),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'response_mode':mode};(response_errors if mode else errors).append(record)
        else:
            if error:raise ValueError('Expected source failure')
            cases.append({'input':{'node':node,'arguments':serial(arguments)},'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'outputs':serial(result)})
    image=TensorProbe('image');end=TensorProbe('end');references=TensorProbe('references',(2,720,720,3));video=VideoProbe();audio={'sample_rate':8000,'waveform':TensorProbe('audio',(1,1,24000))};camera=namespace['KlingCameraControl'](type='down_back',config=namespace['KlingCameraConfig'](horizontal=1,vertical=0,pan=0,tilt=0,roll=0,zoom=0))
    baselines={
    'KlingCameraControls':dict(camera_control_type='simple',horizontal_movement=0,vertical_movement=0,pan=0,tilt=0,roll=0,zoom=0),
    'KlingTextToVideoNode':dict(prompt='source',negative_prompt='',cfg_scale=0.5,mode=next(iter(namespace['MODE_TEXT2VIDEO'])),aspect_ratio='16:9'),
    'KlingCameraControlT2VNode':dict(prompt='source',negative_prompt='',cfg_scale=0.5,aspect_ratio='16:9',camera_control=camera),
    'KlingImage2VideoNode':dict(start_frame=image,prompt='source',negative_prompt='',model_name='kling-v2-5-turbo',cfg_scale=0.5,mode='std',aspect_ratio='16:9',duration='5',camera_control=camera),
    'KlingCameraControlI2VNode':dict(start_frame=image,prompt='source',negative_prompt='',cfg_scale=0.5,aspect_ratio='16:9',camera_control=camera),
    'KlingStartEndFrameNode':dict(start_frame=image,end_frame=end,prompt='source',negative_prompt='',cfg_scale=0.5,aspect_ratio='16:9',mode=next(iter(namespace['MODE_START_END_FRAME']))),
    'KlingVideoExtendNode':dict(prompt='source',negative_prompt='',cfg_scale=0.5,video_id='source-video-42'),
    'KlingDualCharacterVideoEffectNode':dict(image_left=image,image_right=end,effect_scene='hug',model_name='kling-v1',mode='pro',duration='5'),
    'KlingSingleImageVideoEffectNode':dict(image=image,effect_scene='squish',model_name='kling-v1-6',duration='5'),
    'KlingLipSyncAudioToVideoNode':dict(video=video,audio=audio,voice_language='en'),
    'KlingLipSyncTextToVideoNode':dict(video=video,text='source',voice='Melody',voice_speed=1.0),
    'KlingVirtualTryOnNode':dict(human_image=image,cloth_image=end,model_name='kolors-virtual-try-on-v1'),
    'KlingImageGenerationNode':dict(model_name='kling-v3',prompt='source',negative_prompt='',image_type='subject',image_fidelity=0.5,human_fidelity=0.45,n=2,aspect_ratio='16:9',image=image),
    'KlingTextToVideoWithAudio':dict(model_name='kling-v2-6',prompt='source',mode='pro',aspect_ratio='16:9',duration=5,generate_audio=True),
    'KlingImageToVideoWithAudio':dict(model_name='kling-v2-6',start_frame=image,prompt='source',mode='pro',duration=5,generate_audio=True),
    'KlingMotionControl':dict(prompt='',reference_image=image,reference_video=video,keep_original_sound=True,character_orientation='image',mode='pro'),
    'KlingOmniProTextToVideoNode':dict(model_name='kling-video-o1',prompt='source',aspect_ratio='16:9',duration=5),
    'KlingOmniProFirstLastFrameNode':dict(model_name='kling-video-o1',prompt='source @image2',duration=5,first_frame=image,end_frame=end),
    'KlingOmniProImageToVideoNode':dict(model_name='kling-video-o1',prompt='source @image',aspect_ratio='16:9',duration=5,reference_images=references),
    'KlingOmniProVideoToVideoNode':dict(model_name='kling-video-o1',prompt='source @video',aspect_ratio='16:9',duration=5,reference_video=video,keep_original_sound=False,reference_images=references),
    'KlingOmniProEditVideoNode':dict(model_name='kling-video-o1',prompt='source @video',video=video,keep_original_sound=True,reference_images=references),
    'KlingOmniProImageNode':dict(model_name='kling-image-o1',prompt='source @image2',resolution='2K',aspect_ratio='16:9',reference_images=references),
    'KlingVideoNode':dict(multi_shot={'multi_shot':'disabled','prompt':'source','negative_prompt':'','duration':5},generate_audio=True,model={'model':'kling-v3','resolution':'1080p','aspect_ratio':'16:9'},seed=42),
    'KlingFirstLastFrameNode':dict(prompt='source',duration=5,first_frame=image,end_frame=end,generate_audio=False,model={'model':'kling-v3','resolution':'720p'},seed=42),
    'KlingAvatarNode':dict(image=image,sound_file=audio,mode='std',seed=42,prompt=''),}
    for node,arguments in baselines.items():await observe(node,arguments)
    for node in ['KlingOmniProFirstLastFrameNode','KlingOmniProVideoToVideoNode','KlingOmniProEditVideoNode','KlingOmniProImageNode']:
        arguments=copy.deepcopy(baselines[node]);arguments.pop('reference_images',None)
        if node=='KlingOmniProFirstLastFrameNode':arguments.pop('end_frame');arguments['reference_images']=references
        await observe(node,arguments)
    for node in ['KlingTextToVideoWithAudio','KlingImageToVideoWithAudio','KlingOmniProVideoToVideoNode','KlingOmniProEditVideoNode','KlingOmniProImageNode','KlingFirstLastFrameNode']:
        await observe(node,{**baselines[node],'prompt':' '},True)
    await observe('KlingVideoNode',{**baselines['KlingVideoNode'],'multi_shot':{'multi_shot':'disabled','prompt':' ','negative_prompt':'','duration':5}},True)
    for mode in namespace['MODE_TEXT2VIDEO']:await observe('KlingTextToVideoNode',{**baselines['KlingTextToVideoNode'],'mode':mode})
    for mode in namespace['MODE_START_END_FRAME']:await observe('KlingStartEndFrameNode',{**baselines['KlingStartEndFrameNode'],'mode':mode})
    for voice in namespace['VOICES_CONFIG']:await observe('KlingLipSyncTextToVideoNode',{**baselines['KlingLipSyncTextToVideoNode'],'voice':voice})
    for scene in namespace['KlingDualCharacterEffectsScene']:await observe('KlingDualCharacterVideoEffectNode',{**baselines['KlingDualCharacterVideoEffectNode'],'effect_scene':scene.value})
    for scene in namespace['KlingSingleImageEffectsScene']:await observe('KlingSingleImageVideoEffectNode',{**baselines['KlingSingleImageVideoEffectNode'],'effect_scene':scene.value})
    for node in ['KlingVideoNode','KlingOmniProTextToVideoNode','KlingOmniProFirstLastFrameNode','KlingOmniProImageToVideoNode']:
        arguments=copy.deepcopy(baselines[node]);selector='multi_shot' if node=='KlingVideoNode' else 'storyboards';arguments[selector]={selector:'2 shots','storyboard_1_prompt':'one','storyboard_1_duration':2,'storyboard_2_prompt':'two','storyboard_2_duration':3}
        if node!='KlingVideoNode':arguments['model_name']='kling-v3-omni'
        if node=='KlingOmniProFirstLastFrameNode':arguments.pop('end_frame')
        await observe(node,arguments)
    for frame in [None,image]:
        for resolution in ['720p','1080p']:
            arguments=copy.deepcopy(baselines['KlingVideoNode']);arguments['model']={'model':'kling-3.0-turbo','resolution':resolution,'aspect_ratio':'9:16'};arguments['start_frame']=frame;await observe('KlingVideoNode',arguments)
    for resolution in ['4k','1080p','720p']:
        for node in ['KlingVideoNode','KlingFirstLastFrameNode']:
            arguments=copy.deepcopy(baselines[node]);arguments['model']['resolution']=resolution
            if node=='KlingVideoNode':arguments['start_frame']=image
            await observe(node,arguments)
    await observe('KlingOmniProImageNode',{**baselines['KlingOmniProImageNode'],'model_name':'kling-v3-omni','series_amount':'4','resolution':'4K'})
    for node in ['KlingTextToVideoNode','KlingImage2VideoNode','KlingOmniProTextToVideoNode','KlingTextToVideoWithAudio','KlingImageGenerationNode']:await observe(node,{**baselines[node],'prompt':''},True)
    for node,override in [('KlingOmniProTextToVideoNode',{'duration':3}),('KlingOmniProTextToVideoNode',{'generate_audio':True}),('KlingOmniProTextToVideoNode',{'resolution':'4k'}),('KlingOmniProFirstLastFrameNode',{'reference_images':references}),('KlingOmniProImageToVideoNode',{'reference_images':TensorProbe('too-many',(8,720,720,3))}),('KlingOmniProImageNode',{'series_amount':'4'}),('KlingOmniProImageNode',{'resolution':'4K'}),('KlingImageGenerationNode',{'negative_prompt':'x'*201}),('KlingAvatarNode',{'sound_file':{'sample_rate':8000,'waveform':TensorProbe('short',(1,1,8000))}}),('KlingMotionControl',{'reference_video':VideoProbe(duration=30)})]:await observe(node,{**baselines[node],**override},True)
    for node in ['KlingTextToVideoNode','KlingVirtualTryOnNode']:await observe(node,baselines[node],True,'missing-id')
    for node in ['KlingTextToVideoNode','KlingOmniProTextToVideoNode','KlingVideoNode','KlingVirtualTryOnNode']:await observe(node,baselines[node],True,'missing-result')
    for node in ['KlingOmniProTextToVideoNode','KlingVideoNode','KlingAvatarNode']:await observe(node,baselines[node],True,'creation-error')
    missing=[]
    for symbol,constant,line in [('execute_video_effect','PATH_VIDEO_EFFECTS',570),('execute_lipsync','PATH_LIP_SYNC',632),('KlingVideoExtendNode.execute','PATH_VIDEO_EXTEND',2118),('KlingVirtualTryOnNode.execute','PATH_VIRTUAL_TRY_ON',2468),('KlingImageGenerationNode.execute','PATH_IMAGE_GENERATIONS',2606)]:
        call=next(node for node in ast.walk(tree) if isinstance(node,ast.Call) and ast.unparse(node.func)=='ApiEndpoint' and node.lineno==line)
        missing.append({'method':'GET','source_template':ast.unparse(next(keyword.value for keyword in call.keywords if keyword.arg=='path')),'path':namespace[constant]+'/{task_id}','symbol':symbol,'source_line':line,'source_path':SOURCES[0],'source_sha256':hashlib.sha256((SOURCE_ROOT/SOURCES[0]).read_bytes()).hexdigest(),'constant':constant,'constant_value':namespace[constant]})
    def unicode_ranges(pattern):
        ranges=[];start=None;end=None;compiled=re.compile(pattern)
        for code in range(0x110000):
            if compiled.fullmatch(chr(code)):
                if start is None:start=code
                end=code
            elif start is not None:ranges.append([start,end]);start=None
        if start is not None:ranges.append([start,end])
        return ranges
    client=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text());polling={ast.unparse(node.targets[0]):ast.literal_eval(node.value) for node in client.body if isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) in {'COMPLETED_STATUSES','FAILED_STATUSES','QUEUED_STATUSES'}}
    enums={name:[member.value for member in value] for name,value in namespace.items() if name.startswith('Kling') and isinstance(value,type) and hasattr(value,'__members__')}
    return {'schema_version':1,'namespace':'zed.sim.provider.kling','scope':'Actual pinned source request/helper/API/validation branches. Native image pixels/base64/upload/media allocation are explicit canonical owner probes.','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'contracts':contracts,'model_modes':namespace['MODE_TEXT2VIDEO'],'start_end_modes':namespace['MODE_START_END_FRAME'],'voices':namespace['VOICES_CONFIG'],'enums':enums,'cases':cases,'preflight_errors':errors,'response_errors':response_errors,'missing_catalog_routes':missing,'polling':polling,'normalized_prompts':[{'input':prompt,'output':namespace['normalize_omni_prompt_references'](prompt)} for prompt in ['', '@image @video2', 'mail@image.com @imageFoo @image3', 'é@image @image٢ @video१२', '@image_1 (@video) @image0', 'ⓐ@image @image⁴ @video４']], 'unicode_version':unicodedata.unidata_version,'unicode_word_ranges':unicode_ranges(r'\w'),'unicode_decimal_ranges':unicode_ranges(r'\d')}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');arguments=parser.parse_args();SOURCE_ROOT=arguments.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode();destination=FOLDER/'source-contract.json'
    if arguments.check:
        if destination.read_bytes()!=encoded:raise SystemExit('Kling source oracle differs')
    else:destination.write_bytes(encoded)
