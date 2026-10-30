"""Execute pinned Gemini request, response filtering and price AST at native boundaries."""
import argparse,ast,asyncio,base64,copy,hashlib,json
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace
SOURCE_ROOT=Path(__file__).resolve().parents[5]/'projects/sim/ComfyUI'
DESTINATION=Path(__file__).with_name('source-contract.json')
SOURCES=['sim_api_nodes/nodes_gemini.py','sim_api_nodes/apis/gemini.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/_helpers.py','sim_api_nodes/util/download_helpers.py']
class Request:
    defaults={}
    def __init__(self,**fields):self.__dict__.update(copy.deepcopy(self.defaults));self.__dict__.update(fields)
    def model_dump(self,exclude_none=True):
        def encode(value):
            if isinstance(value,Request):return value.model_dump(exclude_none)
            if isinstance(value,Enum):return value.value
            if isinstance(value,list):return [encode(member)for member in value]
            if isinstance(value,dict):return {key:encode(member)for key,member in value.items()}
            return value
        return {key:encode(value) for key,value in vars(self).items() if not exclude_none or value is not None}
class Image:
    def __init__(self,slot,count=1,batched=True):self.slot=slot;self.shape=(count,2,2,3) if batched else(2,2,3)
    def __getitem__(self,index):return Image(f'{self.slot}/{index}',batched=False)
class Waveform:
    def __init__(self,slot,count=1):self.slot=slot;self.shape=(count,1,8)
    def __getitem__(self,index):return Waveform(f'{self.slot}/{index}')
    def unsqueeze(self,axis):return self
class Video:
    def __init__(self,slot,duration=1):self.slot=slot;self.duration=duration

def field(value=None,default=None,default_factory=None,**unused):return default_factory() if default_factory else(default if default is not None else value)
def strip(node):
    for member in ast.walk(node):
        if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
            member.returns=None
            for argument in[*member.args.posonlyargs,*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
    return ast.fix_missing_locations(node)
def input_json(value):
    if isinstance(value,Image):return {'native_image':value.slot,'image_count':value.shape[0]if len(value.shape)==4 else 1}
    if isinstance(value,Waveform):return {'native_audio':value.slot,'batch_count':value.shape[0]}
    if isinstance(value,Video):return {'native_video':value.slot,'duration':value.duration}
    if isinstance(value,Request):return value.model_dump()
    if isinstance(value,Enum):return value.value
    if isinstance(value,dict):return {key:input_json(member)for key,member in value.items()}
    if isinstance(value,list):return [input_json(member)for member in value]
    return value
async def generate():
    requests=[];media=[]
    namespace={'__builtins__':__builtins__,'classmethod':classmethod,'BaseModel':Request,'Field':field,'Enum':Enum,'fnmatch':fnmatch,'base64':base64,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'Input':SimpleNamespace(Audio=lambda **fields:fields),'Types':SimpleNamespace(VideoContainer=SimpleNamespace(MP4='mp4'),VideoCodec=SimpleNamespace(H264='h264')),'ApiEndpoint':lambda path,method='GET':SimpleNamespace(path=path,method=method),'folder_paths':SimpleNamespace(get_annotated_filepath=lambda file:str(DESTINATION.parent/file)),'open':open,'torch':SimpleNamespace(Tensor=object)}
    api=ast.parse((SOURCE_ROOT/SOURCES[1]).read_text())
    for node in api.body:
        if not isinstance(node,ast.ClassDef):continue
        if any(ast.unparse(base)=='Enum' for base in node.bases):exec(compile(strip(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),'<source-enum>','exec'),namespace);continue
        defaults={}
        for base in node.bases:
            parent=namespace.get(ast.unparse(base));defaults.update(copy.deepcopy(getattr(parent,'defaults',{})))
        for member in node.body:
            if isinstance(member,ast.AnnAssign):defaults[member.target.id]=eval(compile(ast.fix_missing_locations(ast.Expression(member.value if member.value is not None else ast.Constant(None))),'<source-api-default>','eval'),namespace)
        namespace[node.name]=type(node.name,(Request,),{'defaults':defaults})
    def response(model='gemini-3.1-flash-image-preview'):
        part=namespace['GeminiPart'];content=namespace['GeminiContent'];candidate=namespace['GeminiCandidate']
        return namespace['GeminiGenerateContentResponse'](modelVersion=model,usageMetadata=namespace['GeminiUsageMetadata'](promptTokenCount=1024,thoughtsTokenCount=3,candidatesTokensDetails=[namespace['ModalityTokenCount'](modality=namespace['Modality'].TEXT,tokenCount=17),namespace['ModalityTokenCount'](modality=namespace['Modality'].IMAGE,tokenCount=31),namespace['ModalityTokenCount'](modality=namespace['Modality'].VIDEO,tokenCount=19)]),candidates=[candidate(content=content(role=namespace['GeminiRole'].model,parts=[part(text='first'),part(text='second')]))])
    async def sync(cls,endpoint,data,**options):requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(),'price_extractor':options['price_extractor'].__name__});return response()
    async def upload_images(cls,images,max_images=0,**options):
        count=sum(value.shape[0]if len(value.shape)==4 else 1 for value in images);count=min(count,max_images) if max_images else count
        media.append({'operation':'upload_images','count':count,**options});return [f'https://storage.fixture.invalid/image/{index}'for index in range(count)]
    async def upload_image(cls,payload,**options):media.append({'operation':'upload_image','slot':payload.slot,**options});return 'https://storage.fixture.invalid/'+payload.slot
    async def upload_audio(cls,payload,**options):media.append({'operation':'upload_audio','slot':payload['waveform'].slot,**options});return 'https://storage.fixture.invalid/'+payload['waveform'].slot
    async def upload_video(cls,payload,**options):media.append({'operation':'upload_video','slot':payload.slot,**options});return 'https://storage.fixture.invalid/'+payload.slot
    def inline(kind,payload,**options):
        slot=payload['waveform'].slot if isinstance(payload,dict)else payload.slot;media.append({'operation':'inline_'+kind,'slot':slot,**options});return 'canonical-'+kind+'-encoding:'+slot
    async def image_output(response,thought=False):return {'native_materializer':'image','thought':thought}
    async def video_output(response,cls=None):return {'native_materializer':'video'}
    def duration(video,max_duration):
        if video.duration>max_duration:raise ValueError(f'Video duration exceeds {max_duration} seconds')
    namespace.update(sync_op=sync,upload_images_to_simapi=upload_images,upload_image_to_simapi=upload_image,upload_audio_to_simapi=upload_audio,upload_video_to_simapi=upload_video,tensor_to_base64_string=lambda payload,**options:inline('image',payload,**options),audio_to_base64_string=lambda payload,**options:inline('audio',payload,**options),video_to_base64_string=lambda payload,**options:inline('video',payload,**options),get_number_of_images=lambda value:value.shape[0]if len(value.shape)==4 else 1,validate_video_duration=duration,get_image_from_response=image_output,get_video_from_response=video_output)
    validation=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text());function=next(node for node in validation.body if isinstance(node,ast.FunctionDef)and node.name=='validate_string');exec(compile(strip(ast.Module(body=[copy.deepcopy(function)],type_ignores=[])),'<source-validation>','exec'),namespace)
    nodes=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());selected=[];contracts={}
    helpers={'create_image_parts','_mime_matches','get_parts_by_type','get_text_from_response','calculate_tokens_price','create_video_parts','create_audio_parts','_flatten_images','_flatten_audio','_media_url_part','_media_inline_part','build_gemini_media_parts'}
    for node in nodes.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            name=ast.unparse(node.targets[0]if isinstance(node,ast.Assign)else node.target)
            if name!='GEMINI_IMAGE_2_PRICE_BADGE':selected.append(copy.deepcopy(node))
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name in helpers:selected.append(copy.deepcopy(node))
        elif isinstance(node,ast.ClassDef):
            methods=[copy.deepcopy(member)for member in node.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef))and member.name in{'execute','create_file_part'}]
            if methods:
                selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=methods,decorator_list=[]));execute=next(member for member in methods if member.name=='execute');defaults={argument.arg:ast.literal_eval(default)for argument,default in zip(execute.args.args[-len(execute.args.defaults):],execute.args.defaults)}if execute.args.defaults else{}
                contracts[node.name]={'arguments':[argument.arg for argument in execute.args.args if argument.arg!='cls'],'defaults':defaults}
                for call in ast.walk(node):
                    if isinstance(call,ast.Call)and ast.unparse(call.func)=='IO.Schema':
                        fields={keyword.arg:keyword.value for keyword in call.keywords};contracts[node.name].update(node_identifier=ast.literal_eval(fields['node_id']),hidden=ast.unparse(fields.get('hidden',ast.List(elts=[],ctx=ast.Load()))),outputs=ast.unparse(fields['outputs']))
    exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<source-node-execute>','exec'),namespace)
    cases=[];errors=[]
    async def observe(node,arguments,error=False):
        requests.clear();media.clear()
        try:
            result=namespace[node].execute(**copy.deepcopy(arguments));result=await result if asyncio.iscoroutine(result)else result
        except Exception as failure:
            if not error:raise
            errors.append({'input':{'node':contracts[node]['node_identifier'],'arguments':input_json(arguments)},'message':str(failure),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media)})
        else:
            if error:raise ValueError('Expected source failure')
            cases.append({'input':{'node':contracts[node]['node_identifier'],'arguments':input_json(arguments)},'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'outputs':input_json(list(result))})
    image=Image('image',12);audio={'waveform':Waveform('audio',2),'sample_rate':48000};video=Video('video');files=[namespace['GeminiPart'](inlineData=namespace['GeminiInlineData'](mimeType=namespace['GeminiMimeType'].text_plain,data=base64.b64encode(b'source text').decode()))]
    await observe('GeminiNode',dict(prompt=' legacy ',model='gemini-3-1-pro',seed=42,images=image,audio=audio,video=video,files=files,system_prompt='source system'))
    await observe('GeminiNode',dict(prompt='',model='gemini-3-1-flash-lite',seed=0))
    model=dict(model='Gemini 3.1 Pro',temperature=1.0,top_p=0.95,max_output_tokens=32768,thinking_level='HIGH',images={'image_2':Image('later',12),'image_1':Image('earlier')},audio={'audio_1':audio},video={'video_1':video},files=files)
    await observe('GeminiNodeV2',dict(prompt=' v2 ',model=model,seed=42,system_prompt=' system '))
    await observe('GeminiNodeV2',dict(prompt='v2 lite',model={key:value for key,value in{**model,'model':'Gemini 3.1 Flash-Lite'}.items()if key not in{'images','audio','video','files'}},seed=0))
    await observe('GeminiInputFiles',dict(file='input.txt',GEMINI_INPUT_FILES=files))
    await observe('GeminiImage',dict(prompt=' image ',model='gemini-2.5-flash-image',seed=42,images=Image('one'),files=files,aspect_ratio='',response_modalities='IMAGE',system_prompt='sys'))
    image2=dict(prompt='image2',model='Nano Banana 2 (Gemini 3.1 Flash Image)',seed=18446744073709551615,aspect_ratio='16:9',resolution='2K',response_modalities='IMAGE+TEXT',images=image,files=files,system_prompt='')
    await observe('GeminiImage2',image2)
    await observe('GeminiNanoBanana2',{**image2,'thinking_level':'HIGH'})
    nano=dict(prompt=' nano ',model={'model':'Nano Banana 2 Lite','resolution':'1K','aspect_ratio':'auto','thinking_level':'MINIMAL','images':{'image_2':Image('second',2),'image_1':Image('first')},'files':files},seed=42,response_modalities='IMAGE+TEXT',system_prompt='',temperature=0.5,top_p=0.7)
    await observe('GeminiNanoBanana2V2',nano)
    omni=dict(model={'model':'Omni Flash','prompt':'generate video','images':{'image_1':Image('omni',10)},'videos':{'video_2':Video('second-video'),'video_1':Video('first-video')},'temperature':0.9,'top_p':0.8},seed=42)
    await observe('GeminiVideoOmni',omni)
    await observe('GeminiNodeV2',dict(prompt=' ',model=model,seed=42),True)
    await observe('GeminiImage2',{**image2,'images':Image('too-many',15)},True)
    await observe('GeminiNanoBanana2V2',{**nano,'model':{**nano['model'],'images':{'images':Image('too-many',15)}}},True)
    await observe('GeminiVideoOmni',{**omni,'model':{**omni['model'],'videos':{f'video_{index}':Video(str(index))for index in range(4)}}},True)
    await observe('GeminiVideoOmni',{**omni,'model':{**omni['model'],'videos':{'video_1':Video('long',11)}}},True)
    prices=[]
    for model in ['gemini-2.5-pro','gemini-2.5-flash','gemini-2.5-flash-image','gemini-3-pro-preview','gemini-3.1-pro-preview','gemini-3.1-flash-lite-preview','gemini-3.1-flash-lite','gemini-3-pro-image-preview','gemini-3-pro-image','gemini-3.1-flash-image-preview','gemini-3.1-flash-image','gemini-3.1-flash-lite-image','gemini-omni-flash-preview','unknown',None]:
        value=response(model);prices.append({'response':value.model_dump(),'usd':namespace['calculate_tokens_price'](value)})
    response_cases=[]
    variants=[response(),namespace['GeminiGenerateContentResponse'](candidates=[],promptFeedback=namespace['GeminiPromptFeedback'](blockReason='SAFETY',blockReasonMessage='blocked')),namespace['GeminiGenerateContentResponse'](candidates=[]),namespace['GeminiGenerateContentResponse'](candidates=[namespace['GeminiCandidate'](finishReason='IMAGE_PROHIBITED_CONTENT',content=namespace['GeminiContent'](role=namespace['GeminiRole'].model,parts=[namespace['GeminiPart'](text='blocked')]))]),namespace['GeminiGenerateContentResponse'](candidates=[namespace['GeminiCandidate'](content=namespace['GeminiContent'](role=namespace['GeminiRole'].model,parts=[namespace['GeminiPart'](text='reasoning',thought=True),namespace['GeminiPart'](text='final'),namespace['GeminiPart'](fileData=namespace['GeminiFileData'](fileUri='https://fixture.invalid/image.png',mimeType=namespace['GeminiMimeType'].image_png))]))])]
    for value in variants:
        try:text=namespace['get_text_from_response'](value);parts=namespace['get_parts_by_type'](value,'image/*');result={'text':text,'image_parts':[part.model_dump()for part in parts]}
        except Exception as failure:result={'error':str(failure)}
        response_cases.append({'response':value.model_dump(),'result':result})
    return {'schema_version':1,'namespace':'zed.sim.provider.gemini','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest()for path in SOURCES},'scope':'Actual request/helper/response filtering/token price AST. Native encodings and materialization are explicit boundary probes, never decoded outputs.','contracts':contracts,'model_maps':{'text_v2':namespace['GEMINI_V2_MODELS'],'omni':namespace['OMNI_MODELS']},'cases':cases,'preflight_errors':errors,'price_cases':prices,'price_failure_policy':'shared client suppresses price-extractor exceptions and continues returning the validated response','response_cases':response_cases}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');args=parser.parse_args();SOURCE_ROOT=args.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode()
    if args.check:
        if DESTINATION.read_bytes()!=encoded:raise SystemExit('Gemini source contract differs')
    else:DESTINATION.write_bytes(encoded)
