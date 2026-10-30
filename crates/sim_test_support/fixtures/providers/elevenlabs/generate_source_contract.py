"""Execute pinned ElevenLabs request AST with explicit audio preparation probes."""
import argparse,ast,asyncio,copy,hashlib,json,uuid
from pathlib import Path
from types import SimpleNamespace
SOURCE_ROOT=Path(__file__).resolve().parents[5]/'projects/sim/ComfyUI'
DESTINATION=Path(__file__).with_name('source-contract.json')
SOURCES=['sim_api_nodes/nodes_elevenlabs.py','sim_api_nodes/apis/elevenlabs.py','sim_api_nodes/util/client.py','sim_api_nodes/util/_helpers.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/conversions.py']
class Request:
    defaults={}
    def __init__(self,**fields):self.__dict__.update(copy.deepcopy(self.defaults));self.__dict__.update(fields)
    def model_dump(self,exclude_none=True):
        def encode(value):
            if isinstance(value,Request):return value.model_dump(exclude_none=exclude_none)
            if isinstance(value,list):return [encode(member) for member in value]
            return value
        return {key:encode(value) for key,value in vars(self).items() if not exclude_none or value is not None}
    def model_dump_json(self,exclude_none=True):return json.dumps(self.model_dump(exclude_none),ensure_ascii=False,separators=(',',':'))
def field(value=None,default=None,**unused):return default if default is not None else value
class AudioProbe:
    def __init__(self,slot):self.slot=slot
class EncodedProbe:
    def __init__(self,slot):self.slot=slot
    def getvalue(self):return self
    def seek(self,offset):return None
class FormProbe:
    def __init__(self):self.fields=[]
    def add_field(self,name,value,**options):self.fields.append({'name':name,'value':value,**options})
def strip(node):
    for member in ast.walk(node):
        if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
            member.returns=None
            for argument in [*member.args.posonlyargs,*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
    return ast.fix_missing_locations(node)
def input_json(value):
    if isinstance(value,AudioProbe):return {'source_audio':value.slot}
    if isinstance(value,dict):return {key:input_json(member) for key,member in value.items()}
    return value
async def generate():
    requests=[];uploads=[];preparations=[]
    namespace={'__builtins__':__builtins__,'classmethod':classmethod,'BaseModel':Request,'Field':field,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'json':json,'uuid':SimpleNamespace(uuid4=lambda:uuid.UUID('00000000-0000-4000-8000-000000000042')),'ApiEndpoint':lambda path,method='GET',query_params=None:SimpleNamespace(path=path,method=method,query_params=query_params)}
    client=ast.parse((SOURCE_ROOT/SOURCES[2]).read_text())
    form_loop=next(node for node in ast.walk(client) if isinstance(node,ast.For) and ast.unparse(node.iter)=='cfg.data.items()' and node.lineno>650)
    exec(compile(strip(ast.Module(body=[copy.deepcopy(form_loop)],type_ignores=[])),'<source-multipart-loop>','exec'),{'cfg':SimpleNamespace(data={}), 'form':FormProbe()})
    async def sync(cls,endpoint,data=None,files=None,content_type='application/json',as_binary=False,**options):
        body=data.model_dump() if data else None;form=FormProbe()
        if content_type=='multipart/form-data':
            exec(compile(strip(ast.Module(body=[copy.deepcopy(form_loop)],type_ignores=[])),'<source-multipart-loop>','exec'),{'cfg':SimpleNamespace(data=body or {}),'form':form})
        file_fields=[]
        for name,(filename,payload,mime_type) in (files if isinstance(files,list) else (files or {}).items()):
            file_fields.append({'name':name,'file_name':filename,'media_type':mime_type,'source_slot':payload.slot,'boundary':'canonical MP4/AAC owner required'})
        requests.append({'path':endpoint.path,'method':endpoint.method,'query_params':endpoint.query_params,'body':body,'content_type':content_type,'form_fields':form.fields,'file_fields':file_fields,'as_binary':as_binary})
        return 'canonical audio decoder boundary' if as_binary else SimpleNamespace(text='source transcript',language_code='en',words=[namespace['SpeechToTextWord'](text='café ☁',start=0.0,end=1.5)],voice_id='created-voice-42')
    async def upload(cls,audio,container_format,codec_name,mime_type):
        uploads.append({'slot':audio['waveform'].slot,'container':container_format,'codec':codec_name,'mime_type':mime_type,'url':'https://storage.fixture.invalid/'+audio['waveform'].slot});return uploads[-1]['url']
    def encode(waveform,sample_rate,container,codec):
        preparations.append({'slot':waveform.slot,'sample_rate':sample_rate,'container':container,'codec':codec});return EncodedProbe(waveform.slot)
    namespace.update(sync_op=sync,sync_op_raw=sync,upload_audio_to_simapi=upload,audio_tensor_to_contiguous_ndarray=lambda waveform:waveform,audio_ndarray_to_bytesio=encode,audio_bytes_to_audio_input=lambda value:value)
    for node in ast.parse((SOURCE_ROOT/SOURCES[1]).read_text()).body:
        if isinstance(node,ast.ClassDef):
            defaults={}
            for member in node.body:
                if isinstance(member,ast.AnnAssign):defaults[member.target.id]=eval(compile(ast.Expression(member.value),'<source-api-default>','eval'),namespace)
            namespace[node.name]=type(node.name,(Request,),{'defaults':defaults})
    validation=ast.parse((SOURCE_ROOT/SOURCES[4]).read_text());function=next(node for node in validation.body if isinstance(node,ast.FunctionDef) and node.name=='validate_string')
    exec(compile(strip(ast.Module(body=[copy.deepcopy(function)],type_ignores=[])),'<source-validation>','exec'),namespace)
    nodes=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());selected=[];contracts={}
    for node in nodes.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            name=ast.unparse(node.targets[0] if isinstance(node,ast.Assign) else node.target)
            if name!='ELEVENLABS_STS_VOICE_SETTINGS':selected.append(copy.deepcopy(node))
        elif isinstance(node,ast.ClassDef):
            methods=[copy.deepcopy(member) for member in node.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and member.name=='execute']
            if methods:selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=methods,decorator_list=[]));contracts[node.name]={'arguments':[argument.arg for argument in methods[0].args.args if argument.arg!='cls']}
            for call in ast.walk(node):
                if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema':
                    fields={keyword.arg:keyword.value for keyword in call.keywords};contracts[node.name].update(hidden=ast.unparse(fields.get('hidden',ast.List(elts=[],ctx=ast.Load()))),outputs=ast.unparse(fields['outputs']))
                    if 'price_badge' in fields:
                        badge=fields['price_badge'];contracts[node.name]['price_badge_expr']=next(ast.literal_eval(keyword.value) for keyword in badge.keywords if keyword.arg=='expr')
    exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<source-execute>','exec'),namespace)
    cases=[];errors=[]
    async def observe(node,arguments,error=False):
        requests.clear();uploads.clear();preparations.clear()
        try:
            result=namespace[node].execute(**copy.deepcopy(arguments))
            if asyncio.iscoroutine(result):result=await result
        except Exception as failure:
            if not error:raise
            errors.append({'input':{'node':node,'arguments':input_json(arguments)},'message':str(failure),'requests':copy.deepcopy(requests),'uploads':copy.deepcopy(uploads)})
        else:
            if error:raise ValueError('Expected pinned source failure')
            cases.append({'input':{'node':node,'arguments':input_json(arguments)},'requests':copy.deepcopy(requests),'uploads':copy.deepcopy(uploads),'preparations':copy.deepcopy(preparations),'outputs':list(result)})
    audio={'waveform':AudioProbe('audio'),'sample_rate':48000}
    transcript=dict(audio=audio,model={'model':'scribe_v2','diarize':False,'tag_audio_events':True,'timestamps_granularity':'word','diarization_threshold':0.22,'temperature':0.75},language_code='  en  ',num_speakers=2,seed=42)
    await observe('ElevenLabsSpeechToText',transcript)
    await observe('ElevenLabsSpeechToText',{**transcript,'model':{**transcript['model'],'diarize':True},'language_code':' ','num_speakers':0})
    await observe('ElevenLabsVoiceSelector',dict(voice='Roger (male, american)'))
    tts=dict(voice='CwhRBWXzGAHq8TQ4Fs17',text='  hello café ☁  ',stability=0.5,apply_text_normalization='auto',model={'model':'eleven_multilingual_v2','similarity_boost':0.75,'speed':1.0,'use_speaker_boost':False,'style':0.0},language_code=' en ',seed=42,output_format='mp3_44100_192')
    await observe('ElevenLabsTextToSpeech',tts)
    await observe('ElevenLabsTextToSpeech',{**tts,'model':{'model':'eleven_v3','similarity_boost':0.75,'speed':1.0},'language_code':'','output_format':'opus_48000_192'})
    await observe('ElevenLabsAudioIsolation',dict(audio=audio))
    sfx=dict(text='  breeze ☁  ',model={'model':'eleven_sfx_v2','duration':5.5,'prompt_influence':0.3,'loop':True},output_format='opus_48000_192')
    await observe('ElevenLabsTextToSoundEffects',sfx)
    await observe('ElevenLabsInstantVoiceClone',dict(files={'sample_2':audio,'sample_1':{'waveform':AudioProbe('audio2'),'sample_rate':24000}},remove_background_noise=True))
    sts=dict(voice='CwhRBWXzGAHq8TQ4Fs17',audio=audio,stability=0.5,model={**tts['model'],'model':'eleven_multilingual_sts_v2'},output_format='mp3_44100_192',seed=42,remove_background_noise=False)
    await observe('ElevenLabsSpeechToSpeech',sts)
    dialogue=dict(stability=0.5,apply_text_normalization='off',model='eleven_v3',inputs={'inputs':'2','text1':'hi café','voice1':'first-voice','text2':'hello ☁','voice2':'second-voice'},language_code=' ',seed=4294967295,output_format='mp3_44100_192')
    await observe('ElevenLabsTextToDialogue',dialogue)
    await observe('ElevenLabsSpeechToText',{**transcript,'model':{**transcript['model'],'diarize':True}},True)
    await observe('ElevenLabsVoiceSelector',dict(voice='unknown voice'),True)
    await observe('ElevenLabsTextToSpeech',{**tts,'text':' '},True)
    await observe('ElevenLabsTextToSoundEffects',{**sfx,'text':' '},True)
    await observe('ElevenLabsTextToDialogue',{**dialogue,'inputs':{**dialogue['inputs'],'text2':' '}},True)
    return {'schema_version':1,'namespace':'zed.sim.provider.elevenlabs','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'scope':'Actual pinned node/request/default and multipart plain-field AST; MP4/AAC input encoding, storage upload, UUID generation and native audio/custom voice materialization are explicit canonical owner boundaries.','contracts':contracts,'voice_map':namespace['ELEVENLABS_VOICE_MAP'],'cases':cases,'preflight_errors':errors}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');args=parser.parse_args();SOURCE_ROOT=args.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode()
    if args.check:
        if DESTINATION.read_bytes()!=encoded:raise SystemExit('ElevenLabs source contract differs')
    else:DESTINATION.write_bytes(encoded)
