"""Pinned actual OpenAI/Sora execute/API oracle; no live service or native gold tuning."""
import argparse,ast,asyncio,base64,copy,hashlib,json,math,os,mimetypes,tempfile
from pathlib import Path
from types import SimpleNamespace
from enum import Enum
from typing import Optional
from io import BytesIO
import aiohttp
import torch
import numpy as np
from PIL import Image
ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI');DEST=Path(__file__).with_name('source-contract.json')
FILES=['sim_api_nodes/nodes_openai.py','sim_api_nodes/nodes_sora.py','sim_api_nodes/apis/openai.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/client.py','sim/utils.py','sim_api_nodes/util/download_helpers.py']
NODES=['OpenAIDalle2','OpenAIDalle3','OpenAIGPTImage1','OpenAIGPTImageNodeV2','OpenAIChatNode','OpenAIChatConfig','OpenAIInputFiles','OpenAIVideoSora2']
torch.set_num_threads(1)
def selected(path,names,namespace,classes=False,methods=None):
    nodes=[]
    for original in ast.parse((ROOT/path).read_text()).body:
        if getattr(original,'name',None) not in names:continue
        node=copy.deepcopy(original)
        if classes:node.body=[member for member in node.body if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) and (methods is None and member.name!='define_schema' or methods is not None and member.name in methods)]
        for member in ast.walk(node):
            if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
                member.returns=None
                for argument in [*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
        nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(ROOT/path),'exec'),namespace)
def tensor(desc,mask=False):
    shape=(desc.get('batch',1),desc.get('height',2),desc.get('width',2))
    if mask:return torch.full(shape,desc.get('value',.25),dtype=torch.float32)
    result=torch.empty((*shape,desc.get('channels',3)),dtype=torch.float32)
    for channel in range(result.shape[-1]):result[...,channel]=[.5,.25,.125,1.][channel]
    return result
async def generate():
    requests=[];polls=[];downloads=[];poll_override={}
    namespace={'__builtins__':__builtins__,'torch':torch,'np':np,'Image':Image,'BytesIO':BytesIO,'math':math,'Enum':Enum,'Optional':Optional,'base64':base64,'os':os,'mimetypes':mimetypes,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),'RESPONSES_ENDPOINT':'/proxy/openai/v1/responses'}
    exec(compile((ROOT/FILES[2]).read_text(),str(ROOT/FILES[2]),'exec'),namespace)
    selected('sim/utils.py',{'common_upscale'},namespace)
    selected('sim_api_nodes/util/conversions.py',{'downscale_image_tensor','_compute_downscale_dims','tensor_to_pil','tensor_to_base64_string','pil_to_bytesio','tensor_to_bytesio','text_filepath_to_base64_string','text_filepath_to_data_uri'},namespace)
    namespace.update(uuid=SimpleNamespace(uuid4=lambda:'00000000-0000-4000-8000-000000000000'),mimetype_to_extension=lambda mime:mime.split('/')[-1].lower())
    selected('sim_api_nodes/util/validation_utils.py',{'validate_string','get_number_of_images'},namespace)
    selected(FILES[0],{'SupportedOpenAIModel','calculate_tokens_price_image_1','calculate_tokens_price_image_1_5','calculate_tokens_price_image_2_0','validate_and_cast_response'},namespace)
    selected(FILES[1],{'Sora2GenerationRequest','Sora2GenerationResponse'},namespace)
    image_stream=BytesIO();Image.new('RGBA',(2,2),(128,64,32,255)).save(image_stream,format='PNG');output_bytes=image_stream.getvalue()
    def record_files(files):
        if files is None:return []
        entries=list(files.items()) if isinstance(files,dict) else files
        return [{'field':field,'file_name':value[0],'mime_type':value[2],'bytes_hex':value[1].getvalue().hex()} for field,value in entries]
    async def sync(cls,endpoint,response_model,data,**options):
        requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(exclude_none=True),'content_type':options.get('content_type','application/json'),'files':record_files(options.get('files')),'form_fields':[{'name':key,'value':str(value)} for key,value in data.model_dump(exclude_none=True).items()] if options.get('content_type')=='multipart/form-data' else []})
        if response_model is namespace['OpenAIImageGenerationResponse']:
            result=response_model.model_validate({'data':[{'b64_json':base64.b64encode(output_bytes).decode()}],'usage':{'input_tokens':12,'output_tokens':34}})
            requests[-1]['price_usd']=options['price_extractor'](result) if options.get('price_extractor') else None
            return result
        if response_model is namespace['Sora2GenerationResponse']:return response_model.model_validate({'id':'video-42','status':'queued'})
        return response_model.model_validate({'id':'response-42','status':'queued'})
    async def poll(cls,endpoint=None,response_model=None,poll_endpoint=None,**options):
        endpoint=endpoint or poll_endpoint
        polls.append({'path':endpoint.path,'method':endpoint.method,'poll_interval':options.get('poll_interval',5.),'completed_statuses':options.get('completed_statuses'),'estimated_duration':options.get('estimated_duration')})
        if response_model is namespace['Sora2GenerationResponse']:return response_model.model_validate({'id':'video-42','status':'completed'})
        if poll_override:return response_model.model_validate(poll_override)
        return response_model.model_validate({'id':'response-42','status':'incomplete','output':[{'type':'reasoning','content':[]},{'type':'message','content':[{'type':'other','text':'ignored'},{'type':'output_text','text':'first ☁'},{'type':'output_text','text':'ignored second'}]}]})
    async def download(url,destination,**options):downloads.append({'url':url,'timeout':options.get('timeout'),'cls':options.get('cls') is not None});destination.write(output_bytes)
    async def video(url,**options):downloads.append({'url':url,'cls':options.get('cls') is not None});return {'video_url':url}
    namespace.update(sync_op=sync,poll_op=poll,download_url_to_bytesio=download,download_url_to_video_output=video,ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method))
    selected(FILES[0],set(NODES[:-1]),namespace,True)
    selected(FILES[1],{NODES[-1]},namespace,True)
    def base(node):
        settings={NODES[0]:{'prompt':' ☁ ','seed':7,'n':1,'size':'1024x1024'},NODES[1]:{'prompt':' ☁ ','seed':7,'style':'natural','quality':'standard','size':'1024x1024'},NODES[2]:{'prompt':' ☁ ','seed':7,'quality':'low','background':'opaque','n':1,'size':'1024x1024','custom_width':1024,'custom_height':1024,'model':'gpt-image-1'},NODES[3]:{'prompt':' ☁ ','model':{'model':'gpt-image-1','size':'1024x1024','background':'opaque','quality':'low'},'n':1,'seed':7},NODES[4]:{'prompt':' ☁ ','persist_context':False,'model':'gpt-5'},NODES[5]:{'truncation':'auto','instructions':' ☁ ','max_output_tokens':4096},NODES[6]:{'file':'note.txt'},NODES[7]:{'model':'sora-2','prompt':' ☁ ','size':'1280x720','duration':8,'seed':7}}[node]
        return {'node':node,'settings':settings,'images':[],'mask':None,'advanced_options':None,'files':None}
    def kwargs(value):
        output=copy.deepcopy(value['settings']);images=value['images'];mask=value['mask']
        if value['node']==NODES[3]:
            output['model']['images']={entry['role']:tensor(entry) if not entry.get('null') else None for entry in images}
            if mask is not None:output['model']['mask']=tensor(mask,True)
        elif value['node']==NODES[4]:
            if images:output['images']=tensor(images[0])
            if value['advanced_options'] is not None:output['advanced_options']=namespace['ModelResponseProperties'].model_validate(value['advanced_options'])
            if value['files'] is not None:output['files']=[namespace['InputFileContent'].model_validate(entry) for entry in value['files']]
        elif value['node']==NODES[6]:
            if value['files'] is not None:output['OPENAI_INPUT_FILES']=[namespace['InputFileContent'].model_validate(entry) for entry in value['files']]
        else:
            if images:output['image']=tensor(images[0])
            if mask is not None:output['mask']=tensor(mask,True)
        return output
    def normalize(value):
        if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'pixels':value.flatten().tolist()}
        if hasattr(value,'model_dump'):return value.model_dump(exclude_none=True)
        if isinstance(value,(tuple,list)):return [normalize(item) for item in value]
        return value
    cases=[];errors=[]
    with tempfile.TemporaryDirectory(prefix='openai-source-file-') as folder:
        (Path(folder)/'note.txt').write_bytes('source ☁\n'.encode());namespace['folder_paths']=SimpleNamespace(get_annotated_filepath=lambda name:str(Path(folder)/name))
        async def observe(value):
            requests.clear();polls.clear();downloads.clear()
            try:
                arguments=kwargs(value);method=namespace[value['node']].execute;output=method(**arguments);output=await output if asyncio.iscoroutine(output) else output
            except Exception as error:errors.append({'input':copy.deepcopy(value),'error_type':type(error).__name__,'requests':len(requests)});return
            cases.append({'input':copy.deepcopy(value),'requests':copy.deepcopy(requests),'polls':copy.deepcopy(polls),'downloads':copy.deepcopy(downloads),'output':normalize(output)})
        for node in NODES:await observe(base(node))
        facts={}
        for path in FILES[:2]:
            for node in ast.parse((ROOT/path).read_text()).body:
                if not isinstance(node,ast.ClassDef) or node.name not in NODES:continue
                schema=next(member for member in node.body if isinstance(member,ast.FunctionDef) and member.name=='define_schema');records=[]
                for call in ast.walk(schema):
                    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or call.func.attr!='Input' or not call.args or not isinstance(call.args[0],ast.Constant):continue
                    record={'name':call.args[0].value,'type':ast.unparse(call.func.value)}
                    for keyword in call.keywords:
                        if keyword.arg in {'options','min','max','default'}:
                            try:record[keyword.arg]=ast.literal_eval(keyword.value)
                            except (ValueError,TypeError):pass
                    records.append(record)
                    for option in record.get('options',[])+[record[key] for key in ['min','max'] if key in record]:
                        value=base(node.name)
                        if record['name'] in value['settings'] and not isinstance(value['settings'][record['name']],dict):value['settings'][record['name']]=option;await observe(value)
                facts[node.name]=records
        for node in [NODES[0],NODES[2],NODES[3],NODES[4],NODES[7]]:
            for batch in [1,2]:
                value=base(node);value['images']=[{'role':'imageZ','batch':batch,'width':2,'height':2}];await observe(value)
                if node in [NODES[0],NODES[2],NODES[3]]:value['mask']={'batch':1,'width':2,'height':2};await observe(value)
            if node in [NODES[0],NODES[2],NODES[3]]:
                value=base(node);value['mask']={'batch':1,'width':2,'height':2};await observe(value)
        for node in [NODES[2],NODES[3]]:
            for model in ['gpt-image-1','gpt-image-1.5','gpt-image-2','unknown']:
                for background in ['opaque','transparent']:
                    value=base(node)
                    if node==NODES[3]:value['settings']['model'].update(model=model,background=background)
                    else:value['settings'].update(model=model,background=background)
                    await observe(value)
            for width,height in [(1024,1024),(1000,1024),(3840,2160),(3856,2160),(256,256),(2048,2048),(3840,1280),(3840,1264)]:
                value=base(node);model=value['settings']['model'] if node==NODES[3] else value['settings'];model.update(model='gpt-image-2',size='Custom',custom_width=width,custom_height=height);await observe(value)
        for persist in [False,True]:
            for config in [None,{'truncation':'auto','instructions':' sys ','max_output_tokens':16},{'temperature':.5,'top_p':.25},{'model':'override'},{'temperature':3.}]:
                value=base(NODES[4]);value['settings']['persist_context']=persist;value['advanced_options']=config;await observe(value)
        for node in [NODES[0],NODES[1],NODES[2],NODES[3],NODES[4],NODES[7]]:
            for prompt in ['', ' ', '☁'*10001]:value=base(node);value['settings']['prompt']=prompt;await observe(value)
        value=base(NODES[3]);value['images']=[{'role':'imageZ','batch':1,'width':2,'height':2},{'role':'imageA','batch':2,'width':3,'height':2}];await observe(value)
        value=base(NODES[6]);value['files']=[{'file_id':'prior-42','type':'input_file','filename':'old.pdf'}];await observe(value)
        value=base(NODES[4]);value['files']=[{'file_id':'prior-42','type':'input_file','filename':'old.pdf'}];await observe(value)
        for truncation in ['disabled',True,None]:value=base(NODES[5]);value['settings']['truncation']=truncation;await observe(value)
    class Writer:
        def __init__(self):self.bytes=bytearray()
        async def write(self,data):self.bytes.extend(data)
    for case in cases:
        for request in case['requests']:
            if request['content_type']!='multipart/form-data':continue
            form=aiohttp.FormData(default_to_multipart=True)
            for field in request['form_fields']:form.add_field(field['name'],field['value'])
            for file in request['files']:form.add_field(file['field'],BytesIO(bytes.fromhex(file['bytes_hex'])),filename=file['file_name'],content_type=file['mime_type'])
            form._writer=aiohttp.MultipartWriter('form-data',boundary='sim-openai-canonical-boundary')
            payload=form()
            writer=Writer();await payload.write(writer)
            request['wire_hex']=writer.bytes.hex();request['actual_payload_content_type']=payload.headers['Content-Type']
    response_cases=[]
    for mode in ['base64-precedes-url','url','mixed-dimensions','missing-data','missing-payload','malformed-base64']:
        first={'b64_json':base64.b64encode(output_bytes).decode(),'url':'https://source.fixture.invalid/image.png'}
        data=[first]
        if mode=='url':data=[{'url':'https://source.fixture.invalid/image.png'}]
        elif mode=='mixed-dimensions':
            stream=BytesIO();Image.new('RGBA',(3,1),(10,20,30,128)).save(stream,format='PNG');data.append({'b64_json':base64.b64encode(stream.getvalue()).decode()})
        elif mode=='missing-data':data=[]
        elif mode=='missing-payload':data=[{}]
        elif mode=='malformed-base64':data=[{'b64_json':'!!!!'}]
        downloads.clear()
        try:
            result=await namespace['validate_and_cast_response'](namespace['OpenAIImageGenerationResponse'].model_validate({'data':data}))
            response_cases.append({'mode':mode,'response':{'data':data},'output':normalize(result),'downloads':copy.deepcopy(downloads)})
        except Exception as error:response_cases.append({'mode':mode,'response':{'data':data},'error_type':type(error).__name__,'downloads':copy.deepcopy(downloads)})
    chat_response_cases=[]
    for output in [[{'type':'message','content':[{'type':'output_text','text':None},{'type':'output_text','text':'later'}]}],[{'type':'message','content':[{'type':'other'}]}],[{'type':'reasoning','content':[]}],[{'type':'message','content':None}]]:
        poll_override.update(id='response-42',status='completed',output=output)
        try:
            result=await namespace['OpenAIChatNode'].execute(prompt='source',model='gpt-5',persist_context=False)
            chat_response_cases.append({'response':copy.deepcopy(poll_override),'output':normalize(result)})
        except Exception as error:chat_response_cases.append({'response':copy.deepcopy(poll_override),'error_type':type(error).__name__})
        poll_override.clear()
    return {'chat_response_cases':chat_response_cases,'response_cases':response_cases,'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES},'versions':{'torch':torch.__version__,'pillow':Image.__version__,'aiohttp':aiohttp.__version__},'cases':cases,'errors':errors,'schema_facts':facts,'response_image_png_hex':output_bytes.hex(),'scope':'actual pinned execute/API/validation/tiny torch image-mask encoding/config/file helpers; decoder output tinyRGBA; native/guest bounds independent'}
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();result=asyncio.run(generate());encoded=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if args.check:
    if DEST.read_text()!=encoded:raise SystemExit('source oracle drift')
else:DEST.write_text(encoded)
print(f"OpenAI actual source: {len(result['cases'])} successes,{len(result['errors'])} ordered failures")
