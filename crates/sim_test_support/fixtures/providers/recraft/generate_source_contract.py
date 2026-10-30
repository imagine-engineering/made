"""Independent pinned Recraft execute, API, multipart and tiny image oracle."""
import argparse,ast,asyncio,copy,hashlib,json,math,inspect,sys
from enum import Enum
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace,ModuleType
import aiohttp,numpy as np,torch
from PIL import Image,UnidentifiedImageError
ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI');DEST=Path(__file__).with_name('source-contract.json')
FILES=['sim_api_nodes/nodes_recraft.py','sim_api_nodes/apis/recraft.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/client.py','sim_api_nodes/util/download_helpers.py','sim/utils.py','sim_api/latest/_util/image_types.py']
NODES={'RecraftColorRGB':'RecraftColorRGBNode','RecraftControls':'RecraftControlsNode','RecraftStyleV3RealisticImage':'RecraftStyleV3RealisticImageNode','RecraftStyleV3DigitalIllustration':'RecraftStyleV3DigitalIllustrationNode','RecraftStyleV3LogoRaster':'RecraftStyleV3LogoRasterNode','RecraftStyleV3InfiniteStyleLibrary':'RecraftStyleInfiniteStyleLibrary','RecraftCreateStyleNode':'RecraftCreateStyleNode','RecraftTextToImageNode':'RecraftTextToImageNode','RecraftImageToImageNode':'RecraftImageToImageNode','RecraftImageInpaintingNode':'RecraftImageInpaintingNode','RecraftTextToVectorNode':'RecraftTextToVectorNode','RecraftVectorizeImageNode':'RecraftVectorizeImageNode','RecraftReplaceBackgroundNode':'RecraftReplaceBackgroundNode','RecraftRemoveBackgroundNode':'RecraftRemoveBackgroundNode','RecraftCrispUpscaleNode':'RecraftCrispUpscaleNode','RecraftCreativeUpscaleNode':'RecraftCreativeUpscaleNode','RecraftV4TextToImageNode':'RecraftV4TextToImageNode','RecraftV4TextToVectorNode':'RecraftV4TextToVectorNode'}
torch.set_num_threads(1)
def selected(path,names,namespace,classes=False):
    output=[]
    for original in ast.parse((ROOT/path).read_text()).body:
        if getattr(original,'name',None) not in names:continue
        node=copy.deepcopy(original)
        if classes:node.body=[member for member in node.body if not isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)) or member.name!='define_schema']
        for member in ast.walk(node):
            if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
                member.returns=None
                for argument in [*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
        output.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=output,type_ignores=[])),str(ROOT/path),'exec'),namespace)
def tensor(desc,mask=False):
    shape=(desc.get('batch',1),desc.get('height',2),desc.get('width',2))
    if mask:return torch.full(shape,desc.get('value',.5),dtype=torch.float32)
    result=torch.empty((*shape,desc.get('channels',3)),dtype=torch.float32)
    for channel in range(result.shape[-1]):result[...,channel]=[.5,.25,.125,1.][channel]
    return result
async def generate():
    requests=[];downloads=[];progress=[];phases=[];case_options={};namespace={'__builtins__':__builtins__,'torch':torch,'np':np,'Image':Image,'BytesIO':BytesIO,'math':math,'Enum':Enum,'aiohttp':aiohttp,'UnidentifiedImageError':UnidentifiedImageError,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values)}
    api_module=ModuleType('pinned_recraft_api');sys.modules[api_module.__name__]=api_module
    exec(compile((ROOT/FILES[1]).read_text(),str(ROOT/FILES[1]),'exec'),api_module.__dict__);namespace.update(api_module.__dict__)
    selected(FILES[6],{'common_upscale'},namespace)
    selected(FILES[2],{'bytesio_to_image_tensor','tensor_to_bytesio','tensor_to_pil','pil_to_bytesio','downscale_image_tensor','_compute_downscale_dims','resize_mask_to_image'},namespace)
    namespace.update(uuid=SimpleNamespace(uuid4=lambda:'00000000-0000-4000-8000-000000000000'),mimetype_to_extension=lambda mime:mime.split('/')[-1].lower())
    selected(FILES[3],{'validate_string'},namespace)
    selected(FILES[7],{'SVG'},namespace)
    image_stream=BytesIO();Image.new('RGBA',(2,2),(128,64,32,128)).save(image_stream,format='PNG');output_png=image_stream.getvalue();output_svg=b'<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><path d="M0 0L2 2"/></svg>'
    class ProgressBar:
        def __init__(self,total):phases.append({'progress_total':total});progress.append({'total':total})
        def update(self,value):phases.append({'progress_update':value});progress.append({'update':value})
    class Writer:
        def __init__(self):self.bytes=bytearray()
        async def write(self,data):self.bytes.extend(data)
    async def sync(cls,endpoint,response_model,data=None,files=None,**options):
        body=data.model_dump(exclude_none=True) if data is not None else None;parts=[];form_fields=[];wire=None
        if files is not None:
            for field,value in files.items() if isinstance(files,dict) else files:
                if isinstance(value,tuple):filename,value,mime=value
                else:filename=field;mime='application/octet-stream'
                raw=value.getvalue() if hasattr(value,'getvalue') else value
                parts.append({'field':field,'file_name':filename,'mime_type':mime,'bytes_hex':raw.hex()})
        if options.get('content_type')=='multipart/form-data':
            form=options['multipart_parser'](body) if options.get('multipart_parser') and body else aiohttp.FormData(default_to_multipart=True)
            if not options.get('multipart_parser') or not body:
                for key,value in (body or {}).items():form.add_field(key,str(value))
            for attrs,headers,value in form._fields:form_fields.append({'name':attrs['name'],'value':value})
            for part in parts:form.add_field(part['field'],bytes.fromhex(part['bytes_hex']),filename=part['file_name'],content_type=part['mime_type'])
            form._writer=aiohttp.MultipartWriter('form-data',boundary='sim-recraft-canonical-boundary');writer=Writer();await form().write(writer);wire=writer.bytes.hex()
        phases.append({'request':endpoint.path});requests.append({'path':endpoint.path,'method':endpoint.method,'body':body,'files':parts,'form_fields':form_fields,'content_type':options.get('content_type','application/json'),'wire_hex':wire,'max_retries':options.get('max_retries',3)})
        if response_model is namespace['RecraftCreateStyleResponse']:return response_model.model_validate({'id':'style-42'})
        count=(body or {}).get('n',1)
        urls=[{'image_id':f'image-{index}','url':f'https://source.fixture.invalid/{len(requests)}-{index}.bin'} for index in range(max(count,0))]
        response={'created':12,'credits':34,'data':urls}
        if case_options.get('single_image'):response['image']={'image_id':'preferred-single','url':'https://source.fixture.invalid/preferred-single.bin'}
        if case_options.get('missing_data'):response['data']=None
        if case_options.get('malformed_image'):response['image']={'url':'https://source.fixture.invalid/invalid.bin'}
        return response_model.model_validate(response)
    async def download(url,**options):
        request=requests[-1];vector=(request['body'] or {}).get('style')=='vector_illustration' or request['path'].endswith('/vectorize')
        raw=output_svg if vector else output_png
        if case_options.get('corrupt_first') and not downloads:raw=b'not an encoded image'
        phases.append({'download':url});downloads.append({'url':url,'timeout':options.get('timeout'),'bytes_hex':raw.hex()});return BytesIO(raw)
    original_decode=namespace['bytesio_to_image_tensor']
    def observed_decode(stream):
        phases.append({'decode_sha256':hashlib.sha256(stream.getvalue()).hexdigest()});return original_decode(stream)
    namespace['bytesio_to_image_tensor']=observed_decode
    namespace.update(sync_op=sync,download_url_as_bytesio=download,ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method),ProgressBar=ProgressBar)
    selected(FILES[0],{'recraft_multipart_parser','handle_recraft_file_request'},namespace)
    selected(FILES[0],{'handle_recraft_image_output',*NODES.values()},namespace,True)
    def normalize(value):
        if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'pixels':value.flatten().tolist()}
        if isinstance(value,namespace['SVG']):return {'svg_bytes':[stream.getvalue().hex() for stream in value.data]}
        if isinstance(value,namespace['RecraftColorChain']):return {'kind':'color','colors':[color.color for color in value.colors],'api':[x.model_dump(exclude_none=True) for x in value.create_api_model()] if value.create_api_model() else None}
        if isinstance(value,namespace['RecraftControls']):return {'kind':'controls','colors':normalize(value.colors) if value.colors is not None else None,'background_color':normalize(value.background_color) if value.background_color is not None else None,'api':value.create_api_model().model_dump(exclude_none=True) if value.create_api_model() else None}
        if isinstance(value,namespace['RecraftStyle']):return {'kind':'style','style':value.style,'substyle':value.substyle,'style_id':value.style_id}
        if isinstance(value,(list,tuple)):return [normalize(item) for item in value]
        return value
    def chain(values):
        result=namespace['RecraftColorChain']()
        for color in values:result.add(namespace['RecraftColor'](*color))
        return result
    def base(node):
        settings={'prompt':' ☁ ','n':2,'seed':7,'size':'1024x1024','strength':.555,'negative_prompt':' bad ','substyle':'None'};images=[];mask=None
        if node=='RecraftColorRGB':settings={'r':12,'g':34,'b':56}
        elif node=='RecraftControls':settings={}
        elif node.startswith('RecraftStyleV3'):
            settings={'style_id':'style-uuid'} if node.endswith('InfiniteStyleLibrary') else {'substyle':'None'}
        elif node=='RecraftCreateStyleNode':settings={'style':'realistic_image'};images=[{'role':'imageZ','batch':1,'width':2,'height':2}]
        elif 'V4' in node:settings.update(model={'model':'recraftv4' if 'Image' in node else 'recraftv4_vector','size':'1024x1024'})
        if node in {'RecraftImageToImageNode','RecraftImageInpaintingNode','RecraftVectorizeImageNode','RecraftReplaceBackgroundNode','RecraftRemoveBackgroundNode','RecraftCrispUpscaleNode','RecraftCreativeUpscaleNode'}:images=[{'role':'image','batch':1,'width':2,'height':2}]
        if node=='RecraftImageInpaintingNode':mask={'batch':1,'width':1,'height':1,'value':.5}
        return {'node':node,'settings':settings,'images':images,'mask':mask,'style':None,'controls':None,'colors':None,'background_color':None}
    def kwargs(value):
        node=value['node'];settings=value['settings'];method=namespace[NODES[node]].execute;parameters=list(inspect.signature(method).parameters)
        output={key:copy.deepcopy(settings[key]) for key in parameters if key in settings}
        if 'image' in parameters:output['image']=tensor(value['images'][0])
        if 'images' in parameters:output['images']={entry['role']:tensor(entry) for entry in value['images']}
        if value['mask'] is not None:output['mask']=tensor(value['mask'],True)
        if value['style'] is not None:output['recraft_style']=namespace['RecraftStyle'](**value['style'])
        if value['controls'] is not None:output['recraft_controls']=namespace['RecraftControls'](colors=chain(value['controls']['colors']) if value['controls'].get('colors') is not None else None,background_color=chain(value['controls']['background_color']) if value['controls'].get('background_color') is not None else None)
        if node=='RecraftColorRGB' and value['colors'] is not None:output['recraft_color']=chain(value['colors'])
        if node=='RecraftControls':
            for key in ['colors','background_color']:
                if value[key] is not None:output[key]=chain(value[key])
        return output
    cases=[];errors=[]
    async def observe(value):
        requests.clear();downloads.clear();progress.clear();phases.clear();case_options.clear();case_options.update(value.get('oracle_options',{}))
        try:
            result=namespace[NODES[value['node']]].execute(**kwargs(value));result=await result if asyncio.iscoroutine(result) else result
        except Exception as error:errors.append({'input':copy.deepcopy(value),'error_type':type(error).__name__,'requests':copy.deepcopy(requests),'downloads':copy.deepcopy(downloads),'progress':copy.deepcopy(progress),'phases':copy.deepcopy(phases)});return
        cases.append({'input':copy.deepcopy(value),'requests':copy.deepcopy(requests),'downloads':copy.deepcopy(downloads),'progress':copy.deepcopy(progress),'phases':copy.deepcopy(phases),'output':normalize(result)})
    for node in NODES:await observe(base(node))
    facts={};price_badges={}
    for cls in ast.parse((ROOT/FILES[0]).read_text()).body:
        if not isinstance(cls,ast.ClassDef) or cls.name not in NODES.values():continue
        node=next(key for key,value in NODES.items() if value==cls.name);schema=next(member for member in cls.body if isinstance(member,ast.FunctionDef) and member.name=='define_schema');records=[]
        for call in ast.walk(schema):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='PriceBadge':
                price_badges[node]=''.join(next(keyword.value.value for keyword in call.keywords if keyword.arg=='expr').split())
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or call.func.attr!='Input' or not call.args or not isinstance(call.args[0],ast.Constant):continue
            record={'name':call.args[0].value,'type':ast.unparse(call.func.value)}
            for keyword in call.keywords:
                if keyword.arg in {'options','default','min','max'}:
                    try:record[keyword.arg]=ast.literal_eval(keyword.value)
                    except (ValueError,TypeError):pass
            records.append(record)
            for option in record.get('options',[])+[record[key] for key in ['min','max'] if key in record]:
                value=base(node)
                if record['name'] in value['settings'] and not isinstance(value['settings'][record['name']],dict):value['settings'][record['name']]=option;await observe(value)
        facts[node]=records
    for node in NODES:
        value=base(node)
        if value['images']:
            value['images'][0]['batch']=2;await observe(value)
        if 'prompt' in value['settings']:
            for prompt in ['', ' ', '☁'*1001, '☁'*10001]:value=base(node);value['settings']['prompt']=prompt;await observe(value)
        if node in {'RecraftTextToImageNode','RecraftImageToImageNode','RecraftImageInpaintingNode','RecraftReplaceBackgroundNode'}:
            value=base(node);value['controls']={'colors':[[1,2,3],[4,5,6]],'background_color':[[7,8,9],[10,11,12]]};await observe(value)
            for style in [{'style':'digital_illustration','substyle':'None'},{'style_id':'style-42'}]:value=base(node);value['style']=style;await observe(value)
    for style in ['realistic_image','digital_illustration','logo_raster']:
        node={'realistic_image':'RecraftStyleV3RealisticImage','digital_illustration':'RecraftStyleV3DigitalIllustration','logo_raster':'RecraftStyleV3LogoRaster'}[style]
        for substyle in namespace['get_v3_substyles'](style):value=base(node);value['settings']['substyle']=substyle;await observe(value)
    value=base('RecraftColorRGB');value['colors']=[[1,2,3],[4,5,6]];await observe(value)
    value=base('RecraftControls');value['colors']=[[1,2,3],[4,5,6]];value['background_color']=[[7,8,9],[10,11,12]];await observe(value)
    value=base('RecraftStyleV3InfiniteStyleLibrary');value['settings']['style_id']='';await observe(value)
    for strength in [.125,.545,.555,.565,.995]:value=base('RecraftImageToImageNode');value['settings']['strength']=strength;await observe(value)
    for val in [.49,.5,.51]:value=base('RecraftImageInpaintingNode');value['mask']['value']=val;await observe(value)
    for node in ['RecraftV4TextToImageNode','RecraftV4TextToVectorNode']:
        for model in ['recraftv4','recraftv4_pro','recraftv4_vector','recraftv4_pro_vector']:
            for size in namespace['RECRAFT_V4_PRO_SIZES'] if 'pro' in model else namespace['RECRAFT_V4_SIZES']:value=base(node);value['settings']['model']={'model':model,'size':size};await observe(value)
    for node in ['RecraftTextToImageNode','RecraftImageToImageNode','RecraftV4TextToImageNode']:
        for options in [{'single_image':True},{'corrupt_first':True},{'single_image':True,'missing_data':True},{'malformed_image':True}]:
            value=base(node);value['oracle_options']=options;await observe(value)
    for node in ['RecraftVectorizeImageNode','RecraftImageToImageNode']:
        value=base(node);value['images'][0]['batch']=0;await observe(value)
    for node in ['RecraftTextToVectorNode','RecraftTextToImageNode']:
        value=base(node);value['settings']['n']=0;await observe(value)
    return {'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES},'versions':{'torch':torch.__version__,'pillow':Image.__version__,'aiohttp':aiohttp.__version__},'cases':cases,'errors':errors,'schema_facts':facts,'price_badges':price_badges,'sizes':{'v4':namespace['RECRAFT_V4_SIZES'],'v4_pro':namespace['RECRAFT_V4_PRO_SIZES']},'substyles':namespace['dict_recraft_substyles_v3'],'scope':'actual source execute/API/multipart/Torch/Pillow local helpers and downloaded RGBA/SVG; service only mocked, native/guest separate'}
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args();result=asyncio.run(generate());encoded=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if args.check:
    if DEST.read_text()!=encoded:raise SystemExit('source oracle drift')
else:DEST.write_text(encoded)
print(f"Recraft actual source: {len(result['cases'])} successes,{len(result['errors'])} ordered failures")
