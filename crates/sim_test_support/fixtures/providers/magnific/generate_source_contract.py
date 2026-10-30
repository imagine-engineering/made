"""Pinned Magnific execute/API oracle; shape-only resizing is not pixel parity."""
import argparse,ast,asyncio,copy,hashlib,json,math,uuid
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO
import numpy as np
import torch
from PIL import Image
ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
DEST=Path(__file__).with_name('source-contract.json')
FILES=['sim_api_nodes/nodes_magnific.py','sim_api_nodes/apis/magnific.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/client.py','sim_api_nodes/util/download_helpers.py']
NODES=['MagnificImageRelightNode','MagnificImageSkinEnhancerNode','MagnificImageStyleTransferNode','MagnificImageUpscalerCreativeNode','MagnificImageUpscalerPreciseV2Node']
torch.set_num_threads(1)
def selected(path,names,namespace,classes=False):
    nodes=[]
    for original in ast.parse((ROOT/path).read_text()).body:
        if getattr(original,'name',None) in names:
            node=copy.deepcopy(original)
            if classes:node.body=[member for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute']
            for member in ast.walk(node):
                if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    member.returns=None
                    for argument in [*member.args.args,*member.args.kwonlyargs]:argument.annotation=None
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(ROOT/path),'exec'),namespace)
class ImageProbe:
    def __init__(self,role,width=160,height=160,batch=1):self.role,self.shape=role,(batch,height,width,4)
    def __len__(self):return self.shape[0]
    def movedim(self,source,destination):
        value=ImageProbe(self.role);shape=list(self.shape);shape.insert(destination%len(shape),shape.pop(source));value.shape=tuple(shape);return value
async def generate():
    uploads=[];requests=[];polls=[];downloads=[];resizes=[]
    namespace={'__builtins__':__builtins__,'math':math,'torch':torch,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values)}
    exec(compile((ROOT/FILES[1]).read_text(),str(ROOT/FILES[1]),'exec'),namespace)
    for assignment in ast.parse((ROOT/FILES[0]).read_text()).body:
        if isinstance(assignment,ast.Assign) and any(getattr(target,'id',None)=='_EUR_TO_USD' for target in assignment.targets):namespace['_EUR_TO_USD']=ast.literal_eval(assignment.value)
    selected(FILES[0],{'_tier_price_eur','_calculate_magnific_upscale_price_usd'},namespace)
    selected(FILES[2],{'get_image_dimensions','get_number_of_images','validate_image_dimensions','validate_image_aspect_ratio','_assert_ratio_bounds','_ratio_from_tuple'},namespace)
    def upscale(image,width,height,method,crop):
        resizes.append({'role':image.role,'width':width,'height':height,'method':method,'crop':crop});value=ImageProbe(image.role,width,height);value.shape=(1,4,height,width);return value
    namespace['common_upscale']=upscale
    selected(FILES[3],{'downscale_image_tensor','_compute_downscale_dims'},namespace)
    async def upload(cls,image,**options):
        values=image if isinstance(image,list) else [image]
        result=[]
        for value in values:
            uploads.append({'role':value.role,'width':value.shape[2],'height':value.shape[1],'max_images':options.get('max_images',8),'total_pixels':options.get('total_pixels',2048*2048),'declared_content_type':options.get('mime_type'),'encoded_mime_type':'image/png'})
            result.append('https://storage.fixture.invalid/'+value.role+'.png')
        return result
    async def sync(cls,endpoint,response_model,data,**options):
        requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(exclude_none=True)})
        return response_model.model_validate({'data':{'task_id':'task-42','task_status':'queued'}})
    async def poll(cls,endpoint,response_model,**options):
        polls.append({'path':endpoint.path,'method':endpoint.method,'poll_interval':options.get('poll_interval',5.),'progress_price_usd':options['price_extractor'](None) if options.get('price_extractor') else None})
        return response_model.model_validate({'data':{'task_id':'task-42','status':'done','generated':['https://output.fixture.invalid/first.png','https://output.fixture.invalid/ignored.png']}})
    async def download(url):downloads.append({'url':url,'mode':'RGBA'});return {'url':url,'mode':'RGBA'}
    namespace.update(upload_images_to_simapi=upload,ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method),sync_op=sync,poll_op=poll,download_url_to_image_tensor=download)
    selected(FILES[0],set(NODES),namespace,True)
    def base(node):
        settings={
            NODES[0]:{'prompt':'  ☁  ','light_transfer_strength':100,'style':'standard','interpolate_from_original':False,'change_background':True,'preserve_details':True,'advanced_settings':{'advanced_settings':'disabled'}},
            NODES[1]:{'sharpen':0,'smart_grain':2,'mode':{'mode':'creative'}},
            NODES[2]:{'prompt':'  ☁  ','style_strength':100,'structure_strength':50,'flavor':'faithful','engine':'balanced','portrait_mode':{'portrait_mode':'disabled'},'fixed_generation':True},
            NODES[3]:{'prompt':'  ☁  ','scale_factor':'2x','optimized_for':'standard','creativity':0,'hdr':0,'resemblance':0,'fractality':0,'engine':'automatic','auto_downscale':False},
            NODES[4]:{'scale_factor':'2x','flavor':'sublime','sharpen':7,'smart_grain':7,'ultra_detail':30,'auto_downscale':False},
        }[node]
        value={'node':node,'settings':settings,'image':{'width':160,'height':160,'batch':1},'reference_image':None}
        if node==NODES[2]:value['reference_image']={'width':320,'height':160,'batch':1}
        return value
    def kwargs(value):
        output=copy.deepcopy(value['settings']);output['image']=ImageProbe('image',**value['image'])
        if value.get('reference_image') is not None:output['reference_image']=ImageProbe('reference_image',**value['reference_image'])
        return output
    def clear():
        for values in [uploads,requests,polls,downloads,resizes]:values.clear()
    cases=[];errors=[]
    async def observe(value):
        clear();output=await namespace[value['node']].execute(**kwargs(value))
        cases.append({'input':copy.deepcopy(value),'uploads':copy.deepcopy(uploads),'resizes':copy.deepcopy(resizes),'request':copy.deepcopy(requests[0]),'poll':copy.deepcopy(polls[0]),'downloads':copy.deepcopy(downloads),'output':output[0]})
    for node in NODES:await observe(base(node))
    schemas={};scalars={}
    for node in ast.parse((ROOT/FILES[0]).read_text()).body:
        if not isinstance(node,ast.ClassDef) or node.name not in NODES:continue
        schema=next(member for member in node.body if isinstance(member,ast.FunctionDef) and member.name=='define_schema');facts=[]
        for call in ast.walk(schema):
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or call.func.attr!='Input' or not call.args or not isinstance(call.args[0],ast.Constant):continue
            fact={'name':call.args[0].value,'type':ast.unparse(call.func.value)}
            for keyword in call.keywords:
                if keyword.arg in {'options','min','max','default'}:
                    try:fact[keyword.arg]=ast.literal_eval(keyword.value)
                    except (ValueError,TypeError):pass
            facts.append(fact)
            if fact.get('options') and fact['name'] in base(node.name)['settings']:
                for option in fact['options']:
                    value=base(node.name);value['settings'][fact['name']]=option;await observe(value)
            if fact.get('type')=='IO.Int' and fact['name'] in base(node.name)['settings']:
                for key in ['min','max']:
                    if key in fact:
                        value=base(node.name);value['settings'][fact['name']]=fact[key];await observe(value)
        schemas[node.name]=facts
    for node in [NODES[3],NODES[4]]:
        for width,height in [(160,160),(160,480),(480,160),(1300,1000),(3000,1000),(2000,2000),(4000,4000),(6000,3000),(10060,10060),(10061,10061),(25000,25000)]:
            for scale in ['2x','4x','8x','16x']:
                value=base(node);value['image'].update(width=width,height=height);value['settings'].update(scale_factor=scale,auto_downscale=True);await observe(value)
    for style in ['standard','pop','super_pop']:
        for beautifier in ['none','beautify_face','beautify_face_max']:
            value=base(NODES[2]);value['settings']['portrait_mode']={'portrait_mode':'enabled','portrait_style':style,'portrait_beautifier':beautifier};await observe(value)
    advanced={'advanced_settings':'enabled','whites':50,'blacks':50,'brightness':50,'contrast':50,'saturation':50,'engine':'automatic','transfer_light_a':'automatic','transfer_light_b':'automatic','fixed_generation':True}
    for fact in schemas[NODES[0]]:
        if fact['name'] not in advanced:continue
        for option in fact.get('options',[])+([fact['min'],fact['max']] if 'min' in fact else []):
            value=base(NODES[0]);value['reference_image']={'width':160,'height':480,'batch':1};value['settings']['advanced_settings']={**advanced,fact['name']:option};await observe(value)
    for mode in [{'mode':'faithful','skin_detail':0},{'mode':'faithful','skin_detail':100},*[{'mode':'flexible','optimized_for':option} for option in ['enhance_skin','improve_lighting','enhance_everything','transform_to_real','no_make_up']],{'mode':'unknown','optimized_for':'enhance_skin'}]:
        value=base(NODES[1]);value['settings']['mode']=mode;await observe(value)
    for node in [NODES[0],NODES[2],NODES[3]]:
        for prompt in ['', ' ', '☁'*10001]:value=base(node);value['settings']['prompt']=prompt;await observe(value)
    invalid=[]
    for node in NODES:
        for extra in [{'width':159},{'height':159},{'width':481,'height':160},{'batch':2}]:value=base(node);value['image'].update(extra);invalid.append(value)
    for node in [NODES[3],NODES[4]]:
        value=base(node);value['image'].update(width=6000,height=3000);value['settings']['scale_factor']='16x';invalid.append(value)
    value=base(NODES[2]);value['reference_image']['batch']=2;invalid.append(value)
    value=base(NODES[0]);value['reference_image']={'width':159,'height':160,'batch':1};invalid.append(value)
    for value in invalid:
        clear()
        try:await namespace[value['node']].execute(**kwargs(value))
        except Exception as error:errors.append({'input':value,'error_type':type(error).__name__,'requests':len(requests),'uploads':len(uploads)})
        else:raise ValueError('expected source preflight failure')
    prices=[{'width':pixels,'height':1,'scale':scale,'usd':namespace['_calculate_magnific_upscale_price_usd'](pixels,1,scale)} for pixels in [1299999,1300000,1300001,2999999,3000000,3000001,6399999,6400000,6400001] for scale in [2,4,8,16]]
    response_cases=[]
    for body in [{'task_id':'x','status':'DONE'},{'data':{'task_id':'x','task_status':'queued','generated':['first','second']}},{'data':'ignored','task_id':'x','status':'done'}, {'task_id':'x','status':'done','task_status':'failed'},{'task_id':'x','status':None,'task_status':'done'},{'status':'done'},{'task_id':'x','status':'done','generated':[1]}]:
        try:result={'model':namespace['TaskResponse'].model_validate(body).model_dump(exclude_none=True)}
        except Exception as error:result={'error_type':type(error).__name__}
        response_cases.append({'input':body,**result})
    # Actual tiny encoding and storage DTO preserve source None; no signed HTTP is invented.
    uploads.clear();requests.clear();storage=[];puts=[]
    namespace.update(np=np,Image=Image,BytesIO=BytesIO,uuid=SimpleNamespace(uuid4=lambda:'00000000-0000-4000-8000-000000000000'),mimetype_to_extension=lambda mime:mime.split('/')[-1].lower(),time=SimpleNamespace(monotonic=lambda:0.))
    selected(FILES[3],{'tensor_to_bytesio','tensor_to_pil','pil_to_bytesio','downscale_image_tensor','_compute_downscale_dims'},namespace)
    selected(FILES[4],{'UploadRequest','UploadResponse','upload_images_to_simapi','upload_file_to_simapi'},namespace)
    async def storage_sync(cls,endpoint,response_model,data,**options):storage.append(data.model_dump(exclude_none=True));return response_model.model_validate({'upload_url':'https://signed.fixture.invalid/input','download_url':'https://storage.fixture.invalid/input.png'})
    async def put(cls,url,stream,**options):puts.append({'url':url,'file_name':stream.name,'bytes_hex':stream.getvalue().hex(),'declared_content_type':options.get('content_type')})
    namespace.update(sync_op=storage_sync,upload_file=put)
    pixel=torch.empty((1,160,160,3),dtype=torch.float32);pixel[...,0]=.5;pixel[...,1]=.25;pixel[...,2]=.125
    output=await namespace['upload_images_to_simapi'](object,pixel,max_images=1)
    reference=pixel.repeat(1,1,2,1)
    output+=await namespace['upload_images_to_simapi'](object,reference,max_images=1)
    client_constants={};client_defaults={}
    for node in ast.parse((ROOT/FILES[5]).read_text()).body:
        if isinstance(node,ast.Assign) and any(getattr(target,'id',None) in {'COMPLETED_STATUSES','FAILED_STATUSES','QUEUED_STATUSES'} for target in node.targets):client_constants[node.targets[0].id]=ast.literal_eval(node.value)
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in {'sync_op','poll_op'}:
            values={}
            for argument,default in [*zip(node.args.args[-len(node.args.defaults):],node.args.defaults),*zip(node.args.kwonlyargs,node.args.kw_defaults)]:
                if default is not None:
                    try:values[argument.arg]=ast.literal_eval(default)
                    except (TypeError,ValueError):pass
            client_defaults[node.name]=values
    return {'client_constants':client_constants,'client_defaults':client_defaults,'sources':{file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in FILES},'versions':{'torch':torch.__version__,'pillow':Image.__version__},'cases':cases,'preflight_errors':errors,'schema_facts':schemas,'price_cases':prices,'response_cases':response_cases,'real_png_upload':{'storage_bodies':storage,'puts':puts,'urls':output},'scope':'actual source execute/API/preflight/price plus shape-only resize call plan; actual tiny Torch/Pillow PNG and storage DTO; native resize/codec/guest acceptance separate'}
parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
result=asyncio.run(generate());encoded=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if args.check:
    if DEST.read_text()!=encoded:raise SystemExit('source oracle drift')
else:DEST.write_text(encoded)
print(f"Magnific actual source: {len(result['cases'])} success cases, {len(result['preflight_errors'])} preflight failures,36 prices,7 response aliases,real PNG/None MIME storage DTO")
