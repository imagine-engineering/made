"""Execute pinned Bria node request AST with explicit media boundary probes."""
import argparse
import ast
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE_ROOT = Path(__file__).resolve().parents[5] / 'projects/sim/ComfyUI'
DESTINATION = Path(__file__).with_name('source-contract.json')
SOURCES = ['sim_api_nodes/nodes_bria.py','sim_api_nodes/apis/bria.py','sim_api_nodes/util/client.py','sim_api_nodes/util/_helpers.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/download_helpers.py','sim_api_nodes/util/conversions.py']

class MediaProbe:
    def __init__(self,name,kind='image',duration=2.5,channels=4):
        self.name,self.kind,self.duration,self.channels=name,kind,duration,channels
    def __getitem__(self,key):
        return MediaProbe(self.name,self.kind,self.duration,3)
    def get_duration(self):
        return self.duration

class RequestProbe:
    defaults={}
    def __init__(self,**fields):
        self.fields={**self.defaults,**fields}
    def model_dump(self,exclude_none=True):
        return {key:value for key,value in self.fields.items() if value is not None}

def field(value=None,default=None,**unused):
    return default if default is not None else value

def strip_annotations(node):
    for child in ast.walk(node):
        if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)):
            child.returns=None
            for argument in [*child.args.posonlyargs,*child.args.args,*child.args.kwonlyargs]: argument.annotation=None
    return ast.fix_missing_locations(node)

async def generate():
    requests=[];uploads=[];polls=[];decodes=[]
    async def upload(cls,media,wait_label='Uploading'):
        uploads.append({'role':media.name,'kind':media.kind,'channels':media.channels,'wait_label':wait_label})
        return 'https://storage.fixture.invalid/'+media.name
    async def sync_op(cls,endpoint,data,response_model):
        requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(exclude_none=True)})
        return SimpleNamespace(request_id='source-request-42',status_url='https://ignored-status.fixture.invalid/task',warning='source warning')
    async def poll_op(cls,endpoint,**options):
        polls.append({'path':endpoint.path,'method':endpoint.method})
        return SimpleNamespace(status='completed',result=SimpleNamespace(image_url='https://download.fixture.invalid/image.png',video_url='https://download.fixture.invalid/video.webm',structured_prompt='{"source":"structured"}'))
    async def download(url):
        return url
    def decode_alpha(video):
        decodes.append('libvpx-vp9 RGB plus inverted alpha MASK owner required')
        return 'image-frames-probe','inverted-alpha-mask-probe'
    namespace={'__builtins__':__builtins__,'classmethod':classmethod,'Field':field,'BaseModel':RequestProbe,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values),
               'ApiEndpoint':lambda path,method='GET':SimpleNamespace(path=path,method=method),
               'sync_op':sync_op,'poll_op':poll_op,'upload_image_to_simapi':upload,'upload_video_to_simapi':upload,
               'convert_mask_to_image':lambda mask:mask,'download_url_to_image_tensor':download,'download_url_to_video_output':download,
               '_video_to_images_and_mask':decode_alpha,'logging':SimpleNamespace(error=lambda *args:None)}
    validation=ast.parse((SOURCE_ROOT/SOURCES[5]).read_text())
    functions=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef) and node.name=='validate_video_duration']
    exec(compile(strip_annotations(ast.Module(body=functions,type_ignores=[])),'<pinned-bria-validation>','exec'),namespace)
    for node in ast.parse((SOURCE_ROOT/SOURCES[1]).read_text()).body:
        if isinstance(node,ast.ClassDef) and node.name.endswith('Request'):
            defaults={}
            for member in node.body:
                if isinstance(member,ast.AnnAssign) and isinstance(member.target,ast.Name): defaults[member.target.id]=eval(compile(ast.Expression(member.value),'<pinned-request-default>','eval'),namespace)
            namespace[node.name]=type(node.name,(RequestProbe,),{'defaults':defaults})
        elif isinstance(node,ast.ClassDef): namespace[node.name]=object
    selected=[]
    for node in ast.parse((SOURCE_ROOT/SOURCES[0]).read_text()).body:
        if isinstance(node,ast.ClassDef) and node.name.startswith('Bria') and node.name!='BriaExtension':
            node=copy.deepcopy(node);node.body=[member for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute'];selected.append(node)
    exec(compile(strip_annotations(ast.Module(body=selected,type_ignores=[])),'<pinned-bria-execute>','exec'),namespace)
    cases=[]
    def candidate(node,**extra):
        return {'node':node,'model':'FIBO','prompt':'  change cloud ☁  ','negative_prompt':'grain','structured_prompt':'','seed':42,'guidance_scale':3.75,'steps':50,'moderation':{'prompt_content_moderation':True,'visual_input_moderation':False,'visual_output_moderation':True},'mask':False,'background_color':'Blue','green_shade':'chroma_green','background_image':False,'background_video':False,'duration_seconds':2.5,'background_duration_seconds':2.5,**extra}
    async def observe(input,kwargs):
        requests.clear();uploads.clear();polls.clear();decodes.clear()
        result=await namespace[input['node']].execute(**kwargs)
        cases.append({'input':input,'request':copy.deepcopy(requests[0]),'uploads':copy.deepcopy(uploads),'poll':copy.deepcopy(polls[0]),'alpha_decoder':copy.deepcopy(decodes),'outputs':list(result)})
    image=MediaProbe('image');mask=MediaProbe('mask',channels=3);video=MediaProbe('video','video');background_image=MediaProbe('background_image');background_video=MediaProbe('background_video','video')
    for with_mask in [False,True]:
        value=candidate('BriaImageEditNode',mask=with_mask)
        await observe(value,{'model':'FIBO','image':image,'prompt':value['prompt'],'negative_prompt':'grain','structured_prompt':'','seed':42,'guidance_scale':3.75,'steps':50,'moderation':value['moderation'],'mask':mask if with_mask else None})
    value=candidate('BriaImageEditNode',prompt='',negative_prompt='',structured_prompt=' {"edit":"sky"} ',moderation={})
    await observe(value,{'model':'FIBO','image':image,'prompt':'','negative_prompt':'','structured_prompt':value['structured_prompt'],'seed':42,'guidance_scale':3.75,'steps':50,'moderation':{},'mask':None})
    value=candidate('BriaRemoveImageBackground');await observe(value,{'image':image,'moderation':value['moderation'],'seed':42})
    value=candidate('BriaRemoveVideoBackground');await observe(value,{'video':video,'background_color':'Blue','seed':42})
    value=candidate('BriaVideoGreenScreen');await observe(value,{'video':video,'green_shade':'chroma_green','seed':42})
    for is_video in [False,True]:
        value=candidate('BriaVideoReplaceBackground',background_image=not is_video,background_video=is_video)
        await observe(value,{'video':video,'seed':42,'background_image':None if is_video else background_image,'background_video':background_video if is_video else None})
    value=candidate('BriaTransparentVideoBackground');await observe(value,{'video':video,'seed':42})
    errors=[]
    async def observe_error(value,kwargs):
        requests.clear();uploads.clear()
        try:
            await namespace[value['node']].execute(**kwargs)
        except ValueError as error:
            errors.append({'input':value,'message':str(error),'requests':len(requests),'uploads':len(uploads)})
        else:
            raise ValueError('Expected pinned source preflight error')
    value=candidate('BriaImageEditNode',prompt='',structured_prompt='')
    await observe_error(value,{'model':'FIBO','image':image,'prompt':'','negative_prompt':'','structured_prompt':'','seed':42,'guidance_scale':3.75,'steps':50,'moderation':{},'mask':None})
    for both in [False,True]:
        value=candidate('BriaVideoReplaceBackground',background_image=both,background_video=both)
        await observe_error(value,{'video':video,'seed':42,'background_image':background_image if both else None,'background_video':background_video if both else None})
    value=candidate('BriaVideoGreenScreen',duration_seconds=61.0)
    await observe_error(value,{'video':MediaProbe('video','video',duration=61.0),'green_shade':'chroma_green','seed':42})
    return {'schema_version':1,'namespace':'zed.sim.provider.bria','scope':'Actual pinned node.execute AST, request defaults and duration validation; image/mask conversion, storage upload, transport, RGBA/VIDEO decoding and VP9 alpha frames remain explicit unexecuted canonical media boundaries.','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'cases':cases,'preflight_errors':errors}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);args=parser.parse_args();SOURCE_ROOT=args.source_root
    encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode()
    if args.check:
        if DESTINATION.read_bytes()!=encoded:raise SystemExit('Bria source contract differs')
    else:DESTINATION.write_bytes(encoded)
