"""Execute pinned Krea AST and actual Pydantic/Pillow/aiohttp owners independently."""
import argparse,ast,asyncio,copy,hashlib,json,re,uuid
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO
import numpy as np
import torch
from PIL import Image
import aiohttp
ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
DEST=Path(__file__).with_name('source-contract.json')
FILES=['sim_api_nodes/nodes_krea.py','sim_api_nodes/apis/krea.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/conversions.py','sim_api_nodes/util/client.py','sim_api_nodes/util/download_helpers.py']
torch.set_num_threads(1)
def select(path,names,namespace,classes=False):
    nodes=[]
    for original in ast.parse((ROOT/path).read_text()).body:
        if getattr(original,'name',None) in names:
            node=copy.deepcopy(original)
            if classes:node.body=[member for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute']
            for member in ast.walk(node):
                if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    member.returns=None
                    for arg in [*member.args.args,*member.args.kwonlyargs]:arg.annotation=None
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(ROOT/path),'exec'),namespace)
async def generate():
    calls=[];polls=[];downloads=[];uploads=[]
    namespace={'__builtins__':__builtins__,'re':re,'IO':SimpleNamespace(ComfyNode=object,NodeOutput=lambda *values:values)}
    exec(compile((ROOT/FILES[1]).read_text(),str(ROOT/FILES[1]),'exec'),namespace)
    for node in ast.parse((ROOT/FILES[0]).read_text()).body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/FILES[0]),'exec'),namespace)
    select(FILES[2],{'validate_string'},namespace)
    async def sync(cls,endpoint,response_model,data=None,files=None,**options):
        if files:
            field,(name,stream,mime)=files[0];uploads.append({'path':endpoint.path,'method':endpoint.method,'field':field,'file_name':name,'mime':mime,'bytes_hex':stream.getvalue().hex(),'options':options})
            return response_model.model_validate({'id':'asset-42','image_url':'https://asset.fixture.invalid/style.png','uploaded_at':'source-time'})
        calls.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(exclude_none=True)})
        return response_model.model_validate({'job_id':'job-42','status':'queued','created_at':'source-time'})
    async def poll(cls,endpoint,response_model,**options):
        polls.append({'path':endpoint.path,'method':endpoint.method,'queued_statuses':options['queued_statuses']})
        return response_model.model_validate({'job_id':'job-42','status':'completed','created_at':'source-time','result':{'urls':['https://output.fixture.invalid/first.png','https://output.fixture.invalid/ignored.png']}})
    async def download(url):downloads.append(url);return {'mode':'RGBA','url':url}
    namespace.update(ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method),sync_op=sync,poll_op=poll,download_url_to_image_tensor=download)
    conversion={'__builtins__':__builtins__,'np':np,'torch':torch,'Image':Image,'BytesIO':BytesIO,'uuid':SimpleNamespace(uuid4=lambda:'00000000-0000-4000-8000-000000000000'),'mimetype_to_extension':lambda mime:mime.split('/')[-1].lower()}
    select(FILES[3],{'tensor_to_bytesio','tensor_to_pil','pil_to_bytesio','downscale_image_tensor','_compute_downscale_dims'},conversion)
    namespace['tensor_to_bytesio']=conversion['tensor_to_bytesio']
    select(FILES[0],{'_upload_image_to_krea_assets'},namespace)
    select(FILES[0],{'Krea2ImageNode','Krea2StyleReferenceNode'},namespace,True)
    cases=[];errors=[]
    def base():return {'prompt':'  source ☁  ','model':{'model':'Krea 2 Medium','aspect_ratio':'1:1','resolution':'1K','creativity':'medium'},'seed':2147483647}
    async def observe(value):
        calls.clear();polls.clear();downloads.clear()
        output=await namespace['Krea2ImageNode'].execute(**value)
        cases.append({'input':copy.deepcopy(value),'request':copy.deepcopy(calls[0]),'poll':copy.deepcopy(polls[0]),'downloads':copy.deepcopy(downloads),'output':output[0]})
    for model in namespace['_MODEL_ENDPOINTS']:
        for ratio in namespace['_ASPECT_RATIOS']:
            for creativity in namespace['_CREATIVITY_LEVELS']:
                value=base();value['model'].update(model=model,aspect_ratio=ratio,creativity=creativity);await observe(value)
    for extra in [{'prompt':' '},{'prompt':'☁'*1025},{'model':{'moodboard_id':' \u001cABCDEF00-1234-5678-9ABC-DEF012345678\u001c ','moodboard_strength':None}},{'model':{'moodboard_id':'00000000-0000-0000-0000-000000000000','moodboard_strength':'-0.5'}},{'model':{'style_reference':[{'url':'https://style.fixture.invalid/a','strength':'-2','ignored':'retained only by chain'},{'url':None,'strength':2}]}},{'model':{'style_reference':[]}}, {'model':{'style_reference':[{'url':'https://style.fixture.invalid/a','strength':1}]*10}}]:
        value=base();value['model'].update(extra.get('model',{}));value.update({key:val for key,val in extra.items() if key!='model'});await observe(value)
    for extra in [{'prompt':''},{'model':{'model':'unknown'}},{'model':{'moodboard_id':'bad'}},{'model':{'moodboard_id':'00000000-0000-0000-0000-000000000000','moodboard_strength':1.6}},{'model':{'style_reference':[{'url':'x','strength':1}]*11}},{'model':{'style_reference':[{'url':'x','strength':2.1}]}},{'model':{'style_reference':[{'strength':1}]}}]:
        value=base();value['model'].update(extra.get('model',{}));value.update({key:val for key,val in extra.items() if key!='model'});calls.clear()
        try:await namespace['Krea2ImageNode'].execute(**value)
        except Exception as error:errors.append({'input':value,'error_type':type(error).__name__,'requests':len(calls)})
        else:raise ValueError('expected source preflight error')
    image=torch.tensor([[[[.5,.25,.125],[1.,0.,.75]],[[.1,.2,.3],[.99,.01,.7]]]],dtype=torch.float32)
    chain_cases=[]
    for count,strength in [(0,1),(9,-2)]:
        uploads.clear();chain=[{'url':'https://prior.fixture.invalid/'+str(i),'strength':.5,'extra':'preserved'} for i in range(count)];previous=copy.deepcopy(chain)
        output=await namespace['Krea2StyleReferenceNode'].execute(image,strength,chain)
        assert chain==previous
        chain_cases.append({'incoming':chain,'strength':strength,'output':output[0],'upload':copy.deepcopy(uploads[0])})
    uploads.clear()
    try:await namespace['Krea2StyleReferenceNode'].execute(image,1,[{}]*10)
    except ValueError:pass
    else:raise ValueError('expected source chain failure')
    assert not uploads
    encoded=bytes.fromhex(chain_cases[0]['upload']['bytes_hex']);name=chain_cases[0]['upload']['file_name']
    form=aiohttp.FormData();form.add_field('file',BytesIO(encoded),filename=name,content_type='image/png');payload=form();payload._boundary=b'krea-source-boundary'
    class Writer:
        def __init__(self):self.data=bytearray()
        async def write(self,data):self.data.extend(data)
    writer=Writer();await payload.write(writer)
    client=ast.parse((ROOT/FILES[4]).read_text());constants={}
    for node in client.body:
        if isinstance(node,ast.Assign) and any(getattr(target,'id',None) in {'COMPLETED_STATUSES','FAILED_STATUSES'} for target in node.targets):constants[node.targets[0].id]=ast.literal_eval(node.value)
    defaults={}
    for file,names in [(FILES[4],{'sync_op','poll_op'}),(FILES[5],{'download_url_to_bytesio'})]:
        for node in ast.parse((ROOT/file).read_text()).body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
                values={}
                for argument,default in [*zip(node.args.args[-len(node.args.defaults):],node.args.defaults),*zip(node.args.kwonlyargs,node.args.kw_defaults)]:
                    if default is not None:
                        try:values[argument.arg]=ast.literal_eval(default)
                        except (ValueError,TypeError):pass
                defaults[node.name]=values
    return {'client_defaults':defaults,'sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES},'versions':{'pillow':Image.__version__,'torch':torch.__version__,'aiohttp':aiohttp.__version__},'cases':cases,'errors':errors,'chain_cases':chain_cases,'chain_limit_before_upload':True,'multipart':{'boundary':'krea-source-boundary','bytes_hex':bytes(writer.data).hex()},'status_constants':constants,'price_badge_gap':'Title-case schema model options; lower-case price lookup keys; widget normalization unproven','scope':'actual execute/API AST and actual Pillow PNG/aiohttp; transport/native owner gates separate'}
args=argparse.ArgumentParser();args.add_argument('--check',action='store_true');args=args.parse_args()
result=json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n'
if args.check:
    if DEST.read_text()!=result:raise SystemExit('source oracle drift')
else:DEST.write_text(result)
print('Krea pinned source: 103 generation cases, seven preflight failures, two real PNG style chains, actual aiohttp multipart')
