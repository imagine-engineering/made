import argparse,ast,asyncio,copy,hashlib,io,itertools,json,re
from pathlib import Path
from types import SimpleNamespace
FOLDER=Path(__file__).resolve().parent
SOURCE_ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
SOURCES=['sim_api_nodes/nodes_quiver.py','sim_api_nodes/apis/quiver.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/upload_helpers.py','sim_api_nodes/util/conversions.py','sim_api/latest/_util/image_types.py']
class TensorProbe:
 def __init__(self,slot='image',count=1):self.slot=slot;self.shape=(count,720,720,3)
def serial(value):
 if isinstance(value,TensorProbe):return {'native_slot':value.slot,'shape':list(value.shape)}
 if isinstance(value,dict):return {key:serial(member) for key,member in value.items()}
 if isinstance(value,(list,tuple)):return [serial(member) for member in value]
 return value
def strip(node):
 for member in ast.walk(node):
  if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
   member.returns=None
   for argument in member.args.args+member.args.kwonlyargs:argument.annotation=None
 return ast.fix_missing_locations(node)
async def generate():
 namespace={'__builtins__':__builtins__,'IO':SimpleNamespace(NodeOutput=lambda *values:values),'BytesIO':io.BytesIO};exec(compile((SOURCE_ROOT/SOURCES[1]).read_text(),'<actual-source-api>','exec'),namespace);exec(compile((SOURCE_ROOT/SOURCES[6]).read_text(),'<actual-source-svg-wrapper>','exec'),namespace)
 requests=[];media=[];override={};contracts={};prices={};tree=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());registered=['QuiverTextToSVGNode','QuiverImageToSVGNode']
 async def upload(cls,image,**options):
  media.append({'operation':'upload_image','image':serial(image),'mime_type':options.get('mime_type'),'max_images':1,'wait_label':'Uploading','show_batch_index':False,'total_pixels':4194304})
  if image.shape[0]==0:raise IndexError('source first uploaded image absent')
  return f'https://storage.fixture.invalid/{image.slot}.png'
 async def sync(cls,endpoint,response_model,data=None,**options):
  requests.append({'path':endpoint.path,'method':endpoint.method,'body':data.model_dump(mode='json',exclude_none=True),'options':options});return response_model.model_validate(override.get('response',{'id':'source','data':[{'svg':'<svg xmlns="http://www.w3.org/2000/svg"><text>source α 😀</text></svg>'},{'svg':'','mime_type':'ignored/source'}],'usage':{'total_tokens':5}}))
 namespace.update(upload_image_to_simapi=upload,sync_op=sync,ApiEndpoint=lambda path,method='GET':SimpleNamespace(path=path,method=method))
 validation=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text());exec(compile(strip(ast.Module(body=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef) and node.name=='validate_string'],type_ignores=[])),'<source-validation>','exec'),namespace)
 selected=[]
 for node in tree.body:
  if isinstance(node,ast.Assign):selected.append(copy.deepcopy(node))
  elif isinstance(node,ast.ClassDef) and node.name in registered:
   method=next(copy.deepcopy(member) for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute');schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords};contracts[node.name]={'arguments':[argument.arg for argument in method.args.args if argument.arg!='cls'],'defaults':{argument.arg:ast.literal_eval(value) for argument,value in zip(method.args.args[-len(method.args.defaults):],method.args.defaults)} if method.args.defaults else {},'schema':ast.unparse(schema),'hidden':ast.unparse(fields['hidden']),'outputs':ast.unparse(fields['outputs'])};prices[node.name]=ast.literal_eval(next(keyword.value for keyword in fields['price_badge'].keywords if keyword.arg=='expr'));selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=[method],decorator_list=[]))
 exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-source-execute>','exec'),namespace)
 cases=[];errors=[];response_errors=[]
 def input_value(node,arguments):
  value=serial(arguments)
  if arguments.get('reference_images') is not None:value['reference_images']={'ordered_entries':[{'key':key,'image':serial(image)} for key,image in arguments['reference_images'].items()]}
  return {'node':node,'arguments':value}
 async def observe(node,arguments,failure=False,response=None):
  requests.clear();media.clear();override.clear();override.update({'response':response} if response is not None else {})
  try:result=await namespace[node].execute(**copy.deepcopy(arguments))
  except Exception as error:
   if not failure:raise
   (response_errors if response is not None else errors).append({'input':input_value(node,arguments),'message':str(error),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),**({'response':response} if response is not None else {})})
  else:
   if failure:raise ValueError('Expected source failure '+node)
   cases.append({'input':input_value(node,arguments),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'svg_utf8_hex':[buffer.getvalue().hex() for buffer in result[0].data],**({'response':response} if response is not None else {})})
 bases={'QuiverTextToSVGNode':{'prompt':'  source prompt  ','model':{'model':'arrow-1.1'},'seed':0},'QuiverImageToSVGNode':{'image':TensorProbe(),'auto_crop':False,'model':{'model':'arrow-1.1'},'seed':0}}
 for node,base in bases.items():
  for model in namespace['_ARROW_MODELS']+['custom','custom-max-preview','']:
   for options in [{},{'temperature':0,'top_p':0,'presence_penalty':-2},{'temperature':2,'top_p':1,'presence_penalty':2},{'temperature':1,'top_p':.05,'presence_penalty':0}]:await observe(node,{**base,'model':{'model':model,**options},'seed':2147483647})
  for field,value in [('temperature',-.1),('temperature',2.1),('top_p',-.1),('top_p',1.1),('presence_penalty',-2.1),('presence_penalty',2.1)]:await observe(node,{**base,'model':{'model':'arrow-1.1',field:value}},True)
  await observe(node,{**base,'model':{}},True)
  for response in [{'data':[]},{'data':[{'svg':''}]},{'data':[{'svg':'not xml','mime_type':'ignored/source'}]},{'data':[{'svg':'<svg>α😀\u0000</svg>'},{'svg':'<svg>second</svg>'}]}]:await observe(node,base,response=response)
  for response in [{},{'data':None},{'data':[{}]},{'data':[{'svg':None}]},{'data':[{'svg':42}]},{'data':[{'svg':'source','mime_type':42}]},{'data':[{'svg':'source'}],'usage':{'total_tokens':'invalid'}}]:await observe(node,base,True,response)
 for prompt in ['',None,' ','\u001c\u001f','x'*10000]:await observe('QuiverTextToSVGNode',{**bases['QuiverTextToSVGNode'],'prompt':prompt},prompt in ['',None])
 for instructions in [None,'',' ','\u001c\u001f\u0085\u00a0source\u001c\u001f','\u200bsource\u200b',' source guidance ']:await observe('QuiverTextToSVGNode',{**bases['QuiverTextToSVGNode'],'instructions':instructions})
 for count in range(6):await observe('QuiverTextToSVGNode',{**bases['QuiverTextToSVGNode'],'reference_images':{f'ref_{9-index}':TensorProbe(f'ref-{index}') for index in range(count)}},count>4)
 await observe('QuiverTextToSVGNode',{**bases['QuiverTextToSVGNode'],'reference_images':{'ref_2':TensorProbe('first'),'ref_1':TensorProbe('empty',0)}},True)
 await observe('QuiverImageToSVGNode',{**bases['QuiverImageToSVGNode'],'image':TensorProbe('empty',0)},True)
 for crop,target in itertools.product([False,True],[None,128,1024,4096]):await observe('QuiverImageToSVGNode',{**bases['QuiverImageToSVGNode'],'auto_crop':crop,'model':{'model':'arrow-1.1','target_size':target}})
 for target in [127,4097]:await observe('QuiverImageToSVGNode',{**bases['QuiverImageToSVGNode'],'model':{'model':'arrow-1.1','target_size':target}},True)
 price_values=[float(value) for value in re.findall(r'"usd":\s*([0-9.]+)',prices['QuiverTextToSVGNode'])];assert len(price_values)==3;rates={'max':price_values[0],'preview':price_values[1],'base':price_values[2]};quotes=[]
 for model in namespace['_ARROW_MODELS']+['custom','custom-max-preview','']:
  quote=rates['max'] if 'max' in model else rates['preview'] if 'preview' in model else rates['base'];quotes.append({'model':model,'maximum_microunits':round(quote*1000000)})
 return {'schema_version':1,'namespace':'zed.sim.provider.quiver','scope':'Pinned source node AST/Pydantic and actual SVG ordered BytesIO wrapper; native upload/artifact allocation injected, no native ABI result. Reference mapping insertion order uses ordered_entries boundary.','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'contracts':contracts,'models':namespace['_ARROW_MODELS'],'cases':cases,'preflight_errors':errors,'response_errors':response_errors,'price_expressions':prices,'price_rates':rates,'price_cases':quotes,'api_schemas':{name:namespace[name].model_json_schema() for name in ['QuiverTextToSVGRequest','QuiverImageToSVGRequest','QuiverSVGResponse']}}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');arguments=parser.parse_args();SOURCE_ROOT=arguments.source_root;encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode();destination=FOLDER/'source-contract.json'
 if arguments.check:
  if destination.read_bytes()!=encoded:raise SystemExit('Quiver source oracle differs')
 else:destination.write_bytes(encoded)
