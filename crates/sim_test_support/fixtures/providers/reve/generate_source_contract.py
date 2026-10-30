import argparse,ast,asyncio,base64,copy,hashlib,io,itertools,json,re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
FOLDER=Path(__file__).resolve().parent
SOURCE_ROOT=Path('/home/asvegah/repos/sim/projects/sim/ComfyUI')
SOURCES=['sim_api_nodes/nodes_reve.py','sim_api_nodes/apis/reve.py','sim_api_nodes/util/client.py','sim_api_nodes/util/validation_utils.py','sim_api_nodes/util/conversions.py']
class TensorProbe:
 def __init__(self,slot='image',count=1):self.slot=slot;self.shape=(count,720,720,3)
class TensorArrayProbe:
 def __init__(self,array):self.array=array
 def unsqueeze(self,axis):
  array=np.expand_dims(self.array,axis);return {'shape':list(array.shape),'dtype':str(array.dtype),'values':array.flatten().tolist(),'scope':'Actual source PIL/NumPy decoder; TensorArrayProbe replaces torch allocation'}
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
 requests=[];media=[];headers={};override={};contracts={};prices={}
 namespace={'__builtins__':__builtins__,'IO':SimpleNamespace(NodeOutput=lambda *values:values),'BytesIO':io.BytesIO,'np':np,'Image':Image,'torch':SimpleNamespace(from_numpy=TensorArrayProbe)}
 exec(compile((SOURCE_ROOT/SOURCES[1]).read_text(),'<actual-source-api>','exec'),namespace)
 conversions=ast.parse((SOURCE_ROOT/SOURCES[4]).read_text());conversion_functions=[copy.deepcopy(node) for node in conversions.body if isinstance(node,ast.FunctionDef) and node.name in ['bytesio_to_image_tensor','pil_to_bytesio']];exec(compile(strip(ast.Module(body=conversion_functions,type_ignores=[])),'<actual-source-PIL-NumPy-conversion>','exec'),namespace)
 image=Image.new('RGBA',(1,1),(128,64,32,96));buffer=io.BytesIO();image.save(buffer,format='WEBP',lossless=True);webp=buffer.getvalue()
 png=namespace['pil_to_bytesio'](Image.new('RGB',(1,1),(128,64,32))).getvalue();encoded=base64.b64encode(png).decode()
 def encode(image,**options):
  media.append({'operation':'encode_base64','image':serial(image),'mime_type':options.get('mime_type','image/png'),'total_pixels':options.get('total_pixels',4194304),'first_batch':True})
  if image.shape[0]==0:raise IndexError('source first image absent')
  return encoded
 async def sync(cls,endpoint,data=None,**options):
  requests.append({'path':endpoint.path,'method':endpoint.method,'headers':endpoint.headers,'body':data.model_dump(mode='json',exclude_none=True),'options':{key:(value.__name__ if callable(value) else value) for key,value in options.items()}})
  options['response_header_validator'](headers)
  return override.get('bytes',webp)
 namespace.update(tensor_to_base64_string=encode,sync_op_raw=sync,ApiEndpoint=lambda path,method='GET',headers=None:SimpleNamespace(path=path,method=method,headers=headers or {}))
 validation=ast.parse((SOURCE_ROOT/SOURCES[3]).read_text());exec(compile(strip(ast.Module(body=[copy.deepcopy(node) for node in validation.body if isinstance(node,ast.FunctionDef) and node.name=='validate_string'],type_ignores=[])),'<source-validation>','exec'),namespace)
 tree=ast.parse((SOURCE_ROOT/SOURCES[0]).read_text());registered=['ReveImageCreateNode','ReveImageEditNode','ReveImageRemixNode'];selected=[]
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and node.name in ['_build_postprocessing','_reve_price_extractor','_reve_response_header_validator']:selected.append(copy.deepcopy(node))
  elif isinstance(node,ast.ClassDef) and node.name in registered:
   method=next(copy.deepcopy(member) for member in node.body if isinstance(member,ast.AsyncFunctionDef) and member.name=='execute');schema=next(call for call in ast.walk(node) if isinstance(call,ast.Call) and ast.unparse(call.func)=='IO.Schema');fields={keyword.arg:keyword.value for keyword in schema.keywords};contracts[node.name]={'arguments':[argument.arg for argument in method.args.args if argument.arg!='cls'],'defaults':{},'schema':ast.unparse(schema),'hidden':ast.unparse(fields['hidden']),'outputs':ast.unparse(fields['outputs'])};prices[node.name]=ast.literal_eval(next(keyword.value for keyword in fields['price_badge'].keywords if keyword.arg=='expr'));selected.append(ast.ClassDef(name=node.name,bases=[],keywords=[],body=[method],decorator_list=[]))
 exec(compile(strip(ast.Module(body=selected,type_ignores=[])),'<actual-source-execute>','exec'),namespace)
 cases=[];errors=[];response_errors=[]
 def input_value(node,arguments):
  value=serial(arguments)
  if arguments.get('reference_images') is not None:value['reference_images']={'ordered_entries':[{'key':key,'image':serial(image)} for key,image in arguments['reference_images'].items()]}
  return {'node':node,'arguments':value}
 async def observe(node,arguments,failure=False,response_headers=None,response_bytes=None):
  requests.clear();media.clear();headers.clear();override.clear();headers.update(response_headers or {});override.update({'bytes':response_bytes} if response_bytes is not None else {})
  try:result=await namespace[node].execute(**copy.deepcopy(arguments))
  except Exception as error:
   if not failure:raise
   (response_errors if response_headers is not None or response_bytes is not None else errors).append({'input':input_value(node,arguments),'message':re.sub(r'0x[0-9a-fA-F]+','<address>',str(error)),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),**({'response_headers':response_headers,'response_bytes_hex':response_bytes.hex() if response_bytes is not None else webp.hex()} if response_headers is not None or response_bytes is not None else {})})
  else:
   if failure:raise ValueError('Expected source failure '+node)
   cases.append({'input':input_value(node,arguments),'requests':copy.deepcopy(requests),'media':copy.deepcopy(media),'image_probe':result[0],**({'response_headers':response_headers} if response_headers is not None else {})})
 bases={node:{'model':{'model':'reve-create@20250915' if node==registered[0] else 'reve-edit@20250915' if node==registered[1] else 'reve-remix@20250915','aspect_ratio':'1:1','test_time_scaling':1},'upscale':{'upscale':'disabled'},'remove_background':False,'seed':0,**({'edit_instruction':' source edit ','image':TensorProbe()} if node==registered[1] else {'prompt':' source prompt '}),**({'reference_images':{'image_9':TensorProbe('first'),'image_2':TensorProbe('second')}} if node==registered[2] else {})} for node in registered}
 for node,base in bases.items():
  for ratio,scaling,upscale,remove in itertools.product(['auto','1:1','custom'],[1,5,15],[{'upscale':'disabled'},{'upscale':'enabled','upscale_factor':2},{'upscale':'enabled','upscale_factor':3},{'upscale':'enabled','upscale_factor':4}],[False,True]):await observe(node,{**base,'model':{**base['model'],'aspect_ratio':ratio,'test_time_scaling':scaling},'upscale':upscale,'remove_background':remove,'seed':2147483647})
  for scaling in [0,-2,None,16]:await observe(node,{**base,'model':{**base['model'],'test_time_scaling':scaling}},node==registered[0] or scaling==16)
  for model in [{},{'model':'custom','aspect_ratio':'1:1'},{'model':'custom','test_time_scaling':1}]:await observe(node,{**base,'model':model},True)
  for factor in [1,5,None]:await observe(node,{**base,'upscale':{'upscale':'enabled','upscale_factor':factor}},factor is not None)
  for upscale in [{},{'upscale':'enabled'}]:await observe(node,{**base,'upscale':upscale},True)
  field='edit_instruction' if node==registered[1] else 'prompt'
  for prompt in ['',None,' ','α'*2560,'😀'*2561]:await observe(node,{**base,field:prompt},prompt is None or not prompt.strip() or len(prompt.strip())>2560)
  for response_headers in [{'x-reve-error-code':'CODE'},{'x-reve-content-violation':'TRUE'},{'x-reve-error-code':'CODE','x-reve-content-violation':'true'}]:await observe(node,base,True,response_headers)
  for response_headers in [{'x-reve-error-code':''},{'x-reve-content-violation':' true '},{'x-reve-content-violation':'false'}]:await observe(node,base,response_headers=response_headers)
  await observe(node,base,True,response_bytes=b'not-an-image')
 for count in [0,1,6,7]:await observe(registered[2],{**bases[registered[2]],'reference_images':{f'image_{9-index}':TensorProbe(f'ref-{index}') for index in range(count)}},count in [0,7])
 for node in registered[1:]:
  args={**bases[node]};args['image' if node==registered[1] else 'reference_images']=TensorProbe('empty',0) if node==registered[1] else {'image_1':TensorProbe('first'),'image_2':TensorProbe('empty',0)};await observe(node,args,True)
 header_prices=[]
 for credits in [None,'0','524.48','52.448','-2','invalid','','NaN','inf']:
  price_headers={} if credits is None else {'x-reve-credits-used':credits}
  try:price=namespace['_reve_price_extractor'](price_headers)
  except Exception as error:header_prices.append({'headers':price_headers,'suppressed_exception':type(error).__name__})
  else:header_prices.append({'headers':price_headers,'price_usd':repr(price) if price is not None else None})
 return {'schema_version':1,'namespace':'zed.sim.provider.reve','scope':'Actual source node AST/API/PIL/NumPy decoder; native image preparation and Torch allocation injected. No native ABI publication.','source_sha256':{path:hashlib.sha256((SOURCE_ROOT/path).read_bytes()).hexdigest() for path in SOURCES},'contracts':contracts,'cases':cases,'preflight_errors':errors,'response_errors':response_errors,'price_expressions':prices,'price_rates':{node:[float(value) for value in re.findall(r'"usd":\s*([0-9.]+)',price)] for node,price in prices.items()},'header_prices':header_prices,'encoded_image':{'png_hex':png.hex(),'base64':encoded,'webp_hex':webp.hex(),'producer':'Actual PIL PNG helper and PIL lossless RGBA WebP; input tensor preparation injected, output decoder NumPy probe substitutes Torch allocation'},'api_schemas':{name:namespace[name].model_json_schema() for name in ['ReveImageCreateRequest','ReveImageEditRequest','ReveImageRemixRequest']}}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT);parser.add_argument('--check',action='store_true');arguments=parser.parse_args();SOURCE_ROOT=arguments.source_root;encoded=(json.dumps(asyncio.run(generate()),indent=2,ensure_ascii=False)+'\n').encode();destination=FOLDER/'source-contract.json'
 if arguments.check:
  if destination.read_bytes()!=encoded:raise SystemExit('Reve source oracle differs')
 else:destination.write_bytes(encoded)
