import json
from pathlib import Path
import shlex
import subprocess

build = Path('/workspace/dspark-attention-build')
source = Path('/workspace/preparation/kernels-bf16')
rows = json.loads((build/'compile_commands.json').read_text())
objects=[]
for row in rows:
    path=Path(row['file'])
    if path.parent == source/'csrc/flash_attn' or path == source/'csrc/xpu/attn/attn_interface.cpp':
        args=shlex.split(row['command'])
        args=[arg for arg in args if not arg.startswith('-DTORCH_EXTENSION_NAME=')]
        args.insert(1,'-DTORCH_EXTENSION_NAME=_minicpm_dspark_fa2')
        print('COMPILE',path.name,flush=True)
        subprocess.run(args,cwd=build,check=True)
        objects.append(str(build/row['output']))
library=build/'_minicpm_dspark_fa2.abi3.so'
torchlib='/opt/venv/lib/python3.12/site-packages/torch/lib'

def link_command(target):
    line=subprocess.check_output(['/opt/venv/bin/ninja','-C',str(build),'-t','commands',target],text=True).splitlines()[-1]
    tokens=shlex.split(line)
    start=next(i for i,value in enumerate(tokens) if value.endswith('/icpx'))
    end=tokens.index('&&',start) if '&&' in tokens[start:] else len(tokens)
    return tokens[start:end]

native_library='libminicpm_dspark_attn_kernels.so'
args=link_command('attn_kernels_xe_2')
args[args.index('-o')+1]=native_library
args=[('-Wl,-soname,'+native_library) if arg.startswith('-Wl,-soname,') else arg for arg in args]
args.insert(1,'-Wl,-Bsymbolic')
subprocess.run(args,cwd=build,check=True)
args=link_command('_vllm_fa2_C')
args[args.index('-o')+1]=str(library)
args=[native_library if arg=='libattn_kernels_xe_2.so' else arg for arg in args]
args=[('-Wl,-rpath,$ORIGIN:'+torchlib) if arg.startswith('-Wl,-rpath,') else arg for arg in args]
args.insert(1,'-Wl,-Bsymbolic')
print('LINK',library,flush=True)
subprocess.run(args,cwd=build,check=True)
print('DONE',flush=True)
