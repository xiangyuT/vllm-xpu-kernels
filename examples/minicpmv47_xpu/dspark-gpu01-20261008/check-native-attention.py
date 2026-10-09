import ast
import json
from pathlib import Path
import sys
from typing import Optional

import torch
import dspark_attention
import vllm_xpu_kernels.flash_attn_interface as attention
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

source=Path('/workspace/preparation/kernels-bf16/tests/flash_attn/test_flash_attn_varlen_func.py')
tree=ast.parse(source.read_text())
nodes=[]
for name in ['ref_paged_attn','test_varlen_with_paged_kv']:
    node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
    node.decorator_list=[]
    nodes.append(node)
namespace={'torch':torch,'Optional':Optional,
           'flash_attn_varlen_func':flash_attn_varlen_func}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),namespace)
results=[]
block_size=int(sys.argv[1]) if len(sys.argv)>1 else 64
for dtype in [torch.bfloat16,torch.float16]:
    for query_len,kv_len in [(15,32),(15,512),(15,2048),(15,8192),(7,512),(16,512)]:
        before=dspark_attention.native_success_calls
        warnings_before=set(attention._warned_missing_configs)
        with torch.inference_mode():
            namespace['test_varlen_with_paged_kv'](
                seq_lens=[(query_len,kv_len)],num_heads=(4,1),head_size=256,
                window_size=(-1,-1),dtype=dtype,block_size=block_size,soft_cap=None,
                num_blocks=256,fa_version=2,q_dtype=None,is_sink=False,
                is_casual=False,is_paged=True,fp8_dtype=None,stride_pad=0)
        assert dspark_attention.native_success_calls == before+1
        assert attention._warned_missing_configs == warnings_before
        result={'dtype':str(dtype),'query_len':query_len,'kv_len':kv_len,'block_size':block_size,'passed':True}
        results.append(result)
        print('PASS',result,flush=True)
output='native-attention-results.json' if block_size==64 else f'native-attention-block{block_size}-results.json'
Path('/workspace/evidence',output).write_text(json.dumps(results,indent=2)+'\n')
print('COMPLETE',len(results),'native kernel comparisons with the existing paged-attention reference',flush=True)
