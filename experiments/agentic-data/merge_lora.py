import torch, sys
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel
base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = Qwen3_5ForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
m = PeftModel.from_pretrained(m, adapter)
m = m.merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
try: AutoProcessor.from_pretrained(base).save_pretrained(out)
except Exception as e: print("processor save skipped:", e)
print("merged ->", out)
