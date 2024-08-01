from itertools import zip_longest
import json
import time
import os.path as osp
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
    LlamaTokenizer, LlamaForCausalLM
from peft import (
    PeftModel,
     PeftModelForCausalLM
)

def read_txt_to_list_of_dict(fname: str):
    # fname should be a .jsonl
    results = []
    for l in open(fname).readlines():
        d = json.loads(l)
        results.append(d)
    return results


def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') -->
    ('a','b','c'), ('d','e','f'), ('g','x','x')"""
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def file_exists(wfname, overwrite):
    if osp.exists(wfname):
        if overwrite:
            print(f"Warning: {wfname} exists and will overwrite.")
            print("Staring in 5 seconds.")
            time.sleep(5)
            return False
        else:
            print(f"Error: {wfname} exists!")
            return True
    else:
        return False


def write_list_of_dict_to_jsonl(fwname: str, l_dict, overwrite=False):
    if file_exists(fwname, overwrite=overwrite):
        return

    fw = open(fwname, "w")
    for d in l_dict:
        fw.write(json.dumps(d, ensure_ascii=False) + "\n")
    fw.close()

    return


def genereate_wrapper(input_ids, model, generation_config, a_position_to_mask=None):
    attention_mask = torch.ones_like(input_ids)
    if a_position_to_mask is not None:
        # assert a_position_to_mask < input_ids.shape[-1]

        # set to 0: <unk>
        input_ids[0, a_position_to_mask] = 0
        attention_mask[0, a_position_to_mask] = 0
    # print('generate_wrapper:', input_ids)
    # print('generate_wrapper:', attention_mask)
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config
    )

    # generate_ids[0] is the start token <s>, automatically added by hf.
    # generate_ids[input_token_num] is the first token that is generated.

    # check text.
    # tokenizer.batch_decode(generated_ids)
    return generated_ids


def get_template(model_name):
    # vicuna
    # system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    # input_text = f"{system} USER:{query} ASSISTANT:"
    vicuna_template = {
        'prefix': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. User:",
        'postfix': " ASSISTANT:"
    }
    # llama2-chat
    # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L358C19-L358C70
    llama_template = {
        'prefix': "[INST] ",
        'postfix': " [/INST]"
    }

    template = {
        'llama2-7b-chat': llama_template,
        'vicuna1.5-7b': vicuna_template,
        # TO ADD.
    }[model_name]

    return template


def load_model(model_name):
    if model_name == 'llama-7b':
        model_path = '/root/codespace/ongoing_projects/baselines/model_weights/llama-hf/llama-7b'
        device_map = "auto"  # model parallel
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(model.hf_device_map)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        block_name = "self.model.model.layers"
        embedding_name = "self.model.model.embedding"
        embedding_token_name = "self.model.model.embed_tokens.weight"
        vocab_size = model.model.vocab_size
        embed_dim = 4096

    elif model_name == 'alpaca-lora-7b':
        model_path = '/root/codespace/cjm_code/pretrained_model/llama-7b'
        device_map = "auto"  # model parallel
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(model.hf_device_map)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)

        lora_weights = "/root/codespace/cjm_code/pretrained_model/alpaca-lora-7b"
        model = PeftModelForCausalLM.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float32,
        )
        print("PeftModel loaded.")
        block_name = "self.model.base_model.model.model.layers"
        embedding_name = "self.model.base_model.model.model.embedding"
        embedding_token_name = "self.model.base_model.model.model.embed_tokens.weight"
        vocab_size = model.base_model.model.model.vocab_size
        embed_dim = 4096

    elif model_name == 'vicuna-7b':
        model_path = '/root/model_weights/vicuna-7b'
        device_map = "auto"  # model parallel
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(model.hf_device_map)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        block_name = "self.model.model.layers"
        embedding_name = "self.model.model.embedding"
        embedding_token_name = "self.model.model.embed_tokens.weight"
        vocab_size = model.model.vocab_size
        embed_dim = 4096

    elif model_name == 'stanford-alpaca-7b':
        model_path = '/root/codespace/cjm_code/pretrained_model/stanford-alpaca-7b'
        device_map = "auto"  # model parallel
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(model.hf_device_map)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        block_name = "self.model.model.layers"
        embedding_name = "self.model.model.embedding"
        embedding_token_name = "self.model.model.embed_tokens.weight"
        vocab_size = model.model.vocab_size
        embed_dim = 4096

    elif model_name == 'llama-13b':
        model_path = '/root/codespace/cjm_code/pretrained_model/llama-13b'
        device_map = "auto"  # model parallel
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(model.hf_device_map)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        block_name = "self.model.model.layers"
        embedding_name = "self.model.model.embedding"
        embedding_token_name = "self.model.model.embed_tokens.weight"
        vocab_size = model.model.vocab_size
        embed_dim = 5120

    elif model_name == 'alpaca-lora-13b':
        model_path = '/root/codespace/cjm_code/pretrained_model/llama-13b'
        device_map = "auto"  # model parallel
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(model.hf_device_map)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)

        lora_weights = "/root/codespace/cjm_code/pretrained_model/alpaca-lora-13b"
        model = PeftModelForCausalLM.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float32,
        )
        print("PeftModel loaded.")
        block_name = "self.model.base_model.model.model.layers"
        embedding_name = "self.model.base_model.model.model.embedding"
        embedding_token_name = "self.model.base_model.model.model.embed_tokens.weight"
        vocab_size = model.base_model.model.model.vocab_size
        embed_dim = 5120

    elif model_name == 'vicuna-13b':
        model_path = '/root/codespace/cjm_code/pretrained_model/vicuna-13b'
        device_map = "auto"  # model parallel
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print(model.hf_device_map)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        block_name = "self.model.model.layers"
        embedding_name = "self.model.model.embedding"
        embedding_token_name = "self.model.model.embed_tokens.weight"
        vocab_size = model.model.vocab_size
        embed_dim = 5120

    return model, tokenizer, block_name, embedding_name, embedding_token_name, vocab_size, embed_dim