from interpretor import Interpretor
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, set_seed, tokenization_utils_base
from peft import (
    PeftModel,
     PeftModelForCausalLM
)
import json
from tqdm import tqdm
import argparse
import os
import numpy as np

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
        model =  PeftModelForCausalLM.from_pretrained(
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
        
    elif model_name == 'vicuna-7b-1':
        model_path = '/root/model_weights/output_2048/checkpoint-350'
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
    
    elif model_name == 'vicuna-7b-2':
        model_path = '/root/model_weights/output_2048/checkpoint-700'
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
    
    elif model_name == 'vicuna-7b-3':
        model_path = '/root/model_weights/output_2048/checkpoint-1050'
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
    
    elif model_name == 'vicuna-7b-4':
        model_path = '/root/model_weights/output_2048/checkpoint-1400'
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
    
    elif model_name == 'vicuna-7b-5':
        model_path = '/root/model_weights/output_2048/checkpoint-1750'
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
    
    elif model_name == 'vicuna-7b-6':
        model_path = '/root/model_weights/output_2048/checkpoint-2100'
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
        model =  PeftModelForCausalLM.from_pretrained(
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
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate explanantion results')
    
    parser.add_argument('--method', type=str,
                        default='bt',
                        choices=['lime', 'bt_avg', 'bt', 'ig_avg', 'ig', 'ot_steps', 'ot_one'])
    
    parser.add_argument('--out_path', type=str,
                        default='./output/')
    
    parser.add_argument('--model_name', type=str,
                        default='llama-7b',
                        choices=['llama-7b', 'stanford-alpaca-7b', 'alpaca-lora-7b', 'vicuna-7b', 'vicuna-7b-1', 'vicuna-7b-2', 'vicuna-7b-3', \
                                 'vicuna-7b-4', 'vicuna-7b-5', 'vicuna-7b-6', 'llama-13b', 'vicuna-13b', 'alpaca-lora-13b']
                       )
    
    parser.add_argument('--num_samples', type=int,
                        default=10
                       )
    parser.add_argument('--num_beams', type=int,
                        default=10
                       )
    parser.add_argument('--steps', type=int,
                        default=10
                       )
    parser.add_argument('--max_mask_token', type=int,
                        default=10
                       )
    parser.add_argument('--topk', type=int,
                        default=10
                       )
    parser.add_argument('--if_or', type=bool,
                        default=False
                       )
    parser.add_argument('--concat', type=bool,
                        default=False
                       )
    parser.add_argument('--prompt', type=str,
                        default='alpaca',
                        choices=['alpaca', 'vicuna']
                       )
    parser.add_argument('--neg', type=bool,
                        default=False
                       )
    
    args = parser.parse_args()
    if args.method in ['ot_steps', 'ot_one']:
        file_path = args.out_path + f'results_{args.model_name}_{args.method}_{args.num_samples}_{args.prompt}_steps{args.steps}_topk{args.topk}_or{args.if_or}_concat{args.concat}_beams{args.num_beams}.json'
    else:
        file_path = args.out_path + f'results_{args.model_name}_{args.method}_{args.num_samples}_{args.prompt}.json'
    
    results = json.load(open(file_path))
    
    model, tokenizer, block_name, embedding_name, embedding_token_name, vocab_size, embed_dim = load_model(args.model_name)
    interpretor = Interpretor(model, block_name=block_name, embedding_name=embedding_name, embed_token_name=embedding_token_name)
    
    generation_config1 = GenerationConfig(
                temperature=0.7,
                repetition_penalty=1,
                top_p=0.0,
                num_beams=1,
                max_new_tokens=100,
                do_sample=False
            )
    generation_config2 = GenerationConfig(
                temperature=0.8,
                repetition_penalty=1,
                top_p=0.0,
                num_beams=4,
                max_new_tokens=80,
                do_sample=True
            )
    
    perb_steps = np.arange(0,1.1,0.1)
    pertub_probs_1 = [0]*len(perb_steps)
    pertub_probs_2 = [0]*len(perb_steps)
    
    for idx, res in enumerate(tqdm(results)):
        input_text = res['prompt']
        exp_result = torch.tensor(res['exp'])
        if len(exp_result.shape) > 1:
            exp_result = exp_result.mean(0)
            
        if args.neg:
            exp_result = -exp_result
        
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs.to(0)
        p = inputs.input_ids.shape[-1]
        
        generate_ids1 = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask\
                                       , generation_config=generation_config1)
        token_len_1 = generate_ids1.shape[-1]
        generate_ids2 = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask\
                                       , generation_config=generation_config2)
        token_len_2 = generate_ids2.shape[-1]
        
        for step_idx, step in enumerate(perb_steps):
            num_perb_tokens = int(step*p)
            # print(exp_result)
            _, indices = exp_result.topk(k=num_perb_tokens)
            # print('neg', indices)
            # inputs = tokenization_utils_base.BatchEncoding()
            # inputs['input_ids'] = generate_ids1
            # inputs['attention_mask'] = torch.ones(generate_ids1.shape)
            
            if step != 0:
                inputs['input_ids'][:, indices] = 0
                inputs['attention_mask'][:, indices] = 0
                
            with torch.no_grad():
                generate_ids1_perturb = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask\
                                                       , generation_config=generation_config1)
            C = interpretor._create_cost_matrix(generate_ids1, generate_ids1_perturb)
            pertub_probs_1[step_idx] += torch.diagonal(C[0]).detach().cpu().mean()
            
            with torch.no_grad():
                generate_ids2_perturb = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask\
                                                       , generation_config=generation_config2)
            C = interpretor._create_cost_matrix(generate_ids2, generate_ids2_perturb)
            pertub_probs_2[step_idx] += torch.diagonal(C[0]).detach().cpu().mean()
        
        # exit()
            
#             with torch.no_grad():
#                 logits1 = model.forward(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)['logits']
#             probs1 = torch.softmax(logits1, dim=-1)
#             for j in np.arange(p, token_len_1):
#                 pertub_probs_1[step_idx] += probs1[:, j, generate_ids1[0,j].detach().cpu()].detach().cpu()
                
            # inputs = tokenization_utils_base.BatchEncoding()
            # inputs['input_ids'] = generate_ids2
            # inputs['attention_mask'] = torch.ones(generate_ids2.shape)
            
#             if step != 0:
#                 inputs['input_ids'][:, indices] = 0
#                 inputs['attention_mask'][:, indices] = 0
                
#             with torch.no_grad():
#                 logits2 = model.forward(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)['logits']
#             probs2 = torch.softmax(logits2, dim=-1)
#             for k in np.arange(p, token_len_2):
#                 pertub_probs_2[step_idx] += probs2[:, k, generate_ids2[0,k].detach().cpu()].detach().cpu()
                
                
    np.save(f'./output_perturb/pertub_probs_1_{args.model_name}_{args.method}_{args.num_samples}_{args.neg}.npy', np.array(pertub_probs_1)/len(results))
    np.save(f'./output_perturb/pertub_probs_2_{args.model_name}_{args.method}_{args.num_samples}_{args.neg}.npy', np.array(pertub_probs_2)/len(results))
    print(np.array(pertub_probs_1)/len(results))
    print(np.array(pertub_probs_2)/len(results))
            
            
       
       
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    

    
    