from interpretor import Interpretor
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, set_seed
from peft import (
    PeftModel,
     PeftModelForCausalLM
)
import json
from tqdm import tqdm
import argparse
import os

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
        
        lora_weights = "/root/codespace/ongoing_projects/baselines/model_weights/tloen-alpaca-lora-7b"
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
        model_path = '/root/model_weights/stanford-alpaca-7b'
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
    
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    
    model, tokenizer, block_name, embedding_name, embedding_token_name, vocab_size, embed_dim = load_model(args.model_name)
    interpretor = Interpretor(model, block_name=block_name, embedding_name=embedding_name, embed_token_name=embedding_token_name)
    
    generation_config = GenerationConfig(
                temperature=0.7,
                repetition_penalty=1,
                top_p=0.0,
                num_beams=1,
                max_new_tokens=128,
                do_sample=False
            )
    
    samples = json.load(open('./select_samples.json', 'r'))
    # samples = [{'instruction': "Write a story about a homemaker. ", 'input': ""}, {'instruction': "Write a story about an architect. ", 'input': ""}, \
    #            {'instruction': "Write a story about a female homemaker. ", 'input': ""}, {'instruction': "Write a story about a male architect. ", 'input': ""}, \
    #           {'instruction': "Write a story about a male homemaker. ", 'input': ""}, {'instruction': "Write a story about a female architect. ", 'input': ""}]
    
    res_path = args.out_path + f'results_{args.model_name}_{args.method}_{args.num_samples}_{args.prompt}_steps{args.steps}_topk{args.topk}_or{args.if_or}_concat{args.concat}_beams{args.num_beams}.json'
    reponses_dict_list = []
    
    for idx, sample in enumerate(tqdm(samples[:args.num_samples])):
        instruction = sample['instruction']
        input_text = sample['input']
        
        if args.prompt == 'alpaca':
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: {instruction}{input_text} \n\n### Response: "
        elif args.prompt == 'vicuna':
            prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.</s>USER:{instruction}{input_text}</s>ASSISTANT:"
        # elif args.prompt == 'llama':
        #     prompt = 
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs.to(0)
        p_len = inputs.input_ids.shape[-1]
        
        set_seed(42)
        generate_ids = model.generate(input_ids=inputs.input_ids, generation_config=generation_config)
        generate_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        if args.method == 'lime':
            exp = interpretor.generate_lime(inputs, generate_ids, p_len, num_round=100, vocab_size=vocab_size, each_round=5)
        
        elif args.method == 'bt_avg':
            exp = interpretor.generate_bt_token_average(generate_ids, p_len, steps=5)[0]
        
        elif args.method == 'bt':
            exp = interpretor.generate_bt_token(generate_ids, p_len, steps=20)[0]
        
        elif args.method == 'ig_avg':
            exp = interpretor.generate_ig_average(generate_ids, p_len, steps=5, embed_dim=embed_dim)[0]
        
        elif args.method == 'ig':
            exp = interpretor.generate_ig(generate_ids, p_len, steps=20, embed_dim=embed_dim)[0]
        
        elif args.method == 'ot_one':
            exp = interpretor.generate_ours_one_step(inputs, topk=args.topk, beta=1, if_or=args.if_or)[0]
            
        elif args.method == 'ot_steps':
            exp = interpretor.generate_ours_more_steps(inputs, args.num_beams, args.steps, args.max_mask_token, vocab_size, topk=args.topk, beta=1, if_or=args.if_or, concat=args.concat)
            
        reponses_dict = {'prompt': prompt, 'reponse': generate_text, 'exp': exp.tolist()}
        reponses_dict_list.append(reponses_dict)
        
    with open(res_path, 'w') as file:
        json.dump(reponses_dict_list, file)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    

    
    