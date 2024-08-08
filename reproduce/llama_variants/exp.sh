CUDA_VISIBLE_DEVICES=1 python generate_exp.py --method ot_steps --num_samples 50 --model_name vicuna-7b --prompt vicuna --num_beams 2 --steps 10 --max_mask_token 5 --topk 100 --if_or 1
#CUDA_VISIBLE_DEVICES=1 python generate_exp.py --method lime --num_samples 50 --model_name vicuna-7b --prompt vicuna
#CUDA_VISIBLE_DEVICES=1 python generate_exp.py --method bt --num_samples 50 --model_name vicuna-7b --prompt vicuna
#CUDA_VISIBLE_DEVICES=1 python generate_exp.py --method ig_avg --num_samples 50 --model_name vicuna-7b --prompt vicuna
#CUDA_VISIBLE_DEVICES=1 python generate_exp.py --method ig --num_samples 50 --model_name vicuna-7b --prompt vicuna
# python generate_exp.py --method bt_avg --num_samples 50 --model_name stanford-alpaca-7b
# python generate_exp.py --method lime --num_samples 50 --model_name stanford-alpaca-7b
# python generate_exp.py --method bt --num_samples 50 --model_name stanford-alpaca-7b
# python generate_exp.py --method ig_avg --num_samples 50 --model_name stanford-alpaca-7b
# python generate_exp.py --method ig --num_samples 50 --model_name stanford-alpaca-7b
# CUDA_VISIBLE_DEVICES=1 python generate_exp.py --method bt_avg --num_samples 50 --model_name alpaca-lora-7b --prompt alpaca
# CUDA_VISIBLE_DEVICES=1 python generate_exp.py --method lime --num_samples 50 --model_name alpaca-lora-7b --prompt alpaca
# CUDA_VISIBLE_DEVICES=1 python generate_exp.py --method bt --num_samples 50 --model_name alpaca-lora-7b --prompt alpaca
# CUDA_VISIBLE_DEVICES=1 python generate_exp.py --method ig_avg --num_samples 50 --model_name alpaca-lora-7b --prompt alpaca
# CUDA_VISIBLE_DEVICES=1 python generate_exp.py --method ig --num_samples 50 --model_name alpaca-lora-7b --prompt alpaca
# python generate_exp.py --method bt_avg --num_samples 50 --model_name vicuna-7b
# python generate_exp.py --method lime --num_samples 50 --model_name vicuna-7b
# python generate_exp.py --method bt --num_samples 50 --model_name vicuna-7b
# python generate_exp.py --method ig_avg --num_samples 50 --model_name vicuna-7b
# python generate_exp.py --method ig --num_samples 50 --model_name vicuna-7b