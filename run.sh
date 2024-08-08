gpu=0
beams=20
m=20

CUDA_VISIBLE_DEVICES=$gpu python interpreter.py --method "optimal_transport" --model_name 'llama2-7b' --beams $beams --max_new_tokens $m