import torch
import argparse
from tqdm import tqdm
from generate_exp_v2 import get_template, load_model_and_tokenizer, genereate_wrapper
from utils import grouper, read_txt_to_list_of_dict, write_list_of_dict_to_jsonl
from transformers import GenerationConfig
from sentence_similarity_utils import Sentence


def get_token_numbers_of_prefix_and_postfix(templates, tokenizer):
    # [('<s>', 0.0, 0.0004760307535747188),
    # ('[', 0.014298457652330399, 0.05699695835060846),
    # ('INST', 0.010563373565673828, 0.047749368725980706),
    # (']', 0.06065724045038223, 0.15000229001601145),
    # prefix = 4

    # ('[', 0.14283913373947144, 0.21214128025750714),
    # ('/', 0.13658569753170013, 0.17456363338953795),
    # ('INST', 0.0261276476085186, 0.06421725430407058),
    # (']', 2.0077450275421143, 0.46110612942212625)]
    # postfix = 4

    inputs_prefix = tokenizer(templates['prefix'].strip(), return_tensors="pt", add_special_tokens=False)
    inputs_postfix = tokenizer(templates['postfix'].strip(), return_tensors="pt", add_special_tokens=False)

    return inputs_prefix.input_ids.shape[-1], inputs_postfix.input_ids.shape[-1]


def batch_generate_greedy(batched_input_ids, batched_attention_masks, model, generation_config):
    batched_generated_ids = model.generate(
        input_ids=batched_input_ids,
        attention_mask=batched_attention_masks,
        generation_config=generation_config
    )
    return batched_generated_ids


def cut_off_text(text, start_pattern, end_pattern):
    index = text.find(start_pattern)
    start_index = index + len(start_pattern) if index != -1 else 0
    end_index = text.find("</s>")
    return text[start_index:end_index].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sequences.')
    parser.add_argument('--model_name', type=str, default='llama2-7b-chat')
    parser.add_argument('--attribution_fname', type=str)
    args = parser.parse_args()
    model_name = args.model_name.lower()

    template = get_template(model_name)
    model, tokenizer = load_model_and_tokenizer(model_name, torch.cuda.device_count() == 1)

    fname = args.attribution_fname
    fwname = fname.replace("attributions__", "eval_results_attributions__")
    attributions = torch.load(args.attribution_fname)

    prefix, postfix = get_token_numbers_of_prefix_and_postfix(template, tokenizer)
    # for <s>
    prefix += 1
    print("prefix", prefix)
    print("postfix", postfix)

    exp_method = fname.split("__")[1].split("method_")[-1]
    if "ours" == exp_method:
        methods = ['l2', 'kl', 'ot']
    else:
        methods = [exp_method]

    max_new_tokens = 256
    generation_config_greedy = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        early_stop=True,
    )
    sentence_transformer = Sentence("../hf_resources/all-MiniLM-L6-v2/")
    # s = sentence_transformer.similarity("This is an example sentence", "Each sentence is converted")

    for i, attributions_i in enumerate(tqdm(attributions)):
        query = attributions_i['instruction'] + attributions_i['input']
        input_text = f"{template['prefix']}{query.strip()}{template['postfix']}"
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs.to(0)

        for m in methods:
            print(m)
            attributions_to_eval = [d[m] for d in attributions_i['attributions']]
            attributions_to_eval = torch.Tensor(attributions_to_eval)
            input_token_num = len(attributions_to_eval)

            eval_results = {
                eval_mode: [] for eval_mode in ["morf", "lerf"]
            }
            for eval_mode in ["morf", "lerf"]:
                # print(eval_mode)
                _, indices = torch.sort(attributions_to_eval, descending=(eval_mode == "morf"))
                bool_mask = torch.logical_and(indices >= prefix, indices < (len(indices) - postfix))
                sorted_indices = indices[bool_mask]
                perb_steps = torch.arange(0, 1.1, 0.1)

                batched_input_ids = inputs.input_ids.repeat(len(perb_steps), 1)
                batched_attention_masks = inputs.attention_mask.repeat(len(perb_steps), 1)

                for step_idx, step in enumerate(perb_steps):
                    num_perb_tokens = int(step * len(sorted_indices))
                    indices_to_mask = sorted_indices[:num_perb_tokens]
                    batched_input_ids[step_idx, indices_to_mask] = 0
                    batched_attention_masks[step_idx, indices_to_mask] = 0

                batched_generate_ids = batch_generate_greedy(
                    batched_input_ids, batched_attention_masks, model, generation_config_greedy
                )

                generated_texts = tokenizer.batch_decode(
                    batched_generate_ids
                )

                for step_idx, step in enumerate(perb_steps):
                    s = sentence_transformer.similarity(
                        cut_off_text(generated_texts[0], "[/INST]", "</s>"),
                        cut_off_text(generated_texts[step_idx], "[/INST]", "</s>")
                    )
                    # print([cut_off_text(generated_texts[step_idx], "[/INST]", "</s>"), generated_texts[step_idx]])

                    eval_results[eval_mode].append(
                        {
                            "generated_text": generated_texts[step_idx],
                            'eval_score': s,
                        }
                    )

            attributions_i[f'eval_results_{m}'] = eval_results
        if i % 10 == 0:
            torch.save(attributions, fwname)

    torch.save(attributions, fwname)
