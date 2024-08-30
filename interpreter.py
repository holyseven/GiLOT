import transformers
from functools import partial
import numpy as np
import sklearn.metrics
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import torch
from typing import Optional
from utils import grouper, genereate_wrapper
import argparse

class Interpreter:
    def __init__(self,
                 model: torch.nn.Module,
                 block_name: Optional[str] = None,
                 embedding_name: Optional[str] = None,
                 embed_token_name: Optional[str] = None):

        self.model = model
        self.model.eval()
        self.model.enable_input_require_grads()
        self.block_name = block_name
        self.embedding_name = embedding_name
        self.embed_token_name = embed_token_name

    @staticmethod
    def _perturbe(inputs, num_samples, p_len):
        inputs_perturb = transformers.tokenization_utils_base.BtchEncoding()
        inputs_perturb['input_ids'] = inputs.input_ids.repeat(num_samples, 1)
        inputs_perturb['attention_mask'] = inputs.attention_mask.repeat(num_samples, 1)
        perturbed_size_list = np.random.randint(1, p_len, num_samples)

        for i, num_perturbed in enumerate(perturbed_size_list):
            if i != 0:
                perturbed_indices = np.random.choice(np.arange(p_len), num_perturbed, replace=True)
            else:
                perturbed_indices = []
            for ii in range(p_len):
                if ii in perturbed_indices:
                    inputs_perturb.input_ids[i][ii] = 0
                    inputs_perturb.attention_mask[i][ii] = 0

        return inputs_perturb

    @staticmethod
    def _create_cost_matrix_cos(normalized_embed_weight, indices):
        embeddings = normalized_embed_weight[indices]
        cosine_matrix = torch.matmul(embeddings, embeddings.T)
        return 1 - cosine_matrix

    @staticmethod
    def _ipot_torch(a1, a2, C, beta=2, max_iter=1000, L=1, use_path=True, return_map=True, return_loss=True):
        # - C is the (ns,nt) metric cost matrix
        # - a and b are source and target weights (sum to 1)

        a1 += 0.0001
        a2 += 0.0001

        n = len(a1)
        v = torch.ones([n, ], dtype=torch.float64).cuda()
        u = torch.ones([n, ], dtype=torch.float64).cuda()

        P = (torch.ones((n, n), dtype=torch.float64) / n ** 2).cuda()

        K = torch.exp(-(C / beta))
        if return_loss:
            loss = []
        for outer_i in range(max_iter):

            Q = K * P

            if not use_path:
                v = torch.ones([n, ], dtype=torch.float64)
                u = torch.ones([n, ], dtype=torch.float64)

            for i in range(L):
                u = a1 / torch.matmul(Q, v)
                v = a2 / torch.matmul(Q.T, u)

            P = torch.unsqueeze(u, dim=1) * Q * torch.unsqueeze(v, dim=0)
            if return_loss:
                W = torch.sum(P * C)
                loss.append(W)

        if return_loss:
            if return_map:
                return P, loss
            else:
                return loss

        else:
            if return_map:
                return P

            else:
                return None

    @staticmethod
    def _topp_intersection(proba1, proba2, topp=0.9999, topk=1000, normalize=True):
        def _get_one(p):
            sorted_p, sorted_indices = torch.sort(p, descending=True)
            cumulative_probs = torch.cumsum(sorted_p, dim=0)
            threshold_index = (cumulative_probs >= topp).nonzero().min().item()
            threshold_index = min(threshold_index+1, topk)
            top_indices = sorted_indices[:threshold_index]
            return top_indices

        indices1 = _get_one(proba1)
        indices2 = _get_one(proba2)

        union_indices = torch.cat((indices1, indices2)).unique()
        s1 = proba1[union_indices]
        s2 = proba2[union_indices]

        if normalize:
            s1 /= s1.sum()
            s2 /= s2.sum()

        return s1, s2, union_indices

    @staticmethod
    def _distances_fn(x):
        return sklearn.metrics.pairwise_distances(
            x, x[:1, :], metric='cosine').ravel() * 100

    @staticmethod
    def _kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    @staticmethod
    def kl_divergence(pj_a, pj):
        kl_loss = torch.nn.KLDivLoss(reduction="sum")
        return kl_loss(torch.log(pj_a), pj)


    def optimal_transport(self, pj_a, p_j):
        embed_weight = eval(self.embedding_name)
        normalized_embed_weight = embed_weight / torch.norm(embed_weight, dim=1)[:, None]
        s1, s2, union_indices = self._topp_intersection(p_j, pj_a)
        C = self._create_cost_matrix_cos(normalized_embed_weight, union_indices).cuda()
        ot_plan, ot_loss = self._ipot_torch(s1, s2, C)
        return ot_loss[-1]

    @staticmethod
    def l2_distance(pj_a, p_j):
        return torch.norm(pj_a - p_j)

    # Baseline method LIME for interpreting model generation
    def generate_lime(self, inputs, generate_ids, p_len, num_round, vocab_size, each_round=5):
        num_tokens = generate_ids.shape[1]
        one_hot = np.zeros((each_round * num_round, generate_ids.shape[1], vocab_size), dtype=np.float32)
        for j in np.arange(p_len, num_tokens):
            one_hot[:, j, generate_ids[0, j].detach().cpu()] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        inputs_perturb = self._perturbe(inputs, num_samples=each_round * num_round, p_len=p_len)
        generate_ids_perturbe = transformers.tokenization_utils_base.BatchEncoding()
        generate_ids_perturbe['input_ids'] = torch.cat(
            (inputs_perturb['input_ids'], generate_ids[:, p_len:].repeat(each_round * num_round, 1)), dim=1)
        generate_ids_perturbe['attention_mask'] = torch.cat((inputs_perturb['attention_mask'], torch.ones(
            (each_round * num_round, (generate_ids.shape[1] - p_len))).cuda()), dim=1)

        probs_all = torch.zeros(each_round * num_round, generate_ids.shape[1], vocab_size)
        with torch.no_grad():
            for i in range(num_round):
                logits = \
                self.model.forward(generate_ids_perturbe.input_ids[(i * each_round):(each_round * i + each_round)],
                                   attention_mask=generate_ids_perturbe.attention_mask[
                                                  (i * each_round):(each_round * i + each_round)])['logits']
                probs = torch.softmax(logits, dim=2).cpu().float()
                probs_all[(i * each_round):(each_round * i + each_round), :, :] = probs

        kernel_fn = partial(self._kernel, kernel_width=25)
        weights = kernel_fn(self._distances_fn(inputs_perturb.attention_mask.detach().cpu().numpy()))

        prob_target = torch.log(probs_all[one_hot.bool()]).reshape(each_round * num_round, -1).sum(dim=1)
        prob_target_scaled = (prob_target - prob_target.min()) / prob_target.var() ** 0.5

        model_g = Ridge(
            alpha=0,
            fit_intercept=False,
            random_state=1234,
            positive=True
        )
        model_g.fit(inputs_perturb.attention_mask.cpu(), prob_target_scaled, sample_weight=weights)
        R = model_g.coef_

        return R

    def generate_ig_avg(self, generate_ids, p_len, steps=1, embed_dim=None):
        b = generate_ids.shape[0]
        total_len = generate_ids.shape[1]
        R = torch.zeros(b, p_len).cpu()

        for i in np.arange(p_len, total_len):
            with torch.no_grad():
                logits = self.model.forward(generate_ids[:, :i])['logits']
                probs = torch.softmax(logits, dim=2).cpu().float()

            ig = torch.zeros(logits.shape[0], logits.shape[1], embed_dim).cuda()

            num_tokens = logits.shape[1]
            one_hot = np.zeros(probs.shape, dtype=np.float32)
            one_hot[0, -1, generate_ids[0][i].detach().cpu()] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)

            for step in np.arange(1, steps + 1):
                logits = self.model.forward(input_ids=generate_ids[:, :i], alpha=(step / steps))['logits']
                probs = torch.softmax(logits, dim=2)
                loss = torch.log(probs[one_hot.bool()]).sum()
                self.model.zero_grad()
                grad = torch.autograd.grad(loss, [eval(self.embedding_name)])[0].detach()
                ig += grad
            intergrated_gradient = ig / steps
            importance = torch.sum(intergrated_gradient * eval(self.embedding_name),
                                   axis=-1).abs().detach().cpu().numpy()[:, :p_len]

            R = R + importance

        return R

    def generate_ig(self, generate_ids, p_len, steps=1, embed_dim=None):
        with torch.no_grad():
            logits = self.model.forward(generate_ids)['logits']
            probs = torch.softmax(logits, dim=2).cpu().float()

        ig = torch.zeros(logits.shape[0], logits.shape[1], embed_dim).cuda()
        num_tokens = logits.shape[1]

        one_hot = np.zeros(probs.shape, dtype=np.float32)
        for j in np.arange(p_len, num_tokens):
            one_hot[:, j, generate_ids[0, j].detach().cpu()] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        for i in np.arange(1, steps + 1):
            logits = self.model.forward(input_ids=generate_ids, alpha=(i / steps))['logits']
            probs = torch.softmax(logits, dim=2)
            loss = torch.log(probs[one_hot.bool()]).sum()
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, [eval(self.embedding_name)])[0].detach()
            ig += grad
        intergrated_gradient = ig / steps
        R = torch.sum(intergrated_gradient * eval(self.embedding_name), axis=-1).abs().detach().cpu().numpy()[:, :p_len]

        return R

    def compute_prob_distribution(self, generate_ids, num_new_tokens, a_position_to_mask=None):
        attention_mask = torch.ones_like(generate_ids)
        if a_position_to_mask is not None:
            generate_ids[:, a_position_to_mask] = 0
            attention_mask[:, a_position_to_mask] = 0
        # print('compute_prob_distribution:', generate_ids)
        # print('compute_prob_distribution:', attention_mask)
        probas_list = []
        batches = grouper(num_new_tokens, range(generate_ids.size(0)), -1)
        for block_index, block in enumerate(batches):
            block = list(filter(lambda data_index: data_index != -1, block))
            block_generate_ids = generate_ids[block]
            block_attention_mask = attention_mask[block]

            with torch.no_grad():
                logits = self.model.forward(
                    input_ids=block_generate_ids,
                    attention_mask=block_attention_mask
                )

                # torch.Size([num_return_sequences, input_token_num+max_new_tokens, vocab_size])
                # e.g., torch.Size([20, 81, 32000])
                probas = torch.softmax(logits.logits, dim=-1)
                probas_list.append(probas)

                del logits

        return torch.cat(probas_list)

    @staticmethod
    def get_jth_token_prob_and_seq_prob(generate_ids, probas, input_token_num, unknown_token_id=0, normalize=True):
        new_token_probas = probas[:, (input_token_num - 1):, :]

        sequence_range = torch.arange(new_token_probas.shape[0])
        generate_ids_slice = generate_ids[:, input_token_num:]
        token_range = torch.arange(generate_ids_slice.shape[-1])
        selected_probas = new_token_probas[
            sequence_range[:, None],
            token_range,
            generate_ids_slice
        ]

        # modifying the unknown token's probability to 1.0, these are tokens after </s>
        selected_probas[generate_ids_slice == unknown_token_id] = 1.0

        # beams probability
        probs_sequences = torch.prod(selected_probas, axis=1)

        # weighted by beams probability
        token_j_prob = (probs_sequences[:, None] * new_token_probas[:, -1, :]).sum(0)

        if normalize:
            # sum normalization.
            token_j_prob /= token_j_prob.sum()

        return token_j_prob, probs_sequences

    def interpret_ours(
        self, input_ids, num_beams, num_new_tokens, distance="optimal_transport"
    ):
        if distance not in ["optimal_transport", "kl_divergence", "l2_distance"]:
            raise ValueError('Distance type not supported!')

        p_len = input_ids.shape[-1]
        config = transformers.GenerationConfig(
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_new_tokens=num_new_tokens,
            do_sample=False,
            early_stop=False,
        )
        generate_ids = genereate_wrapper(input_ids, self.model, config)
        proba_distr = self.compute_prob_distribution(generate_ids, num_new_tokens)
        pj, probs_sequences = self.get_jth_token_prob_and_seq_prob(generate_ids, proba_distr, p_len)

        # step2, masking a-th token and compute prob of J-th output token.

        token_j_probs_masking_input_tokens = []
        for a in range(input_ids.shape[-1]):
            generate_ids_masking_a = genereate_wrapper(input_ids.clone(), self.model, config,
                                                       a_position_to_mask=a)
            proba_distr_masking_a = self.compute_prob_distribution(generate_ids_masking_a, p_len, a_position_to_mask=a)
            pj_a, _ = self.get_jth_token_prob_and_seq_prob(generate_ids_masking_a, proba_distr_masking_a, p_len)
            token_j_probs_masking_input_tokens.append(pj_a)

        attributions = []

        for a, pj_a in enumerate(token_j_probs_masking_input_tokens):
            with torch.no_grad():
                distance_func = getattr(self, distance, None)
                res = {
                    'token_index': a,
                    distance: distance_func(pj_a, pj)
                }
                attributions.append(res)
        return attributions, probs_sequences


if __name__ == "__main__":
    from utils import get_template, load_model, read_txt_to_list_of_dict
    import os
    from tqdm import tqdm
    import time

    parser = argparse.ArgumentParser(description='Generate explanantion results')
    parser.add_argument('--method', type=str, choices=["optimal_transport", "kl_divergence", "l2_distance"])
    parser.add_argument('--model_name', type=str, default='llama2-7b')
    parser.add_argument('--beams', type=int, default=200)
    parser.add_argument('--max_new_tokens', type=int, default=10)
    args = parser.parse_args()

    model_name = args.model_name.lower()
    template = get_template(model_name)
    model, tokenizer, block_name, embedding_name, embed_token_name, _, _ = load_model(model_name)
    interpreter = Interpreter(model, block_name, embedding_name, embed_token_name)

    # read samples
    input_data = read_txt_to_list_of_dict("data/select_queries_0406.jsonl")

    os.makedirs("output", exist_ok=True)
    fwname = f"output/attributions__method_{args.method}__model_{args.model_name.lower()}__beams_{args.beam}__" \
             f"tokens_{args.max_new_tokens}.pt"
    if os.path.exists(fwname):
        print("reading already computed results", fwname)
        res = torch.load(fwname)
        for d in res:
            assert "attributions" in d, "missing attributions in the result"
    else:
        res = []

    for sample_i, d in enumerate(tqdm(input_data)):
        if sample_i < len(res):
            continue
        tik = time.time()
        query = d['instruction'] + d['input']
        input_text = f"{template['prefix']}{query.strip()}{template['postfix']}"

        if args.method in ["optimal_transport", "kl_divergence", "l2_distance"]:
            # multiple J's result will be merged manually.
            inputs = tokenizer(input_text, return_tensors="pt")
            inputs.to(0)
            attributions, probs_sequences = interpreter.interpret_ours(inputs.input_ids, args.beams, args.max_new_tokens,\
                                                                       args.method)
            # probs of sequences returned by beam research.
            d['probs_sequences'] = probs_sequences
            d['template'] = template
        else:
            raise NotImplementedError

        d['attributions'] = attributions
        d['time_cost'] = time.time() - tik
        res.append(d)

        if sample_i % 10 == 0:
            torch.save(res, fwname)

    torch.save(res, fwname)