import transformers
import itertools
import json
import re
import collections
from functools import partial
import numpy as np 
import sklearn.metrics
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import torch
from scipy import stats

class Interpretor:
    def __init__(self, model, block_name=None, embedding_name=None, embed_token_name=None):
        self.model = model
        self.model.eval()
        self.model.enable_input_require_grads()
        self.block_name = block_name
        self.embedding_name = embedding_name
        self.embed_token_name = embed_token_name

    def _perturbe(self, inputs, num_samples, p_len):
        
        inputs_perturb = transformers.tokenization_utils_base.BatchEncoding()
        inputs_perturb['input_ids'] = inputs.input_ids.repeat(num_samples, 1)
        inputs_perturb['attention_mask'] = inputs.attention_mask.repeat(num_samples, 1)
        perturbed_size_list = np.random.randint(1, p_len, num_samples)
    
        for i, num_perturbed in enumerate(perturbed_size_list):
            if i != 0:
                perturbed_indices = np.random.choice(np.arange(p_len),num_perturbed,replace=True)
            else:
                perturbed_indices = []
            for ii in range(p_len):
                if ii in perturbed_indices:
                    inputs_perturb.input_ids[i][ii] = 0
                    inputs_perturb.attention_mask[i][ii] = 0

        return inputs_perturb
    
    def _create_cost_matrix(self, indices_list1, indices_list2):
        
        num_samples = len(indices_list1)
        embedding1 = eval(self.embed_token_name)[indices_list1]
        embedding2 = eval(self.embed_token_name)[indices_list2]
        cos = torch.nn.CosineSimilarity(dim=-1)
        _, num_vocab1, embed_size = embedding1.shape
        _, num_vocab2, embed_size = embedding2.shape
        embedding1_row, embedding2_col = embedding1[:, None, :, :], embedding2[:, :, None, :]
        embedding1_row, embedding2_col = embedding1_row.expand(num_samples, num_vocab2, num_vocab1, embed_size), \
                                                embedding2_col.expand(num_samples, num_vocab2, num_vocab1, embed_size)
        out = cos(embedding1_row, embedding2_col)
        
        return (1-out.clamp(min=-1, max=1))/2

    def _ot_calculate_batch_first(self, inputs, probs, start, end, p_len, num_beams, top_k, vocab_size, beta=1, if_or=True):
        out = np.zeros(p_len)
        num_batch = end-start

        indices = start + torch.arange(num_batch)
        masked_inputs = transformers.tokenization_utils_base.BatchEncoding()
        masked_inputs['input_ids'] = inputs.input_ids.clone().unsqueeze(0).repeat(num_batch, 1, 1)
        masked_inputs['attention_mask'] = inputs.attention_mask.clone().unsqueeze(0).repeat(num_batch, 1, 1)

        masked_inputs['input_ids'][torch.arange(num_batch), :, indices] = 0
        masked_inputs['attention_mask'][torch.arange(num_batch), :, indices] = 0

        masked_inputs['input_ids'] = masked_inputs['input_ids'].reshape(num_batch*num_beams, -1)
        masked_inputs['attention_mask'] = masked_inputs['attention_mask'].reshape(num_batch*num_beams, -1)

        with torch.no_grad():
            logits_masked = self.model.forward(input_ids=masked_inputs.input_ids, attention_mask=masked_inputs.attention_mask,
                                      use_cache=True, output_hidden_states=True)
        probs_masked = torch.softmax(logits_masked.logits, dim=2).reshape(num_batch, num_beams, p_len, vocab_size)
        past = logits_masked.past_key_values
        del masked_inputs, logits_masked

        for i, idx in enumerate(np.arange(start, end)):
            p1, p2, indices1, indices2 = self._topk_intersection(probs[:, -1, :], probs_masked[i:(i+1), 0, -1, :], top_k, if_or=if_or)
            C = self._create_cost_matrix(indices1, indices2)[0]
            # P, loss = self._ipot_torch(p1.flatten().cuda(), p2.flatten().cuda(), C, beta)
            loss = stats.entropy(p1.flatten().cuda(), p2.flatten().cuda())
            out[idx] = loss.detach().cpu().numpy()
            # out[idx] = loss[-1]/p1.shape[-1]

        return out, past, probs_masked
    
    def _ot_calculate_first_indices(self, inputs, probs, indices, p_len, num_beams, top_k, vocab_size, beta=1, if_or=True):
        out = np.zeros(p_len)
        num_batch = len(indices)

        masked_inputs = transformers.tokenization_utils_base.BatchEncoding()
        masked_inputs['input_ids'] = inputs.input_ids.clone().unsqueeze(0).repeat(num_batch, 1, 1)
        masked_inputs['attention_mask'] = inputs.attention_mask.clone().unsqueeze(0).repeat(num_batch, 1, 1)

        for n, indice in enumerate(indices):
            masked_inputs['input_ids'][n, :, indice] = 0
            masked_inputs['attention_mask'][n, :, indice] = 0

        masked_inputs['input_ids'] = masked_inputs['input_ids'].reshape(num_batch*num_beams, -1)
        masked_inputs['attention_mask'] = masked_inputs['attention_mask'].reshape(num_batch*num_beams, -1)

        with torch.no_grad():
            logits_masked = self.model.forward(input_ids=masked_inputs.input_ids, attention_mask=masked_inputs.attention_mask,
                                      use_cache=True, output_hidden_states=True)
        probs_masked = torch.softmax(logits_masked.logits, dim=2).reshape(num_batch, num_beams, p_len, vocab_size)
        past = logits_masked.past_key_values
        del masked_inputs, logits_masked

        for i, idx in enumerate(indices):
            p1, p2, indices1, indices2 = self._topk_intersection(probs[:, -1, :], probs_masked[i:(i+1), 0, -1, :], top_k, if_or=if_or)
            C = self._create_cost_matrix(indices1, indices2)[0]
            # P, loss = self._ipot_torch(p1.flatten().cuda(), p2.flatten().cuda(), C, beta)
            loss = stats.entropy(p1.flatten().cuda(), p2.flatten().cuda())
            # out[idx] = (loss[-1]/p1.shape[-1]).detach().cpu().numpy()
            out[idx] = loss.detach().cpu().numpy()

        return out, past, probs_masked
    
    def _ot_calculate_batch_more_avg(self, inputs, probs, start, end, p_len, num_beams, top_k, vocab_size, beta=1, if_or=True, past=None, probs_previous=None, seq_w=None):
        out = np.zeros(p_len)
        num_batch = end-start
        _, all_len = inputs.attention_mask.shape
        s = all_len-p_len

        indices = start + torch.arange(num_batch)
        masked_inputs = transformers.tokenization_utils_base.BatchEncoding()
        masked_inputs['input_ids'] = inputs.input_ids.clone().unsqueeze(0).repeat(num_batch, 1, 1)
        masked_inputs['attention_mask'] = inputs.attention_mask.clone().unsqueeze(0).repeat(num_batch, 1, 1)
        masked_inputs['attention_mask'][torch.arange(num_batch), :, indices] = 0

        masked_inputs['input_ids'] = masked_inputs['input_ids'].reshape(num_batch*num_beams, -1)
        masked_inputs['attention_mask'] = masked_inputs['attention_mask'].reshape(num_batch*num_beams, -1)

        with torch.no_grad():
            logits_masked = self.model.forward(input_ids=masked_inputs.input_ids, attention_mask=masked_inputs.attention_mask,
                                      use_cache=True, output_hidden_states=True, past_key_values=past)
        probs_masked = torch.softmax(logits_masked.logits, dim=2).reshape(num_batch, num_beams, 1, vocab_size)
        probs_masked = torch.cat([probs_previous, probs_masked], dim=-2)
        past = logits_masked.past_key_values
        del masked_inputs, logits_masked

        for i, idx in enumerate(np.arange(start, end)):
            probs_avg = torch.matmul(seq_w, probs.reshape(num_beams, -1)).reshape(1, -1, vocab_size)
            probs_masked_avg = torch.matmul(seq_w, probs_masked[i:(i+1)].reshape(num_beams, -1)).reshape(1, -1, vocab_size)
            p1, p2, indices1, indices2 = self._topk_intersection(torch.log(probs_avg[:, -(s+1):, :]).sum(dim=1),
                                                       torch.log(probs_masked_avg[:, -(s+1):, :]).sum(dim=1),
                                                       top_k, if_or=if_or)

            C = self._create_cost_matrix(indices1, indices2)[0]

            # P, loss = self._ipot_torch(p1.flatten().cuda(), p2.flatten().cuda(), C, beta)
            # out[idx] = loss[-1]/p1.shape[-1]
            loss = stats.entropy(p1.flatten().cuda(), p2.flatten().cuda())
            out[idx] = loss.detach().cpu().numpy()

        return out, past, probs_masked
    
    def _ot_calculate_more_avg_indices(self, inputs, probs, indices, p_len, num_beams, top_k, vocab_size, beta=1, if_or=True, past=None, probs_previous=None, seq_w=None):
        out = np.zeros(p_len)
        out = np.zeros(p_len)
        num_batch = len(indices)
        _, all_len = inputs.attention_mask.shape
        s = all_len-p_len

        masked_inputs = transformers.tokenization_utils_base.BatchEncoding()
        masked_inputs['input_ids'] = inputs.input_ids.clone().unsqueeze(0).repeat(num_batch, 1, 1)
        masked_inputs['attention_mask'] = inputs.attention_mask.clone().unsqueeze(0).repeat(num_batch, 1, 1)
        
        for n, indice in enumerate(indices):
            masked_inputs['attention_mask'][n, :, indice] = 0

        masked_inputs['input_ids'] = masked_inputs['input_ids'].reshape(num_batch*num_beams, -1)
        masked_inputs['attention_mask'] = masked_inputs['attention_mask'].reshape(num_batch*num_beams, -1)

        with torch.no_grad():
            logits_masked = self.model.forward(input_ids=masked_inputs.input_ids, attention_mask=masked_inputs.attention_mask,
                                      use_cache=True, output_hidden_states=True, past_key_values=past)
        probs_masked = torch.softmax(logits_masked.logits, dim=2).reshape(num_batch, num_beams, 1, vocab_size)
        probs_masked = torch.cat([probs_previous, probs_masked], dim=-2)
        past = logits_masked.past_key_values
        del masked_inputs, logits_masked

        for i, idx in enumerate(indices):
            probs_avg = torch.matmul(seq_w, probs.reshape(num_beams, -1)).reshape(1, -1, vocab_size)
            probs_masked_avg = torch.matmul(seq_w, probs_masked[i:(i+1)].reshape(num_beams, -1)).reshape(1, -1, vocab_size)
            p1, p2, indices1, indices2 = self._topk_intersection(torch.log(probs_avg[:, -(s+1):, :]).sum(dim=1),
                                                       torch.log(probs_masked_avg[:, -(s+1):, :]).sum(dim=1),
                                                       top_k, if_or=if_or)

            C = self._create_cost_matrix(indices1, indices2)[0]

            # P, loss = self._ipot_torch(p1.flatten().cuda(), p2.flatten().cuda(), C, beta)
            # out[idx] = (loss[-1]/p1.shape[-1]).detach().cpu().numpy()
            loss = stats.entropy(p1.flatten().cuda(), p2.flatten().cuda())
            out[idx] = loss.detach().cpu().numpy()

        return out, past, probs_masked
    
    def _ot_calculate_batch_more_concat(inputs, probs, start, end, p_len, num_beams, top_k, vocab_size, beta=1, if_or=True, past=None, probs_previous=None, seq_w=None):
        out = np.zeros(p_len)
        num_batch = end-start
        _, all_len = inputs.attention_mask.shape
        s = all_len-p_len

        indices = start + torch.arange(num_batch)
        masked_inputs = transformers.tokenization_utils_base.BatchEncoding()
        masked_inputs['input_ids'] = inputs.input_ids.clone().unsqueeze(0).repeat(num_batch, 1, 1)
        masked_inputs['attention_mask'] = inputs.attention_mask.clone().unsqueeze(0).repeat(num_batch, 1, 1)
        masked_inputs['attention_mask'][torch.arange(num_batch), :, indices] = 0

        masked_inputs['input_ids'] = masked_inputs['input_ids'].reshape(num_batch*num_beams, -1)
        masked_inputs['attention_mask'] = masked_inputs['attention_mask'].reshape(num_batch*num_beams, -1)

        with torch.no_grad():
            logits_masked = self.model.forward(input_ids=masked_inputs.input_ids, attention_mask=masked_inputs.attention_mask,
                                      use_cache=True, output_hidden_states=True, past_key_values=past)
        probs_masked = torch.softmax(logits_masked.logits, dim=2).reshape(num_batch, num_beams, 1, vocab_size)
        probs_masked = torch.cat([probs_previous, probs_masked], dim=-2)
        past = logits_masked.past_key_values
        del masked_inputs, logits_masked

        if if_or:
            for i, idx in enumerate(np.arange(start, end)):
                p1_all = [None]*num_beams
                p2_all = [None]*num_beams
                C_all = [None]*num_beams
                num_all = torch.zeros((num_beams+1), dtype=torch.int32)
                for j in range(num_beams):
                    p1, p2, indices1, indices2 = self._topk_intersection(torch.log(probs[j:(j+1), -(s+1):, :]).sum(dim = -2), \
                                                       torch.log(probs_masked[i, j:(j+1), -(s+1):, :]).sum(dim=-2), top_k, if_or=if_or)
                    C = self._create_cost_matrix(indices1, indices2)
                    p1_all[j] = p1
                    p2_all[j] = p2
                    C_all[j] = C
                    num_all[j+1] = p1.shape[0]

                C_diag = torch.ones((num_all.sum(), num_all.sum()), device="cuda", dtype=torch.float32)
                for m, n in enumerate(C_all):
                    C_diag[num_all[:(m+1)].sum():num_all[:(m+2)].sum(), num_all[:(m+1)].sum():num_all[:(m+2)].sum()] = n[0]

                # P, loss = self._ipot_torch((torch.cat(p1_all)), (torch.cat(p2_all)), C_diag, beta)
                # out[idx] = loss[-1]/num_all.sum()
                loss = stats.entropy(p1.flatten().cuda(), p2.flatten().cuda())
                out[idx] = loss.detach().cpu().numpy()
        else:
            for i, idx in enumerate(np.arange(start, end)):
                p1, p2, indices1, indices2 = self._topk_intersection(torch.log(probs[:, -(s+1):, :]).sum(dim = 1), \
                                                           torch.log(probs_masked[i, :, -(s+1):, :]).sum(dim=1), top_k)
                C = self._create_cost_matrix(indices1, indices2)
                C_diag = torch.ones([num_beams*top_k, num_beams*top_k], device="cuda", dtype=torch.float32)
                for m, n in enumerate(C):
                    C_diag[(m*top_k):((m+1)*top_k), (m*top_k):((m+1)*top_k)] = n

                # P, loss = self._ipot_torch(p1.flatten(), p2.flatten(), C_diag, beta)
                # out[idx] = loss[-1]/p1.shape[-1]
                loss = stats.entropy(p1.flatten().cuda(), p2.flatten().cuda())
                out[idx] = loss.detach().cpu().numpy()

        return out, past, probs_masked
    
    def _ipot_torch(self, a1,a2,C,beta=2,max_iter=1000,L=1,use_path = True, return_map = True, return_loss = True):
    # - C is the (ns,nt) metric cost matrix
    # - a and b are source and target weights (sum to 1)

        # 提升ipot的稳定性，不然容易nan
        a1 += 0.0001
        a2 += 0.0001
        
        n = len(a1)
        v = torch.ones([n,], dtype=torch.float64).cuda()
        u = torch.ones([n,], dtype=torch.float64).cuda()

        P = (torch.ones((n,n), dtype=torch.float64)/n**2).cuda()

        K=torch.exp(-(C/beta))
        if return_loss==True:
            loss = []
        for outer_i in range(max_iter):

            Q = K*P

            if use_path == False:
                v = torch.ones([n,], dtype=torch.float64)
                u = torch.ones([n,], dtype=torch.float64)


            for i in range(L):
                u = a1/torch.matmul(Q,v)
                v = a2/torch.matmul(Q.T,u)

            P = torch.unsqueeze(u,dim=1)*Q*torch.unsqueeze(v,dim=0)
            if return_loss==True:
                W = torch.sum(P*C) 
                loss.append(W)

        if return_loss==True:
            if return_map==True:
                return P, loss

            else:
                return loss

        else:
            if return_map==True:
                return P

            else:
                return None
            
    def _topk_intersection(self, scores1, scores2, topk, if_or=True):
        mask1 = scores1 != float('-inf')
        mask2 = scores2 != float('-inf')
        num_samples = scores1.shape[0]

        if (mask1.sum(dim=-1) > topk).all():
            values1, indices1 = torch.topk(scores1, topk)
            mask1 = torch.zeros_like(mask1)
            for i in range(num_samples):
                mask1[i, indices1[i]] = True        

        if ((mask2.sum(dim=-1)) > topk).all():
            values2, indices2 = torch.topk(scores2, topk)
            mask2 = torch.zeros_like(mask2)
            for j in range(num_samples):
                mask2[j, indices2[j]] = True

        if if_or:
            mask = torch.logical_or(mask1, mask2)
            s1 = torch.softmax(scores1[mask], dim=0)
            s2 = torch.softmax(scores2[mask], dim=0)
            indices1 = [[]]*num_samples
            for i in torch.nonzero(mask):
                indices1[i[0]].append(i[1])
            indices1 = torch.tensor(indices1)
            indices2 = indices1
        else:
            s1 = torch.softmax(scores1[mask1], dim=0)
            s2 = torch.softmax(scores2[mask2], dim=0)

        return s1, s2, indices1, indices2
    
    def _distances_fn(self, x):
        return sklearn.metrics.pairwise_distances(
                x, x[:1, :], metric='cosine').ravel() * 100
    
    def _kernel(self, d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    
    def generate_lime(self, inputs, generate_ids, p_len, num_round, vocab_size, each_round=5):
        num_tokens = generate_ids.shape[1]
        one_hot = np.zeros((each_round*num_round, generate_ids.shape[1], vocab_size), dtype=np.float32)
        for j in np.arange(p_len, num_tokens):
            one_hot[:, j, generate_ids[0,j].detach().cpu()]=1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        
        inputs_perturb = self._perturbe(inputs, num_samples=each_round*num_round, p_len=p_len)
        generate_ids_perturbe = transformers.tokenization_utils_base.BatchEncoding()
        generate_ids_perturbe['input_ids'] = torch.cat((inputs_perturb['input_ids'], generate_ids[:, p_len:].repeat(each_round*num_round,1)), dim=1)
        generate_ids_perturbe['attention_mask'] = torch.cat((inputs_perturb['attention_mask'], torch.ones((each_round*num_round, (generate_ids.shape[1]-p_len))).cuda()), dim=1)
        
        probs_all = torch.zeros(each_round*num_round, generate_ids.shape[1], vocab_size)
        with torch.no_grad():
            for i in range(num_round):
                logits = self.model.forward(generate_ids_perturbe.input_ids[(i*each_round):(each_round*i+each_round)], attention_mask=generate_ids_perturbe.attention_mask[(i*each_round):(each_round*i+each_round)])['logits']
                probs = torch.softmax(logits, dim=2).cpu().float()
                probs_all[(i*each_round):(each_round*i+each_round), :, :] = probs
        
        kernel_fn = partial(self._kernel, kernel_width=25)
        weights = kernel_fn(self._distances_fn(inputs_perturb.attention_mask.detach().cpu().numpy()))
  
        prob_target = torch.log(probs_all[one_hot.bool()]).reshape(each_round*num_round, -1).sum(dim=1)
        prob_target_scaled = (prob_target-prob_target.min())/ prob_target.var()**0.5

        model_g = Ridge(
                            alpha=0, 
                            fit_intercept=False, 
                            random_state=1234,
                            positive=True
                    )
        model_g.fit(inputs_perturb.attention_mask.cpu(), prob_target_scaled, sample_weight=weights)
        R = model_g.coef_

       
        return R
    
    def generate_bt_token_average(self, generate_ids, p_len, steps=1, start_layer=8):
        
        b =  generate_ids.shape[0]
        total_len = generate_ids.shape[1]
 
        blocks = eval(self.block_name)[start_layer:]
        R = torch.zeros(b, p_len).cpu()

        for i in np.arange(p_len, total_len):

            with torch.no_grad():
                logits = self.model.forward(generate_ids[:, :i])['logits']
                probs = torch.softmax(logits, dim=2).cpu().float()

            one_hot = np.zeros(probs.shape, dtype=np.float32)
            one_hot[0, -1, generate_ids[0][i].detach().cpu()] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            
            _, num_head, num_tokens, num_tokens = blocks[0].self_attn.get_attn().shape

            exp = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).cuda()

            for block in blocks:
                cam = block.self_attn.get_attn()
                z = block.get_input()
                vproj = block.self_attn.get_vproj()

                order = torch.linalg.norm(vproj, dim=-1).squeeze()/torch.linalg.norm(z, dim=-1).squeeze()
                m = torch.diag_embed(order).float()

                cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(0).float()

                exp = exp + torch.matmul(torch.matmul(cam.cuda(), m.cuda()), exp.cuda())
            
            exp = exp.detach().cpu()
            ig = torch.zeros(b, num_head, num_tokens, num_tokens).cpu()
            for step in range(steps+1):
                logits = self.model.forward(generate_ids[:, :i], alpha=(step/steps))['logits']
                probs = torch.softmax(logits, dim=2)
                loss = torch.sum(logits[one_hot.cuda().bool()])
                self.model.zero_grad()
                grad = torch.autograd.grad(loss, [blocks[-1].self_attn.attn_map])[0].detach().cpu()
                ig += grad
            intergrated_gradient = (ig/steps).mean(1).reshape(b, num_tokens, num_tokens)

            importances = (intergrated_gradient*exp).clamp(min=0).numpy()
            importance = importances[:, -1, :p_len]

            R = R + importance
        
        return R
    
    def generate_bt_token(self, generate_ids, p_len, steps=1, start_layer=8):
 
        blocks = eval(self.block_name)[start_layer:]
        with torch.no_grad():
            logits = self.model.forward(generate_ids)['logits']
            probs = torch.softmax(logits, dim=2).cpu().float()

        b, num_head, num_tokens, num_tokens = blocks[0].self_attn.get_attn().shape
        exp = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).cuda()

        one_hot = np.zeros(probs.shape, dtype=np.float32)
        for j in np.arange(p_len, num_tokens):
            one_hot[:, j, generate_ids[0,j].detach().cpu()]=1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        for block in blocks:
            cam = block.self_attn.get_attn()
            z = block.get_input()
            vproj = block.self_attn.get_vproj()

            order = torch.linalg.norm(vproj, dim=-1).squeeze()/torch.linalg.norm(z, dim=-1).squeeze()
            m = torch.diag_embed(order).float()

            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(0).float()

            exp = exp + torch.matmul(torch.matmul(cam.cuda(), m.cuda()), exp.cuda())

        ig = torch.zeros(b, num_head, num_tokens, num_tokens).cpu()
        for i in range(steps+1):
            logits = self.model.forward(generate_ids, alpha=(i/steps))['logits']
            probs = torch.softmax(logits, dim=2)
            loss = torch.log(probs[one_hot.bool()]).sum()
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, [blocks[-1].self_attn.attn_map])[0].detach().cpu()
            ig += grad
        intergrated_gradient = (ig/steps).mean(1).reshape(b, num_tokens, num_tokens)
        importances = (intergrated_gradient*exp.detach().cpu()).clamp(min=0).numpy()
        R = importances[:, (p_len-1):-1, :p_len].mean(1)

        
        return R
    
    def generate_ig_average(self, generate_ids, p_len, steps=1, start_layer=8, embed_dim=None):
        
        b =  generate_ids.shape[0]
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

            for step in np.arange(1,steps+1):
                logits = self.model.forward(input_ids=generate_ids[:, :i], alpha=(step/steps))['logits']
                probs = torch.softmax(logits, dim=2)
                loss = torch.log(probs[one_hot.bool()]).sum()
                self.model.zero_grad()
                grad = torch.autograd.grad(loss, [eval(self.embedding_name)])[0].detach()
                ig += grad
            intergrated_gradient = ig/steps
            importance = torch.sum(intergrated_gradient*eval(self.embedding_name), axis=-1).abs().detach().cpu().numpy()[:, :p_len]
            
            R = R + importance
        
        return R
    
    def generate_ig(self, generate_ids, p_len, steps=1, start_layer=8, embed_dim=None):
        
        with torch.no_grad():
            logits = self.model.forward(generate_ids)['logits']
            probs = torch.softmax(logits, dim=2).cpu().float()
        
        ig = torch.zeros(logits.shape[0], logits.shape[1], embed_dim).cuda()
        num_tokens = logits.shape[1]
        
        one_hot = np.zeros(probs.shape, dtype=np.float32)
        for j in np.arange(p_len, num_tokens):
            one_hot[:, j, generate_ids[0,j].detach().cpu()]=1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        for i in np.arange(1,steps+1):
            logits = self.model.forward(input_ids=generate_ids, alpha=(i/steps))['logits']
            probs = torch.softmax(logits, dim=2)
            loss = torch.log(probs[one_hot.bool()]).sum()
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, [eval(self.embedding_name)])[0].detach()
            ig += grad
        intergrated_gradient = ig/steps
        R = torch.sum(intergrated_gradient*eval(self.embedding_name), axis=-1).abs().detach().cpu().numpy()[:, :p_len]
        
        return R
    
    
    def generate_ours_one_step(self, inputs, topk=100, beta=1, if_or=True):
        
        with torch.no_grad():
            logits = self.model.forward(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        b, p_len, _ = logits.logits.shape
        res = np.zeros([b, p_len])

        for idx, i in enumerate(inputs.input_ids[0]):
            masked_inputs = transformers.tokenization_utils_base.BatchEncoding()
            masked_inputs['input_ids'] = inputs.input_ids.clone()
            masked_inputs['attention_mask'] = inputs.attention_mask.clone()
            masked_inputs['input_ids'][0][idx] = 0
            masked_inputs['attention_mask'][0][idx] = 0
            with torch.no_grad():
                logits_masked = self.model.forward(input_ids=masked_inputs.input_ids, attention_mask=masked_inputs.attention_mask)
            p1, p2, indices1, indices2 = self._topk_intersection(logits.logits[:, -1, :], logits_masked.logits[:, -1, :], \
                                                           topk, if_or=if_or)
            C = self._create_cost_matrix(indices1, indices2).cuda()

            P, loss = self._ipot_torch(p1.flatten(), p2.flatten(), C[0], 1)
            res[0, idx] = loss[-1]/p1.shape[-1]
            
        return res
    
    def generate_ours_more_steps(self, inputs, num_beams, steps, max_mask_token, vocab_size, topk=100, beta=1, if_or=True, concat=True): 
        _, p_len = inputs.input_ids.shape
        res = np.zeros([steps, p_len])

        batch = (p_len//max_mask_token)+(1-((p_len%max_mask_token)==0))
        all_batch = np.arange(batch+1)*max_mask_token
        all_batch[-1] = p_len
        
        print('p_len', p_len)
        print(all_batch)

        for b in range(batch):
            start, end = all_batch[b], all_batch[b+1]
            generated_inputs = transformers.tokenization_utils_base.BatchEncoding()
            generated_inputs['input_ids'] = inputs.input_ids.clone().repeat(num_beams, 1)
            generated_inputs['attention_mask'] = inputs.attention_mask.clone().repeat(num_beams, 1)
            with torch.no_grad():
                logits = self.model.forward(input_ids=generated_inputs.input_ids, attention_mask=generated_inputs.attention_mask, 
                               use_cache=True, output_hidden_states=True)
            probs = torch.softmax(logits.logits, dim=2)
            past_key_values = logits.past_key_values
            del logits
            top_v , top_indices = torch.topk(probs[:1, -1, :].flatten(), k=num_beams, dim=-1)
            seq_w = torch.softmax(top_v, dim=0).unsqueeze(0)
            out, past, probs_masked_pre = self._ot_calculate_batch_first(generated_inputs, probs[:1], start, end, p_len, num_beams, topk, vocab_size, if_or=if_or, beta=beta)
            res[0] += out

            for step in np.arange(1, steps):
                generated_inputs['input_ids'] = (top_indices%vocab_size).unsqueeze(-1)
                generated_inputs['attention_mask'] = torch.cat([generated_inputs['attention_mask'], torch.ones([num_beams, 1], device="cuda")], dim=-1)
                with torch.no_grad():
                    logits = self.model.forward(input_ids=generated_inputs.input_ids, attention_mask=generated_inputs.attention_mask, 
                               past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                probs = torch.cat([probs, torch.softmax(logits.logits, dim=2)], dim=1)
                if concat:
                    out, past, probs_masked_pre = self._ot_calculate_batch_more_concat(generated_inputs, probs, start, end, p_len, num_beams, topk, vocab_size, beta=beta, if_or=if_or, past=past, probs_previous=probs_masked_pre, seq_w=seq_w)
                else:
                    out, past, probs_masked_pre = self._ot_calculate_batch_more_avg(generated_inputs, probs, start, end, p_len, num_beams, topk, vocab_size, beta=beta, if_or=if_or, past=past, probs_previous=probs_masked_pre, seq_w=seq_w)
                res[step] += out

                top_v , top_indices = torch.topk(torch.log(probs[:, -(step+1):, :]).sum(dim=1).flatten(), k=num_beams, dim=-1)
                c = top_indices//vocab_size
                past_key_values = tuple(tuple(t[c] for t in i) for i in logits.past_key_values)
                past = tuple(tuple(t[c].repeat((end-start), 1, 1, 1) for t in i) for i in past)

                seq_w = torch.softmax(top_v, dim=0).unsqueeze(0)
                
        return res
    
    def generate_ours_more_steps_indices(self, inputs, indices, num_beams, steps, max_mask_token, vocab_size, topk=100, beta=1, if_or=True, concat=True): 
        _, p_len = inputs.input_ids.shape
        res = np.zeros([steps, p_len])

        generated_inputs = transformers.tokenization_utils_base.BatchEncoding()
        generated_inputs['input_ids'] = inputs.input_ids.clone().repeat(num_beams, 1)
        generated_inputs['attention_mask'] = inputs.attention_mask.clone().repeat(num_beams, 1)
        with torch.no_grad():
            logits = self.model.forward(input_ids=generated_inputs.input_ids, attention_mask=generated_inputs.attention_mask, 
                           use_cache=True, output_hidden_states=True)
        probs = torch.softmax(logits.logits, dim=2)
        past_key_values = logits.past_key_values
        del logits
        top_v , top_indices = torch.topk(probs[:1, -1, :].flatten(), k=num_beams, dim=-1)
        seq_w = torch.softmax(top_v, dim=0).unsqueeze(0)
        out, past, probs_masked_pre = self._ot_calculate_first_indices(generated_inputs, probs[:1], indices, p_len, num_beams, topk, vocab_size, if_or=if_or, beta=beta)
        res[0] += out

        for step in np.arange(1, steps):
            generated_inputs['input_ids'] = (top_indices%vocab_size).unsqueeze(-1)
            generated_inputs['attention_mask'] = torch.cat([generated_inputs['attention_mask'], torch.ones([num_beams, 1], device="cuda")], dim=-1)
            with torch.no_grad():
                logits = self.model.forward(input_ids=generated_inputs.input_ids, attention_mask=generated_inputs.attention_mask, 
                           past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            probs = torch.cat([probs, torch.softmax(logits.logits, dim=2)], dim=1)
            out, past, probs_masked_pre = self._ot_calculate_more_avg_indices(generated_inputs, probs, indices, p_len, num_beams, topk, vocab_size, beta=beta, if_or=if_or, past=past, probs_previous=probs_masked_pre, seq_w=seq_w)
            res[step] += out

            top_v , top_indices = torch.topk(torch.log(probs[:, -(step+1):, :]).sum(dim=1).flatten(), k=num_beams, dim=-1)
            c = top_indices//vocab_size
            past_key_values = tuple(tuple(t[c] for t in i) for i in logits.past_key_values)
            past = tuple(tuple(t[c].repeat((len(indices)), 1, 1, 1) for t in i) for i in past)

            seq_w = torch.softmax(top_v, dim=0).unsqueeze(0)
                
        return res

        
        
        
    
    
    
            
    
            
            
      