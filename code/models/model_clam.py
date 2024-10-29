import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.AM_Former import FTTransformer
import numpy as np
import pdb

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""

class BoostingModel(nn.Module):
    def __init__(self, modalities, boosting_rounds=3, **kwargs):
        super().__init__()
        self.modalities = modalities
        self.boosting_rounds = boosting_rounds
        self.models = nn.ModuleDict()
        for modality in modalities:
            self.models[modality] = CLAM_SB(modality=modality, **kwargs)
        self.alpha = nn.Parameter(torch.ones(boosting_rounds))  # Trainable weight for each boosting round

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        logits_list = []
        attentions_list = []
        results_dict_list = []

        for round_idx in range(self.boosting_rounds):
            modality = self.modalities[round_idx % len(self.modalities)]
            model = self.models[modality]

            logits, Y_prob, Y_hat, A_raw, results_dict = model(
                h, label, instance_eval, return_features, attention_only
            )

            logits_list.append(logits * self.alpha[round_idx])
            attentions_list.append(A_raw)
            results_dict_list.append(results_dict)

        alpha_sum = torch.sum(self.alpha)
        normalized_alpha = self.alpha / alpha_sum

        combined_logits = torch.stack([logits * normalized_alpha[i] for i, logits in enumerate(logits_list)], dim=0).sum(dim=0)
        combined_probs = F.softmax(combined_logits, dim=1)
        combined_Y_hat = torch.argmax(combined_logits, dim=1)

        combined_results_dict = {
            'boosting_results': results_dict_list
        }

        return combined_logits, combined_probs, combined_Y_hat, attentions_list, combined_results_dict

class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import einsum

class FormerAttention(nn.Module):
    def __init__(self, query_dim=512, context_dim=None, heads=8, dim_head=64, dropout=False):
        super(FormerAttention, self).__init__()
        
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        # Scaling factor for dot-product attention
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        # Query, Key, and Value projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        
        # Final output layer after attention
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim)
        )
        
        # Dropout option
        if dropout:
            self.dropout = nn.Dropout(0.25)
        else:
            self.dropout = None
    
    def forward(self, x, context=None, mask=None):
        h = self.heads

        # Query projection
        q = self.to_q(x)

        # Use context if provided, otherwise use x for self-attention
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        # Reshape queries, keys, and values for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Dot product attention
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # Apply mask if provided
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # Apply softmax to attention scores
        attn = sim.softmax(dim=-1)

        # Compute attention-weighted output
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # Final output projection
        out = self.to_out(out)

        # Apply dropout if specified
        if self.dropout:
            out = self.dropout(out)

        return out


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------------#
# SumFusion???,???????????????
#------------------------------------------#
class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        #---------------------------------------#
        # ??x??y??????,???????????
        #---------------------------------------#
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return output

#------------------------------------------#
# ConcatFusion???,?????????
# ???????,?????????????????
#------------------------------------------#
class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = x
        output = self.fc_out(output)
        return output

#------------------------------------------#
# FiLM???????,?????????
#------------------------------------------#
class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """
    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=False):
        super(FiLM, self).__init__()
        self.dim    = input_dim
        self.fc     = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)
        self.x_film = x_film

    def forward(self, x, y):
        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x
        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return output

#------------------------------------------#
# GatedFusion definition
#------------------------------------------#
class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """
    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()
        self.fc_x    = nn.Linear(input_dim, dim)
        self.fc_y    = nn.Linear(input_dim, dim)
        self.fc_out  = nn.Linear(dim, output_dim)
        self.x_gate  = x_gate  # whether to choose the x to obtain the gate
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate   = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate   = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return output


class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, num_cate=56, embed_dim=1024, modality="unimodal",
                 clinical_dim=59, return_probs=False):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
            # attention_net = FormerAttention(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # Adding fusion methods
        if modality == 'sum_fusion':
            self.tabular_fc = nn.Linear(clinical_dim, size[0])
            self.fusion = SumFusion(input_dim=size[0], output_dim=size[0])
            self.classifiers = nn.Linear(size[1], n_classes)
        elif modality == 'concat_fusion':
            self.fusion = ConcatFusion(input_dim=size[1] + clinical_dim, output_dim=size[1])
            self.classifiers = nn.Linear(size[1], n_classes)
        elif modality == 'FiLM':
            self.tabular_fc = nn.Linear(clinical_dim, size[1])
            self.fusion = FiLM(input_dim=size[1], dim=size[1], output_dim=size[1])
            self.classifiers = nn.Linear(size[1], n_classes)
        elif modality == 'gated_fusion':
            self.tabular_fc = nn.Linear(clinical_dim, size[1])
            self.fusion = GatedFusion(input_dim=size[1], dim=size[1], output_dim=size[1])
            self.classifiers = nn.Linear(size[1], n_classes)
        elif modality == 'fusion':
            self.classifiers = nn.Sequential(
                nn.Linear(size[1] + clinical_dim, (size[1] + clinical_dim) // 4),
                nn.LayerNorm((size[1] + clinical_dim) // 4),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear((size[1] + clinical_dim) // 4, (size[1] + clinical_dim) // 16),
                nn.LayerNorm((size[1] + clinical_dim) // 16),
                nn.ReLU(),
                nn.Linear((size[1] + clinical_dim) // 16, n_classes)
            )
        # elif modality == 'Former':
        #     image_feature_dim = size[1]
        #     tabular_feature_dim = 192
        #     d_model = 256
        #     n_heads = 8
            
        #     self.image_fc = nn.Linear(image_feature_dim, d_model)
        #     self.tabular_fc = nn.Linear(tabular_feature_dim, d_model)
            
        #     self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        #     self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
            
        #     self.layer_norm = nn.LayerNorm(d_model)
        #     self.classifiers = nn.Sequential(
        #         nn.Linear(d_model, d_model // 4),
        #         nn.LayerNorm(d_model // 4),
        #         nn.GELU(),
        #         nn.Linear(d_model // 4, n_classes)
        #     )
        #     self.tabular_encoder = FTTransformer()
        elif modality == "product_fusion":
            self.classifiers = nn.Linear(size[1], n_classes)
            self.tabular_fc = nn.Linear(clinical_dim, size[1])
        elif modality == "multimodal":
            self.classifiers = nn.Linear(size[1] + clinical_dim, n_classes)
        elif modality == "unimodal":
            self.classifiers = nn.Linear(size[1], n_classes)

        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.modality = modality
        self.embed_dim = embed_dim
        self.num_cate = num_cate
        self.clinical_dim = clinical_dim
        self.model_type = 'boosting'
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()
    
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        if self.modality != "unimodal"  and self.modality != 'concat_fusion':
            h, c = torch.split(h, [self.embed_dim, self.clinical_dim], dim=1)
            c = c[0].reshape([1, c[0].shape[0]])
        # if self.modality == 'sum_fusion':
        #     c = tabular_feature = self.tabular_fc(c)
        #     h = self.fusion(h, c)
        # elif self.modality == 'concat_fusion':
        #     h = self.fusion(h)
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)

        
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        
        M = torch.mm(A, h)        
        
        if self.modality != "unimodal" and self.modality != 'Former' and self.modality != "tabnet" and self.modality != 'sum_fusion' and self.modality != 'concat_fusion' and self.modality != 'gated_fusion' and self.modality != "FiLM":
            M = torch.cat((M, c), 1)
        elif self.modality == "product_fusion":
            tabular_feature = self.tabular_fc(c)
            M = M * tabular_feature
        elif self.modality == "FiLM" or self.modality == 'gated_fusion':
            c = tabular_feature = self.tabular_fc(c)
            M = self.fusion(M, c)
        elif self.modality == 'concat_fusion':
            M = self.fusion(M)
        elif self.modality == 'Former':
            c_converted_dtype = c.clone()
            categorical_columns_list = []

            for i in range(self.num_cate):
                column_data = c[:, i]
                if column_data.dtype != torch.long:
                    column_data = column_data.long()
                categorical_columns_list.append(column_data.unsqueeze(1))

            categorical_features = torch.cat(categorical_columns_list, dim=1)
            continuous_features = c[:, self.num_cate:]
            tabular_feature = self.tabular_encoder(categorical_features, continuous_features)   
            tabular_feature = self.tabular_fc(tabular_feature)
            image_features = self.image_fc(M)
            M = tabular_feature * image_features

        logits = self.classifiers(M)
        if self.modality == 'sum_fusion':
            logits += c
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,modality="unimodal",clinical_dim=56,return_probs=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
