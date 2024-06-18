import torch
import torch.nn as nn
import torch.nn.functional as F

class MIL_fc(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1,
                 embed_dim=1024,modality=None,clinical_dim=0,return_probs=False):
        super().__init__()
        assert n_classes == 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifier=  nn.Linear(size[1], n_classes)
        self.top_k=top_k

    def forward(self, h, return_features=False):
        h = self.fc(h)
        logits  = self.classifier(h) # K x 2
        
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1, embed_dim=1024,modality=None,clinical_dim=0,return_probs=False):
        super().__init__()
        assert n_classes > 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1
    
    def forward(self, h, return_features=False):       
        h = self.fc(h)
        logits = self.classifiers(h)

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict
    


class MLP(nn.Module):
    def __init__(self, size_arg = None, dropout = 0., n_classes = 2, top_k=1, embed_dim=59,modality=None,clinical_dim=0,return_probs=False):
        super(MLP, self).__init__()
        self.n_classes = n_classes
        hidden_sizes = []
        temp_size = int(embed_dim/2)
        for i in range(3):
            hidden_sizes.append(temp_size)
            temp_size = int(temp_size/2)
        layers = [nn.Linear(embed_dim, hidden_sizes[0]), nn.ReLU(), nn.Dropout(dropout)]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], n_classes))
        
        self.classifier = nn.Sequential(*layers)
        self.return_probs = return_probs

    def forward(self, x,return_features=False):
        logits = self.classifier(x)
        y_probs = F.softmax(logits, dim = 1)    
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]


        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]

        results_dict = {}

        if return_features:
            top_features = torch.index_select(logits, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        if self.return_probs:
            return Y_prob
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


        
