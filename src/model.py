import torch
import torch.nn as nn

class ReflectionChamber(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_x = nn.Linear(dim, dim, bias=False)
        self.gate_h = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, h):
        x_norm = self.norm(x)
        h_norm = self.norm(h)
        g = torch.sigmoid(self.gate_x(x_norm) + self.gate_h(h_norm))
        h_nueva = (1.0 - g) * h + g * x_norm
        return self.norm(h_nueva), h_nueva

class ChamberOfReflectionModel(nn.Module):
    def __init__(self, v_size, dim, num_capas):
        super().__init__()
        self.W_SEMANTICA = nn.Embedding(v_size, dim)
        self.W_SINTAXIS = nn.Embedding(v_size, dim)
        nn.init.normal_(self.W_SEMANTICA.weight, std=0.02)
        nn.init.normal_(self.W_SINTAXIS.weight, std=0.02)
        
        self.capas = nn.ModuleList([ReflectionChamber(dim) for _ in range(num_capas)])
        self.num_capas = num_capas
        self.lm_head = nn.Linear(dim, v_size, bias=False)
        self.lm_head.weight = self.W_SEMANTICA.weight 

    def forward(self, idx_batch, lista_memorias):
        x = self.W_SEMANTICA(idx_batch) + self.W_SINTAXIS(idx_batch)
        nuevas_memorias = []
        for i in range(self.num_capas):
            x, h_nueva = self.capas[i](x, lista_memorias[i])
            nuevas_memorias.append(h_nueva)
        logits = self.lm_head(x)
        return logits, nuevas_memorias
