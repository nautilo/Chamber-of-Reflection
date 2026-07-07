import torch
import torch.nn.functional as F

def generar_texto_blindado(modelo, palabra_semilla, word_to_idx, idx_to_word, vocabulario, dim, num_capas, device, largo=60, temp=0.75, top_k=40, top_p=0.9, rep_penalty=1.3):
    modelo.eval()
    if palabra_semilla not in word_to_idx:
        palabra_semilla = vocabulario[0]
        
    lista_memorias_inf = [torch.zeros((1, dim), device=device) for _ in range(num_capas)]
    idx_target = word_to_idx[palabra_semilla]
    idx_actual = torch.tensor([idx_target], device=device)
    
    resultado = [palabra_semilla.capitalize()]
    historial = [idx_target]
    
    with torch.no_grad():
        for _ in range(largo):
            logits, lista_memorias_inf = modelo(idx_actual, lista_memorias_inf)
            logits = logits[0]
            
            for past_idx in set(historial):
                if logits[past_idx] > 0:
                    logits[past_idx] /= rep_penalty
                else:
                    logits[past_idx] *= rep_penalty
                    
            logits = logits / temp
            
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('Inf')
            
            probs_sort, probs_idx = torch.sort(logits, descending=True)
            probs_acumuladas = torch.cumsum(F.softmax(probs_sort, dim=-1), dim=-1)
            mascara_remover = probs_acumuladas > top_p
            mascara_remover[1:] = mascara_remover[:-1].clone()
            mascara_remover[0] = False
            probs_sort[mascara_remover] = -float('Inf')
            logits.scatter_(0, probs_idx, probs_sort)
            
            probs_finales = F.softmax(logits, dim=0)
            idx_siguiente = torch.multinomial(probs_finales, num_samples=1).item()
            
            historial.append(idx_siguiente)
            resultado.append(idx_to_word[idx_siguiente])
            idx_actual = torch.tensor([idx_siguiente], device=device)
            
    return ">>> " + " ".join(resultado) + "..."
