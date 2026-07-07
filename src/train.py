import torch
import torch.nn.functional as F
import urllib.request
import zipfile
import os
from collections import Counter

from config import Config, device
from model import ChamberOfReflectionModel
from utils import generar_texto_blindado

if __name__ == '__main__':
    print("="*70)
    print(f"🧠 ENGINE: CHAMBER OF REFLECTION V4.0 (MASTER BUILD)")
    print("="*70)
    print(f"✨ Dispositivo: {device.upper()}")

    url = "http://mattmahoney.net/dc/text8.zip"
    file_name = "../dataset/text8.zip"
    os.makedirs("../dataset", exist_ok=True)
    
    if not os.path.exists(file_name):
        print("📥 Descargando corpus original...")
        urllib.request.urlretrieve(url, file_name)

    with zipfile.ZipFile(file_name, 'r') as z:
        texto_crudo = z.read(z.namelist()[0]).decode('utf-8')

    palabras_crudas = texto_crudo.split()[:Config.DATA_LIMIT]
    conteo = Counter(palabras_crudas)
    vocabulario = [w for w, c in conteo.most_common(Config.VOCAB_SIZE)]

    word_to_idx = {w: i for i, w in enumerate(vocabulario)}
    idx_to_word = {i: w for i, w in enumerate(vocabulario)}
    V_SIZE = len(vocabulario)

    data_words = torch.tensor([word_to_idx[p] for p in palabras_crudas if p in word_to_idx], dtype=torch.long)
    n_batches = len(data_words) // (Config.BATCH_SIZE * Config.SEQ_LEN)
    data_batched = data_words[:n_batches * Config.BATCH_SIZE * Config.SEQ_LEN].view(Config.BATCH_SIZE, -1).to(device)

    modelo = ChamberOfReflectionModel(V_SIZE, Config.EMBED_DIM, Config.NUM_CAPAS).to(device)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=Config.LR, weight_decay=0.01)
    
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    print("\n🔥 INICIANDO FORJA PROFUNDA...")
    for epoca in range(Config.EPOCAS):
        lista_memorias = [torch.zeros((Config.BATCH_SIZE, Config.EMBED_DIM), device=device) for _ in range(Config.NUM_CAPAS)]
        modelo.train()
        
        for i in range(0, data_batched.size(1) - 1, Config.SEQ_LEN):
            optimizador.zero_grad(set_to_none=True)
            loss_bloque = 0
            lista_memorias = [m.detach() for m in lista_memorias]
            
            with torch.autocast('cuda', enabled=(device == 'cuda')):
                for t in range(Config.SEQ_LEN):
                    if i + t + 1 >= data_batched.size(1): break
                    idx_actual = data_batched[:, i + t]
                    idx_target = data_batched[:, i + t + 1]
                    
                    logits, lista_memorias = modelo(idx_actual, lista_memorias)
                    loss = F.cross_entropy(logits, idx_target)
                    loss_bloque += loss
            
            if scaler:
                scaler.scale(loss_bloque).backward()
                scaler.unscale_(optimizador)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                scaler.step(optimizador)
                scaler.update()
            else:
                loss_bloque.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                optimizador.step()
                
        loss_final = loss_bloque.item()/Config.SEQ_LEN
        print(f"✅ Época {epoca+1:02d}/{Config.EPOCAS:02d} completada | Loss: {loss_final:.4f}")
        
        if (epoca + 1) % 5 == 0:
            torch.save(modelo.state_dict(), f"checkpoint_ep{epoca+1}.pt")
            print(f"   💾 Checkpoint guardado: checkpoint_ep{epoca+1}.pt")

    torch.save(modelo.state_dict(), Config.SAVE_PATH)
    print(f"🏆 Entrenamiento finalizado. Modelo guardado en {Config.SAVE_PATH}")

    print("\n" + "="*70)
    print("🗣️ RESULTADO DE LA CÁMARA BLINDADA:")
    print("="*70)
    print(generar_texto_blindado(modelo, "history", word_to_idx, idx_to_word, vocabulario, Config.EMBED_DIM, Config.NUM_CAPAS, device))
