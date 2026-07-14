"""
Como usar:
1. Coloque suas fotos numa pasta chamada "fotos/", numeradas em ordem
   (foto_01.jpg, foto_02.jpg, ...).
2. Rode: python main.py
3. Veja o resultado em resultado_final.jpg
"""

import cv2
import numpy as np
import glob


# ---------------------------------------------------------------
# PASSO 1: Carregar todas as fotos da pasta
# ---------------------------------------------------------------

caminhos = sorted(glob.glob("fotos/*.jpg"))
print(f"Encontradas {len(caminhos)} fotos.")

imagens = [cv2.imread(c) for c in caminhos]


# ---------------------------------------------------------------
# PASSO 2: Para cada foto, medir o quão nítido está cada pixel
# ---------------------------------------------------------------
# O operador Laplaciano detecta mudanças bruscas de brilho (bordas).
# Onde há uma borda nítida, o valor é alto. Onde a imagem está
# borrada, o valor fica perto de zero.

mapas_de_nitidez = []

for img in imagens:
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplaciano = cv2.Laplacian(cinza, cv2.CV_64F, ksize=5)
    mapas_de_nitidez.append(np.abs(laplaciano))


# ---------------------------------------------------------------
# PASSO 3: Para cada pixel, descobrir em qual foto ele está mais nítido
# ---------------------------------------------------------------

pilha_nitidez = np.array(mapas_de_nitidez)          # formato: (N_fotos, altura, largura)
indice_vencedor = np.argmax(pilha_nitidez, axis=0)   # formato: (altura, largura)


# ---------------------------------------------------------------
# PASSO 4: Montar a imagem final, pegando a cor certa de cada foto
# ---------------------------------------------------------------

pilha_imagens = np.array(imagens)
resultado = np.take_along_axis(
    pilha_imagens,
    indice_vencedor[np.newaxis, :, :, np.newaxis],
    axis=0
)[0]

resultado = resultado.astype(np.uint8)


# ---------------------------------------------------------------
# PASSO 5: Salvar
# ---------------------------------------------------------------

cv2.imwrite("resultado_final.jpg", resultado)
print("Pronto! Veja resultado_final.jpg")



