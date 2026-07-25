import os
import cv2
import yt_dlp
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Guardian A.N.Y API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CAMINHO_MODELO = 'modelo_classificador_video.pkl'
if os.path.exists(CAMINHO_MODELO):
    modelo = joblib.load(CAMINHO_MODELO)
    print("✅ Modelo de IA carregado com sucesso!")
else:
    modelo = None
    print("⚠️ Atenção: Modelo de IA não encontrado!")

class AnaliseRequest(BaseModel):
    url: str

def normalizar_url_youtube(url: str) -> str:
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    if "watch?v=" in url:
        video_id = url.split("watch?v=")[1].split("&")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def extrair_metricas_e_titulo(url_video: str):
    url_norm = normalizar_url_youtube(url_video)
    print(f"\n[1/4] Extraindo stream via yt-dlp: {url_norm}")
    
    ydl_opts = {
        'format': 'worst[ext=mp4]/worst/b', 
        'cookiefile': 'cookies.txt',
        'quiet': True,
        'no_warnings': True,        
        'extractor_args': {
            'youtube': {
                'player_client': ['web', 'tv_embedded']
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        }
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_norm, download=False)
            
        if not info:
            print("❌ Falha: Nenhuma informação extraída do vídeo.")
            return None, None

        stream_url = info.get("url")
        
        if not stream_url and "formats" in info:
            for f in info["formats"]:
                if f.get("url") and f.get("vcodec") != "none":
                    stream_url = f["url"]
                    break

        if not stream_url:
            print("❌ Falha: Nenhuma URL de stream jogável encontrada.")
            return None, None
            
        titulo = info.get("title", "Vídeo do YouTube")
        print(f"[2/4] Stream obtida com sucesso: {titulo}")
    except Exception as e:
        print(f"❌ Erro no yt-dlp: {e}")
        return None, None

    print("[3/4] Abrindo stream no OpenCV...")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("❌ Erro ao abrir a stream de vídeo no OpenCV.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 24.0

    intervalo_segundos = 0.5
    frame_intervalo = max(1, int(fps * intervalo_segundos))
    
    frame_inicial = int(fps * 5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicial)

    hist_count = 0
    max_amostras = 60
    
    valores_similaridade = []
    valores_contrastes = []
    valores_movimento = []
    hists_anteriores = None
    frame_anterior_gray = None

    frame_atual_idx = 0
    print(f"[4/4] Processando {max_amostras} amostras de frames...")

    while hist_count < max_amostras:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_atual_idx % frame_intervalo == 0:
            if frame is None or frame.size == 0:
                frame_atual_idx += 1
                continue

            frame_pequeno = cv2.resize(frame, (320, 240))
            frame_rgb = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2GRAY)

            if frame_anterior_gray is not None:
                diff = cv2.absdiff(frame_anterior_gray, frame_gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                movimento_score = (np.sum(thresh == 255) / thresh.size) * 100
                valores_movimento.append(movimento_score)
            else:
                valores_movimento.append(0.0)

            frame_anterior_gray = frame_gray.copy()

            frame_lab = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2Lab)
            _, a_channel, b_channel = cv2.split(frame_lab)
            contraste_frame = np.sqrt(np.std(a_channel)**2 + np.std(b_channel)**2)
            valores_contrastes.append(contraste_frame)

            hists_atuais = {}
            correlacoes = []
            for i, cor in enumerate(["red", "green", "blue"]):
                hist = cv2.calcHist([frame_rgb], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hists_atuais[cor] = hist

                if hists_anteriores is not None:
                    correlacoes.append(cv2.compareHist(hists_anteriores[cor], hist, cv2.HISTCMP_CORREL))

            if correlacoes:
                valores_similaridade.append(sum(correlacoes) / len(correlacoes))

            hists_anteriores = hists_atuais
            hist_count += 1

        frame_atual_idx += 1

    cap.release()

    if hist_count == 0:
        print("❌ Nenhum frame foi processado com sucesso.")
        return None, None

    metricas = {
        "Similaridade": sum(valores_similaridade) / len(valores_similaridade) if valores_similaridade else 0,
        "Contraste": sum(valores_contrastes) / len(valores_contrastes) if valores_contrastes else 0,
        "Agitacao": sum(valores_movimento) / len(valores_movimento) if valores_movimento else 0
    }

    print("✨ Extração de métricas concluída!")
    return metricas, titulo

@app.post("/analisar")
def analisar_video(dados: AnaliseRequest):
    if not modelo:
        raise HTTPException(status_code=500, detail="Modelo de IA não carregado no servidor.")

    metricas, titulo = extrair_metricas_e_titulo(dados.url)

    if not metricas:
        raise HTTPException(status_code=400, detail="Não foi possível processar o vídeo. Verifique a URL.")

    X_input = pd.DataFrame([{
        'Similaridade': metricas['Similaridade'],
        'Contraste': metricas['Contraste'],
        'Agitacao': metricas['Agitacao']
    }])

    predicao = modelo.predict(X_input)[0]

    return {
        "titulo": titulo,
        "url": dados.url,
        "classificacao": str(predicao),
        "metricas": metricas
    }