import os
import re
import tempfile
import requests
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

POSSIVEIS_CAMINHOS_COOKIES = [
    "cookies.txt",
    "/opt/render/project/src/cookies.txt",
    os.path.join(os.getcwd(), "cookies.txt")
]

COOKIE_PATH = None

if os.environ.get("YOUTUBE_COOKIES"):
    raw_cookies = os.environ["YOUTUBE_COOKIES"].replace('\\n', '\n')
    line_list = []
    for line in raw_cookies.splitlines():
        if line.strip().startswith("#") or not line.strip():
            line_list.append(line)
        else:
            fixed_line = re.sub(r'[\t ]{2,}', '\t', line.strip())
            line_list.append(fixed_line)
            
    fixed_cookie_content = "\n".join(line_list)
    COOKIE_PATH = "cookies.txt"
    with open(COOKIE_PATH, "w", encoding="utf-8") as f:
        f.write(fixed_cookie_content)
    print(f"🍪 [Env Var] Cookies injetados e salvos com sucesso! ({len(fixed_cookie_content)} bytes)")

if not COOKIE_PATH:
    for caminho in POSSIVEIS_CAMINHOS_COOKIES:
        if os.path.exists(caminho) and os.path.getsize(caminho) > 0:
            COOKIE_PATH = caminho
            print(f"🍪 [Secret File / File System] Encontrado cookie em: {caminho} ({os.path.getsize(caminho)} bytes)")
            break

if not COOKIE_PATH:
    print("⚠️ CRÍTICO: Nenhum arquivo de cookies do YouTube foi encontrado no servidor!")


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
        # 'b' sozinho já é bem permissivo, mas deixamos os filtros como preferência
        # e caímos para QUALQUER formato (mesmo só-áudio) antes de desistir.
        'format': 'worst[ext=mp4][has_drm=false]/worst[has_drm=false]/b/best',
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'extractor_args': {
            'youtube': {
                # android/ios costumam devolver formatos progressivos sem exigir
                # PO token com a mesma frequência que web/mweb em 2026.
                'player_client': ['android', 'ios', 'tv_embedded', 'mweb', 'web_creator', 'web']
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        }
    }

    if COOKIE_PATH and os.path.exists(COOKIE_PATH):
        ydl_opts['cookiefile'] = COOKIE_PATH
        print(f"👉 yt-dlp executando COM cookiefile: {COOKIE_PATH}")
    else:
        print("⚠️ ALERTA: yt-dlp executando SEM arquivo de cookies!")

    # O Render usa IPs de datacenter compartilhados, que o YouTube costuma
    # marcar como suspeitos mesmo com cookies válidos ("Sign in to confirm
    # you're not a bot"). Configure YTDLP_PROXY (ex: um proxy residencial)
    # nas env vars do Render para contornar isso quando necessário.
    proxy_url = os.environ.get("YTDLP_PROXY")
    if proxy_url:
        ydl_opts['proxy'] = proxy_url
        print("🌐 yt-dlp executando através de proxy configurado (YTDLP_PROXY).")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_norm, download=False)
            
        if not info:
            print("❌ Falha: Nenhuma informação extraída do vídeo.")
            return None, None, "Não foi possível extrair informações do vídeo."

        stream_url = info.get("url")

        if not stream_url and "formats" in info:
            for f in info["formats"]:
                if f.get("url") and f.get("vcodec") != "none" and not f.get("has_drm"):
                    stream_url = f["url"]
                    break

        if not stream_url:
            formatos = info.get("formats", [])
            print(f"❌ Falha: Nenhuma URL de stream sem DRM encontrada. "
                  f"Total de formatos retornados pelo yt-dlp: {len(formatos)}")
            for f in formatos[:10]:
                print(f"   • id={f.get('format_id')} ext={f.get('ext')} "
                      f"vcodec={f.get('vcodec')} has_drm={f.get('has_drm')} url_ok={bool(f.get('url'))}")
            return None, None, "Este vídeo é protegido por direitos autorais (DRM) e não permite transmissão direta."
            
        titulo = info.get("title", "Vídeo do YouTube")
        print(f"[2/4] Stream obtida com sucesso: {titulo}")
    except Exception as e:
        erro_str = str(e)
        print(f"❌ Erro no yt-dlp: {erro_str}")
        if "DRM" in erro_str:
            return None, None, "Este vídeo possui proteção contra cópia (DRM) e não pode ser analisado."
        if "Sign in to confirm" in erro_str or "not a bot" in erro_str:
            print("🤖 Bloqueio de bot detectado pelo YouTube (provável reputação do IP do servidor). "
                  "Cookies sozinhos podem não resolver — considere configurar YTDLP_PROXY.")
            return None, None, "O YouTube bloqueou esta requisição por suspeita de bot. Tente novamente em instantes."
        return None, None, f"Erro ao acessar vídeo: {erro_str}"

    print("[3/4] Baixando stream para arquivo temporário...")
    # O backend FFmpeg do OpenCV não usa o proxy configurado no yt-dlp, então
    # abrir a URL direto (cv2.VideoCapture(stream_url)) sai pelo IP "sujo" do
    # Render e falha. Por isso baixamos os bytes aqui, passando pelo mesmo
    # proxy, e entregamos um arquivo local pro OpenCV.
    caminho_temp = None
    try:
        req_proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
        headers = ydl_opts['http_headers']
        with requests.get(stream_url, proxies=req_proxies, headers=headers,
                           stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                caminho_temp = tmp.name
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    tmp.write(chunk)
        print(f"[3/4] Download concluído: {os.path.getsize(caminho_temp)} bytes.")
    except Exception as e:
        print(f"❌ Erro ao baixar a stream: {e}")
        if caminho_temp and os.path.exists(caminho_temp):
            os.remove(caminho_temp)
        return None, None, "Erro ao baixar o fluxo de vídeo."

    print("[3/4] Abrindo arquivo no OpenCV...")
    cap = cv2.VideoCapture(caminho_temp)
    if not cap.isOpened():
        print("❌ Erro ao abrir a stream de vídeo no OpenCV.")
        cap.release()
        os.remove(caminho_temp)
        return None, None, "Erro ao abrir o fluxo de vídeo no OpenCV."

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
    os.remove(caminho_temp)

    if hist_count == 0:
        print("❌ Nenhum frame foi processado com sucesso.")
        return None, None, "Falha no processamento dos quadros do vídeo."

    metricas = {
        "Similaridade": sum(valores_similaridade) / len(valores_similaridade) if valores_similaridade else 0,
        "Contraste": sum(valores_contrastes) / len(valores_contrastes) if valores_contrastes else 0,
        "Agitacao": sum(valores_movimento) / len(valores_movimento) if valores_movimento else 0
    }

    print("✨ Extração de métricas concluída!")
    return metricas, titulo, None


@app.post("/analisar")
def analisar_video(dados: AnaliseRequest):
    if not modelo:
        raise HTTPException(status_code=500, detail="Modelo de IA não carregado no servidor.")

    metricas, titulo, erro_msg = extrair_metricas_e_titulo(dados.url)

    if erro_msg or not metricas:
        raise HTTPException(
            status_code=400, 
            detail=erro_msg or "Não foi possível processar o vídeo. Verifique a URL."
        )

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