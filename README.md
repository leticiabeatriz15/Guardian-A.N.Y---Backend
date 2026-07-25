# 🧩 Guardian A.N.Y — Backend API

> **Guardian A.N.Y.** (*Autista No YouTube*) é uma ferramenta desenvolvida para analisar vídeos do YouTube e classificar seu nível de estimulação sensorial (auditiva/visual) usando inteligência artificial, ajudando a identificar conteúdos adequados ou hiperestimulantes para pessoas no espectro autista (TEA).

---

## 📌 Visão Geral

Esta API é construída com **FastAPI** e atua no processamento pesado da aplicação:
1. Recebe a URL de um vídeo do YouTube enviada pelo frontend.
2. Utiliza o `yt-dlp` para capturar a stream de vídeo de forma otimizada.
3. Extrai métricas visuais e de movimento usando **OpenCV**:
   * **Similaridade Visual**: Variação das cores entre frames sequenciais.
   * **Contraste Cromático**: Intensidade de cor no espaço LAB.
   * **Agitação / Movimento**: Quantidade de mudança física/cortes entre cenas.
4. Processa as métricas através de um modelo de Machine Learning (**RandomForestClassifier**) treinado com `scikit-learn`.
5. Retorna a classificação do nível de estimulação para a interface.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.12**
* **FastAPI**: Framework web de alta performance.
* **OpenCV (`opencv-python-headless`)**: Processamento de imagem e extração de frames.
* **yt-dlp**: Manipulação e extração de streams de vídeo.
* **Scikit-Learn & Pandas**: Carregamento do modelo `.pkl` e predição.
* **Docker**: Containerização do ambiente para deploy.

---

## 🚀 Como Executar Localmente

### Pré-requisitos
* Python 3.10+ instalado.
* FFmpeg instalado no sistema.

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/leticiabeatriz15/Guardian-A.N.Y---Backend.git](https://github.com/leticiabeatriz15/Guardian-A.N.Y---Backend.git)
   cd Guardian-A.N.Y---Backend

```

2. **Crie e ative um ambiente virtual:**
```bash
python -m venv venv
# No Linux/Mac:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

```


3. **Instale as dependências:**
```bash
pip install -r requirements.txt

```


4. **Certifique-se de que o modelo treinado está no diretório:**
O arquivo `modelo_classificador_video.pkl` deve estar localizado na raiz do projeto.
5. **Inicie o servidor de desenvolvimento:**
```bash
uvicorn main:app --reload --port 8000

```


A API estará acessível em `http://localhost:8000` (documentação Swagger disponível em `/docs`).

---

## 🐳 Running com Docker

Para rodar a aplicação em um container local idêntico ao ambiente de produção:

```bash
docker build -t guardian-backend .
docker run -p 8000:8000 guardian-backend

```

---

## ☁️ Deploy

A API está configurada para deploy automatizado na plataforma **Render** utilizando o `Dockerfile` com suporte a bibliotecas nativas de mídia e bypassing de restrições do YouTube.