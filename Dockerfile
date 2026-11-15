# Node 24 + FFmpeg, for Cloud Run (and local Docker).
FROM node:24-bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg python3 python3-pip \
 && pip install --break-system-packages edge-tts \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --omit=dev
COPY . .
# Cloud Run sets PORT (default 8080). No secrets baked into the image.
ENV PORT=8080
EXPOSE 8080
CMD ["node", "server.js"]
