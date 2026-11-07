FROM node:22-alpine AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py config.py ./
COPY backend/ ./backend/
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist/

RUN mkdir -p /app/db/events

EXPOSE 5000
CMD ["python", "app.py"]
