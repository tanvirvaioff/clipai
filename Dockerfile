FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core fontconfig && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Install Montserrat Black to system fonts so libass always finds it
RUN mkdir -p /usr/share/fonts/truetype/montserrat && cp fonts/Montserrat-Black.ttf /usr/share/fonts/truetype/montserrat/ && fc-cache -f
CMD ["gunicorn", "app:app", "-c", "gunicorn.conf.py"]
# force rebuild 1792242344
