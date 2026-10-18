FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core fontconfig libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# ensure cv2 resolves to the contrib build (ultralytics pulls opencv-python which would collide in site-packages/cv2)
RUN pip install --no-cache-dir --force-reinstall --no-deps opencv-contrib-python==4.10.0.84
# pre-download YOLOv8n weights (~6MB) so the first job doesn't stall on download
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
COPY . .
# Install Montserrat Black to system fonts so libass always finds it
RUN mkdir -p /usr/share/fonts/truetype/montserrat && cp fonts/Montserrat-Black.ttf /usr/share/fonts/truetype/montserrat/ && fc-cache -f
CMD ["gunicorn", "app:app", "-c", "gunicorn.conf.py"]
# force rebuild 1792464000
