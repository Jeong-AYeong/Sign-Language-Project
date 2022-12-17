FROM python:3.10.8-bullseye
EXPOSE 8000
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN chmod +x run.sh
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
CMD ["./run.sh"]