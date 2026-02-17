FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential

WORKDIR /app

COPY . .

RUN chmod +x setup.sh

CMD ["bash"]