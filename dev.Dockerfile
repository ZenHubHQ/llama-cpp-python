# Define the image argument and provide a default value
ARG IMAGE=python:3.11.8

# Use the image as specified
FROM ${IMAGE}

# Re-declare the ARG after FROM
ARG IMAGE

# Update and upgrade the existing packages 
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ninja-build \
    libopenblas-dev \
    build-essential \
    git

RUN mkdir /app
WORKDIR /app
COPY . /app

RUN python3 -m pip install --upgrade pip

RUN make deps && make build && make clean

# Set environment variable for the host
ENV GH_TOKEN=$GH_TOKEN
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MODEL=/app/models/mistral-7b-openorca.Q5_K_M.gguf

# # Install depencencies
# RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context psutil prometheus_client

# # Install llama-cpp-python (build with METAL)
# RUN CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install git+https://${GH_TOKEN}@github.com/ZenHubHQ/llama-cpp-python.git --force-reinstall --upgrade --no-cache-dir --verbose

# Expose a port for the server
EXPOSE 8000

# Run the server start script
CMD ["/bin/sh", "/app/docker/simple/run.sh"]
# CMD python3 -m llama_cpp.server --n_gpu_layers -1
