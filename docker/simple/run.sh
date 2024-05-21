#!/bin/bash

make build
# uvicorn --factory llama_cpp.server.app:create_app --host $HOST --port $PORT --reload
python3 -m llama_cpp.server --model $MODEL  --n_gpu_layers -1