#!/bin/bash
rm -rf perf.txt
touch perf.txt
# For Mxnet, best baseline, this is required
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# For TVM, best numbers, following line is required.
export TVM_BIND_MASTER_THREAD=1

export NUM_ITERS=5

export MODEL_PATH=./models/model

# Quantized models
python3 compile.py --symbol-file=$MODEL_PATH/resnet18_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet18_v1-quantized-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1-quantized-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/resnet50_v1b-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1b-quantized-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/mobilenet1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenet1.0-quantized-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/mobilenetv2_1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenetv2_1.0-quantized-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/inceptionv3-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/inceptionv3-quantized-0000.params --image-shape=3,299,299  --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/resnet101_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet101_v1-quantized-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-0000.params --image-shape=3,224,224 --tune=no

# BERT
python3 compile_bert.py --symbol-file=$MODEL_PATH/model_bert_squad_quantized-symbol.json --param-file=$MODEL_PATH/model_bert_squad_quantized-0000.params  --image-shape=3,224,224 --tune=no

# FP32 models
python3 compile.py --symbol-file=$MODEL_PATH/resnet18_v1-symbol.json --param-file=$MODEL_PATH/resnet18_v1-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/resnet50_v1-symbol.json --param-file=$MODEL_PATH/resnet50_v1-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/resnet50_v1b-symbol.json --param-file=$MODEL_PATH/resnet50_v1b-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/mobilenet1.0-symbol.json --param-file=$MODEL_PATH/mobilenet1.0-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/mobilenetv2_1.0-symbol.json --param-file=$MODEL_PATH/mobilenetv2_1.0-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/imagenet1k-inception-bn-symbol.json --param-file=$MODEL_PATH/imagenet1k-inception-bn-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/inceptionv3-symbol.json --param-file=$MODEL_PATH/inceptionv3-0000.params --image-shape=3,299,299  --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/resnet101_v1-symbol.json --param-file=$MODEL_PATH/resnet101_v1-0000.params --image-shape=3,224,224 --tune=no
python3 compile.py --symbol-file=$MODEL_PATH/imagenet1k-resnet-152-symbol.json --param-file=$MODEL_PATH/imagenet1k-resnet-152-0000.params --image-shape=3,224,224 --tune=no


