#!/bin/bash
rm -rf perf.txt
touch perf.txt
# For Mxnet, best baseline, this is required
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# For TVM, best numbers, following line is required.
export TVM_BIND_MASTER_THREAD=1 

NUM_ITERS=5
export MODEL_PATH=./models/model

for i in $(seq 1 $NUM_ITERS)
do
    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/resnet18_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet18_v1-quantized-0000.params --num-inference-batches=2000 |& tee -a perf.txt
    python3 profile_tvm.py --symbol-file=$MODEL_PATH/resnet18_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet18_v1-quantized-0000.params --num-inference-batches=2000 |& tee -a perf.txt
done

for i in $(seq 1 $NUM_ITERS)
do
    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1-quantized-0000.params --num-inference-batches=2000 |& tee -a perf.txt
    python3 profile_tvm.py --symbol-file=$MODEL_PATH/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1-quantized-0000.params --num-inference-batches=2000 |& tee -a perf.txt
done

for i in $(seq 1 $NUM_ITERS)
do
    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/resnet50_v1b-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1b-quantized-0000.params --num-inference-batches=2000 |& tee -a perf.txt
    python3 profile_tvm.py --symbol-file=$MODEL_PATH/resnet50_v1b-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1b-quantized-0000.params --num-inference-batches=2000 |& tee -a perf.txt
done

for i in $(seq 1 $NUM_ITERS)
do
    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/resnet101_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet101_v1-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
    python3 profile_tvm.py --symbol-file=$MODEL_PATH/resnet101_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet101_v1-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
done


for i in $(seq 1 $NUM_ITERS)
do
    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
    python3 profile_tvm.py --symbol-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
done

for i in $(seq 1 $NUM_ITERS)
do
    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/inceptionv3-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/inceptionv3-quantized-0000.params --image-shape=3,299,299   --num-inference-batches=2000  |& tee -a perf.txt
    python3 profile_tvm.py --symbol-file=$MODEL_PATH/inceptionv3-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/inceptionv3-quantized-0000.params --image-shape=3,299,299   --num-inference-batches=2000  |& tee -a perf.txt
done

for i in $(seq 1 $NUM_ITERS)
do
    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
    python3 profile_tvm.py --symbol-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
done

for i in $(seq 1 $NUM_ITERS)
do
    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/mobilenet1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenet1.0-quantized-0000.params --num-inference-batches=2000  --image-shape=3,224,224 |& tee -a perf.txt
    python3 profile_tvm.py --symbol-file=$MODEL_PATH/mobilenet1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenet1.0-quantized-0000.params --num-inference-batches=2000  --image-shape=3,224,224 |& tee -a perf.txt
done

for i in $(seq 1 $NUM_ITERS)
do
    python3 profile_mxnet.py --symbol-file=$MODEL_PATH/mobilenetv2_1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenetv2_1.0-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
    python3 profile_tvm.py --symbol-file=$MODEL_PATH/mobilenetv2_1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenetv2_1.0-quantized-0000.params --image-shape=3,224,224  --num-inference-batches=2000  |& tee -a perf.txt
done



python3 profile_bert.py --symbol-file=../models/model/model_bert_squad_quantized-symbol.json --param-file=../models/model/model_bert_squad_quantized-0000.params  --image-shape=3,224,224  --num-inference-batches=2000
