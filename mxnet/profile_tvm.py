
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import logging
import os
import time
import numpy as np
import statistics

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.relay import expr as _expr
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime


def profile(symbol_file, num_inference_images):
    debug = False
    import tvm
    from tvm.contrib import graph_runtime
    from tvm.contrib.debugger import debug_runtime as debug_runtime

    
    base = os.getcwd() + '/compiled_models/tvm_' + symbol_file.split('/')[-1].replace('.json','')

    path_lib = base + '_deploy_lib.tar'
    path_graph =  base + '_deploy_graph.json'
    path_params = base + '_deploy_params.params'

    graph = open(path_graph).read()
    lib = tvm.runtime.load_module(path_lib)
    params = bytearray(open(path_params, 'rb').read())

    if debug:
        rt_mod = debug_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.load_params(params)
        rt_mod.run()
        return

    rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    rt_mod.load_params(params)

    # warm up
    warm_up = 0
    for i in range(0, 50):
        rt_mod.run()
        warm_up += 1
        if warm_up == 50:
            break


    counter = 0
    time_tvm = list()
    for i in range(0, num_inference_images):
        time0 = time.time()
        rt_mod.run()
        time1 = time.time()
        time_tvm.append(time1 - time0)
        counter += 1
        if counter == num_inference_images:
            break

    avg = lambda x : round(1000*sum(x)/len(x), 6)
    std = lambda x: round(statistics.stdev(x), 6)


    total_tvm = avg(time_tvm)
    sec_tvm = total_tvm/1000
    std_tvm = std(time_tvm)
    min_tvm = round(min(time_tvm), 6)
    min_tvm_ms = round(min(time_tvm)*1000, 6)
    deviation_from_min_tvm = round(sec_tvm/min_tvm*100 - 100, 6)
    deviation_from_std_tvm = round(std_tvm/sec_tvm*100, 6)

    net_name = symbol_file.split('/')[-1].replace('.json','')
    print("Perf", "Tvm", net_name, total_tvm, min_tvm_ms, std_tvm, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=False, help='param file path')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--num-inference-batches', type=int, required=True, help='number of images used for inference')

    args = parser.parse_args()

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    symbol_file = args.symbol_file
    param_file = args.param_file

    image_shape = args.image_shape
    data_shape = tuple([int(i) for i in image_shape.split(',')])

    num_inference_images = args.num_inference_batches
    profile(symbol_file, num_inference_images)
