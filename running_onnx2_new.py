import os
import argparse
import numpy as np
import time

import pyxir
import tvm
from tvm.contrib import graph_executor


from tvm.contrib.download import download_testdata
#import cv2
#FILE_DIR = os.path.dirname(os.path.abspath(__file__))
 

def run(file_path, shape_dict, iterations):
    # DOWNLOAD IMAGE FOR TEST
  #  img_shape = shape_dict[list(shape_dict.keys())[0]][2:]
   # image = Image.open('sample_DPU.jpg')
    
    # IMAGE PRE-PROCESSING
    #image = transform_image(image)
   
    image = np.zeros((1,3,1024,768))
    # RUN #
    inputs = {}
    inputs[list(shape_dict.keys())[0]] = image

    # load the pre-compiled module into memory
    lib = tvm.runtime.load_module(file_path)
    module = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    module.set_input(**inputs)
#tvm_lib.so
    #  VAI FLOW
    for i in range(iterations):
        start = time.time()
        module.run()
        stop = time.time()
        res = [module.get_output(idx).asnumpy()
               for idx in range(module.get_num_outputs())]
        
        inference_time = np.round((stop - start) * 1000, 2)
        
        #res = softmax(res[0])
        #top1 = np.argmax(res)
        print('========================================')
        #print('TVM prediction top-1:', top1, synset[top1])
        print('========================================')
        
        print('========================================')
        print('Inference time: ' + str(inference_time) + " ms")
        print('========================================')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Path to TVM library file (.so)")#, default=FILE_DIR)
    parser.add_argument("--iterations", help="The number of iterations to run.", default=1, type=int)
    args = parser.parse_args()
    file_path = args.f if os.path.isabs(args.f) else os.path.join(os.getcwd(), args.f)
    iterations = args.iterations
    shape_dict = {'input': [1, 3, 1024, 768]}
    run(file_path, shape_dict, iterations)

#im22 = cv2.imread('sample_dpu.jpg',)

#im22.shape

