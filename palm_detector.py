import tensorflow as tf
import os,sys
import cv2
import time
import csv
import math

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, MaxPool2D, Convolution2DTranspose
from tensorflow.keras.layers import ZeroPadding2D, Reshape, ReLU, Add, Concatenate
from tensorflow.keras import Model, Input

print(tf.__version__)

class PalmDetector:

    PATH = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = PATH + "/weights/"
    CONFIDENCE_THRESH = 0.70
    ANCHORS_PATH = PATH + "/anchors.csv"
    IOU_THRESH = 0.1

    class Result:
        def __init__(self,palm_det):
            self.x1 = int(palm_det[0])
            self.y1 = int(palm_det[1])
            self.x2 = int(palm_det[2])
            self.y2 = int(palm_det[3])

            self.conf = palm_det[4]

        def get_top_left_point(self):
            return (self.x1,self.y1)
        def get_bottom_right_point(self):
            return (self.x2,self.y2)



    def __init__(self):
        with open(self.ANCHORS_PATH,"r") as csv_f:
            self.anchors = np.r_[
                    [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ] 

        self.layer_names = ["conv_1",
                            "residualb_1","residualb_2","residualb_3","residualb_4",
                            "residualb_5","residualb_6","residualb_7","residualb_8",
                            "residualb_9","residualb_10","residualb_11","residualb_12",
                            "residualb_13","residualb_14","residualb_15","residualb_16",
                            "residualb_17","residualb_18","residualb_19","residualb_20",
                            "residualb_21","residualb_22","residualb_23","residualb_24",
                            "residualb_25","residualb_26","residualb_27","residualb_28",
                            "residualb_29","residualb_30","residualb_31",
                            "dconv_1","conv_2",
                            "residualb_32","residualb_33","residualb_34","residualb_35",
                            "residualb_36","residualb_37","residualb_38",
                            "convt_1","residualb_39","convt_2","residualb_40",
                            "conv_3","conv_4","conv_5","conv_6","conv_7","conv_8"
                            ]
        self.model = None
    def residual_block(self, tensor, feature_n,name=None):
        if name != None:
            dconv = DepthwiseConv2D(3,padding='same',name=name+"/dconv")(tensor) 
            conv  = Conv2D(feature_n,1,padding='valid',name=name+"/conv")(dconv)
        else:
            dconv = DepthwiseConv2D(3,padding='same')(tensor) 
            conv  = Conv2D(feature_n,1,padding='valid')(dconv)
        add   = Add()([conv, tensor])
        relu = ReLU()(add)
        return relu 
    def residual_block_id(self,tensor, feature_n,name=None):
        if name != None:
            depconv_1  = DepthwiseConv2D(3,2,padding='same',name=name+"/dconv")(tensor)
            conv_2     = Conv2D(feature_n,1,name=name+"/conv")(depconv_1)
        else:
            depconv_1  = DepthwiseConv2D(3,2,padding='same')(tensor)
            conv_2     = Conv2D(feature_n,1)(depconv_1)


        maxpool_1  = MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same')(tensor)
        conv_zeros = Conv2D(feature_n/2,2,strides=2,use_bias=False,kernel_initializer=tf.zeros_initializer())(tensor)

        padding_1  = Concatenate(axis=-1)([maxpool_1,conv_zeros])#self.feature_padding(maxpool_1)

        add = Add()([padding_1,conv_2])
        relu = ReLU()(add)

        return relu
    
    #def feature_padding(self,tensor,channels_n=0):
    #    #pad = tf.keras.layers.ZeroPadding2D(((0,0),(0,0),(0,tensor.shape[3])))(tensor)
    #    return Concatenate(axis=3)([tensor,pad])



    def build(self):
        self.inputs = Input(shape=(256,256,3), name="img")
        self.conv_1  = Conv2D(32,3,strides=(2,2),use_bias=True,padding='same',name="conv_1")(self.inputs)
        self.relu_1  = ReLU()(self.conv_1)

        self.residualb_1 = self.residual_block(self.relu_1, 32,"residualb_1")
        self.residualb_2 = self.residual_block(self.residualb_1, 32,"residualb_2")
        self.residualb_3 = self.residual_block(self.residualb_2, 32,"residualb_3")
        self.residualb_4 = self.residual_block(self.residualb_3, 32,"residualb_4")
        self.residualb_5 = self.residual_block(self.residualb_4, 32,"residualb_5")
        self.residualb_6 = self.residual_block(self.residualb_5, 32,"residualb_6")
        self.residualb_7 = self.residual_block(self.residualb_6, 32,"residualb_7") 

        self.residualb_8 = self.residual_block_id(self.residualb_7, 64,"residualb_8")

        self.residualb_9 = self.residual_block(self.residualb_8, 64,"residualb_9")
        self.residualb_10 = self.residual_block(self.residualb_9, 64,"residualb_10")
        self.residualb_11 = self.residual_block(self.residualb_10, 64,"residualb_11")
        self.residualb_12 = self.residual_block(self.residualb_11, 64,"residualb_12")
        self.residualb_13 = self.residual_block(self.residualb_12, 64,"residualb_13")
        self.residualb_14 = self.residual_block(self.residualb_13, 64,"residualb_14")
        self.residualb_15 = self.residual_block(self.residualb_14, 64,"residualb_15")

        self.residualb_16 = self.residual_block_id(self.residualb_15, 128,"residualb_16")

        self.residualb_17 = self.residual_block(self.residualb_16, 128,"residualb_17")
        self.residualb_18 = self.residual_block(self.residualb_17, 128,"residualb_18")
        self.residualb_19 = self.residual_block(self.residualb_18, 128,"residualb_19")
        self.residualb_20 = self.residual_block(self.residualb_19, 128,"residualb_20")
        self.residualb_21 = self.residual_block(self.residualb_20, 128,"residualb_21")
        self.residualb_22 = self.residual_block(self.residualb_21, 128,"residualb_22")
        self.residualb_23 = self.residual_block(self.residualb_22, 128,"residualb_23")

        self.residualb_24 = self.residual_block_id(self.residualb_23, 256,"residualb_24")

        self.residualb_25 = self.residual_block(self.residualb_24, 256,"residualb_25")
        self.residualb_26 = self.residual_block(self.residualb_25, 256,"residualb_26")
        self.residualb_27 = self.residual_block(self.residualb_26, 256,"residualb_27")
        self.residualb_28 = self.residual_block(self.residualb_27, 256,"residualb_28")
        self.residualb_29 = self.residual_block(self.residualb_28, 256,"residualb_29")
        self.residualb_30 = self.residual_block(self.residualb_29, 256,"residualb_30")
        self.residualb_31 = self.residual_block(self.residualb_30, 256,"residualb_31")

        self.max_pool_1   = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same')(self.residualb_31)
        self.dconv_1      = DepthwiseConv2D(kernel_size=(3,3),strides=(2,2),padding='same',name="dconv_1")(self.residualb_31)
        self.conv_2       = Conv2D(filters=256,kernel_size=(1,1),strides=(1,1),padding='valid',name="conv_2")(self.dconv_1)

        self.add_1        = Add()([self.conv_2, self.max_pool_1])
        self.relu_2       = ReLU()(self.add_1)

        self.residualb_32 = self.residual_block(self.relu_2,256,"residualb_32")
        self.residualb_33 = self.residual_block(self.residualb_32,256,"residualb_33")
        self.residualb_34 = self.residual_block(self.residualb_33,256,"residualb_34")
        self.residualb_35 = self.residual_block(self.residualb_34,256,"residualb_35")
        self.residualb_36 = self.residual_block(self.residualb_35,256,"residualb_36")
        self.residualb_37 = self.residual_block(self.residualb_36,256,"residualb_37")

        #split key
        self.residualb_38 = self.residual_block(self.residualb_37,256,"residualb_38")

        #BRANCH 1
        self.conv_transpose_1 = Convolution2DTranspose(filters=256, kernel_size=(2,2), strides=(2,2),name="convt_1")(self.residualb_38)
        self.relu_3         = ReLU()(self.conv_transpose_1)
        self.add_2          = Add()([self.residualb_31,self.relu_3])
        #split key
        self.residualb_39 = self.residual_block(self.add_2,256,"residualb_39")

        self.conv_transpose_2 = Convolution2DTranspose(filters=128,kernel_size=(2,2),strides=(2,2),name="convt_2")(self.residualb_39)
        self.relu_4     = ReLU()(self.conv_transpose_2)
        self.add_3      = Add()([self.residualb_23,self.relu_4])
        #split key
        self.residualb_40 = self.residual_block(self.add_3,128,"residualb_40")

        # output block 1
        self.conv_3 = Conv2D(filters=2,kernel_size=(1,1),strides=(1,1),padding='same',name="conv_3")(self.residualb_40)
        #self.reshape_1 = tf.reshape(self.conv_3,[1,2048,1])
        self.reshape_1 = Reshape([-1,1])(self.conv_3)

        self.conv_4 = Conv2D(filters=2,kernel_size=(1,1),strides=(1,1),name="conv_4")(self.residualb_39)
        #self.reshape_2 = tf.reshape(self.conv_4,[1,512,1])
        self.reshape_2 = Reshape([-1,1])(self.conv_4)

        self.conv_5 = Conv2D(filters=6,kernel_size=(1,1),strides=(1,1),name="conv_5")(self.residualb_38)
        #self.reshape_3 = tf.reshape(self.conv_5, [1,384,1])
        self.reshape_3 = Reshape([-1,1])(self.conv_5)

        self.concat_1 = Concatenate(axis=1)([self.reshape_1,self.reshape_2,self.reshape_3])

        #output block 2
        self.conv_6 = Conv2D(filters=36,kernel_size=(1,1),strides=(1,1),padding='same',name="conv_6")(self.residualb_40)
        #self.reshape_4 = tf.reshape(self.conv_6,[1,2048,18])
        self.reshape_4 = Reshape([-1,18])(self.conv_6)

        self.conv_7 = Conv2D(filters=36,kernel_size=(1,1),strides=(1,1),padding='same',name="conv_7")(self.residualb_39)
        #self.reshape_5 = tf.reshape(self.conv_7,[1,512,18])
        self.reshape_5 = Reshape([-1,18])(self.conv_7)

        self.conv_8 = Conv2D(filters=108,kernel_size=(1,1),strides=(1,1),name="conv_8")(self.residualb_38)
        #self.reshape_6 = tf.reshape(self.conv_8,[1,384,18])
        self.reshape_6 = Reshape([-1,18])(self.conv_8)

        self.concat_2 = Concatenate(axis=1)([self.reshape_4,self.reshape_5,self.reshape_6])

        self.init_model()

        return self.concat_1, self.concat_2

    def get_inputs_layer(self):
        return self.inputs
    def get_output_layers(self):
        return [self.concat_1, self.concat_2]
    def init_model(self):
        self.model = Model(inputs=self.get_inputs_layer(), outputs=self.get_output_layers(), name="palm_detector")
        print("Loading Model weights")
        for i in range(0,len(self.layer_names)):
            print(self.layer_names[i])
            if "residual" in self.layer_names[i]:
                self.load_weights_to_residual_block(self.layer_names[i])
            else:
                layer = self.model.get_layer(name=self.layer_names[i])
                if "dconv" in self.layer_names[i]:
                    self.load_weights_to_layer(self.WEIGHTS_PATH + self.layer_names[i]+"_w.npy",None,layer)
                else:
                    self.load_weights_to_layer(self.WEIGHTS_PATH + self.layer_names[i]+"_w.npy",self.WEIGHTS_PATH + self.layer_names[i]+"_b.npy",layer)
        print("done loading palm weights")

        tf.keras.utils.plot_model(self.model, 'model.png',show_shapes=True)

    def preprocess(self,image):
        self.image_shape = image.shape
        shape = np.r_[self.image_shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 2
        image_pad = np.pad(image,((pad[0],pad[0]), (pad[1],pad[1]), (0,0)),mode='constant')
        palm_frame = cv2.resize(image_pad, (256,256))
        palm_frame = np.expand_dims(palm_frame,axis=0).astype(np.float32)
        palm_frame = np.ascontiguousarray(2* (palm_frame/255 - .5))
        return palm_frame

    def run_all(self, image):
        palm_frame = self.preprocess(image)
        return self.run(palm_frame)
        
    def run(self, frame):
        time_s = time.time()
        classifactors, regressors = self.model.predict(frame)
        time_model_predict = ( time.time() - time_s ) * 1000
        time_s = time.time()
        palm_dets = self.post_process(classifactors, regressors)
        time_post_process = ( time.time() - time_s ) * 1000
        print(f"PALM : Time for model {time_model_predict} ms")
        print(f"PALM : Time for post proc {time_post_process} ms")
        return palm_dets
    
    def post_process(self, classificators, regressors):
        classificators = self.sigmoid(classificators[0])
        regressors  = regressors[0]

        x1y1 = self.anchors[:,:2] * 256 - regressors[:,2:4]/2
        x2y2 = self.anchors[:,:2] * 256 + regressors[:,2:4]/2
        palm_dets_all = np.concatenate((x1y1,x2y2),axis=1)
        palm_dets_all = np.concatenate((palm_dets_all,classificators),axis=1)
        palm_dets = palm_dets_all[palm_dets_all[:,4] > self.CONFIDENCE_THRESH]

        palm_dets_result_idx = self.nms(palm_dets)
        palm_results = []
        for i in palm_dets_result_idx:
            palm_det = palm_dets[i,:]
            palm_det = self.scale_to_image(palm_det)
            palm_det = PalmDetector.Result(palm_det)
            print(f"PALM : added {palm_det.conf}")
            palm_results.append(palm_det)
        return palm_results

    def scale_to_image(self,det):
        print(self.image_shape)
        scalex = self.image_shape[1] / 256
        scaley = self.image_shape[0] / 256
        det[0] = det[0] * scalex
        det[1] = det[1] * scaley
        det[2] = det[2] * scalex
        det[3] = det[3] * scaley
        return det
    
    def load_weights_to_layer(self,path_w,path_b,layer):
        w = np.load(path_w)
        if "convt" in layer.name:
            w = np.transpose(w,(1,2,0,3))
        else:
            w = np.transpose(w,(1,2,3,0))
        if path_b == None:
            layer.set_weights([w,np.zeros((w.shape[-2]))])
            return
        b = np.load(path_b)
        layer.set_weights([w,b])
    def load_weights_to_residual_block(self,residual_b_name):
        dconv_w_n = self.WEIGHTS_PATH + residual_b_name+"_dconv_w.npy"
        w_n = self.WEIGHTS_PATH + residual_b_name+"_conv_w.npy"
        b_n = self.WEIGHTS_PATH +residual_b_name+"_conv_b.npy"
        dconv = self.model.get_layer(name=residual_b_name+"/dconv")
        conv  = self.model.get_layer(name=residual_b_name+"/conv")
        self.load_weights_to_layer(dconv_w_n,None,dconv)
        self.load_weights_to_layer(w_n,b_n,conv)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def nms(self, dets):

        scores = dets[:, 4]
        order = scores.argsort()[::-1]

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        keep = []
        while order.size > 0:
            #get maximum
            i = order[0]

            #if scores[i] < self.CONFIDENCE_THRESH:
            #    break
            #append to keep
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.IOU_THRESH)[0]
            order = order[inds + 1]
        return keep
    






