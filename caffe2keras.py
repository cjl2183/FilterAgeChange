import collections
import os
import pandas as pd
import re
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32,cnmem=1"
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from extra_layers import LRNLayer
from keras.optimizers import SGD
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from keras.layers.core import MaskedLayer
import numpy as np
from preprocessing import imgprocess
import numpy.testing as npt
from PIL import Image
import pickle as pickle
from scipy import misc

srng = RandomStreams(seed=np.random.randint(10e6))

sys.path.append('/home/ubuntu')
import caffe

def get_caffe_params(netname, paramname):
  '''
  Get the parameters from Caffe, returned in the variable "params"

  INPUT: "netname": caffe .prototxt file, "paramname": caffe .caffemodel file
  OUTPUT: "params": model weights, "net": caffe network
  '''

  #load the model in
  net = caffe.Net(netname, paramname, caffe.TEST)
  params = collections.OrderedDict()

  #read all the paramnames into numpy array
  for layername in net.params:
    caffelayer = net.params[layername]
    params[layername] = []
    for sublayer in caffelayer:
      params[layername].append(sublayer.data)
    print "layer "+layername+" has "+str(len(caffelayer))+" sublayers, shape "+str(params[layername][0].shape)
  return params, net


class HDropout(MaskedLayer):
  '''
  Hinton's dropout.
  '''
  def __init__(self, p):
    super(HDropout, self).__init__()
    self.p = p

  def get_output(self, train=False):
    X = self.get_input(train)
    if self.p > 0.:
      retain_prob = 1. - self.p
      if train:
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
      else:
        # X *= retain_prob
        X = X
    return X

  def get_config(self):
    return {"name": self.__class__.__name__, "p": self.p}

def keras_agenet(cls='fc1',drop=0):
    '''
    Builds the keras version of the caffe agenet model. 

    INPUT: (optional) layer to pop
    OUTPUT: Keras model function to be compiled
    '''
    model = Sequential()

    # Layer 1 conv1
    #model.add(ZeroPadding2D((0, 0), input_shape=(3, 227, 227)))
    model.add(Convolution2D(96, 7, 7, subsample=(4,4), input_shape=(3, 227, 227), activation='relu'))

    # Layer 3 pool1
    model.add(ZeroPadding2D((0, 0)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    # Layer 4 norm1 LRN2D
    model.add(ZeroPadding2D((0, 0)))
    model.add(LRNLayer.LRN2D(n=5, alpha=0.0001, beta=0.75))
 
    # Layer 5 conv2
    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(256, 5, 5, activation='relu'))

    # Layer 6 pool2
    model.add(ZeroPadding2D((0, 0)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    # Layer 7 norm2 LRN2D
    model.add(ZeroPadding2D((0, 0)))
    model.add(LRNLayer.LRN2D(n=5, alpha=0.0001, beta=0.75))

    # Layer 8 conv3
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(384, 3, 3, activation='relu'))

    # Layer 9 pool5
    model.add(ZeroPadding2D((0, 0)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    # Layer 10 fc6 - INNER PRODUCT
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))

    # Layer 11 drop6 - DROPOUT
    model.add(Dropout(drop))

    # Layer 12 fc7 - INNER PRODUCT
    model.add(Dense(512, activation='relu'))

    # Layer 13 drop7 - DROPOUT
    model.add(Dropout(drop))

    # Layer 14 fc8 - INNER PRODUCT
    model.add(Dense(8, activation='softmax'))

    return model


def set_keras_params(model, params):
  '''
  Transfers the caffe weights into the keras model. Handles different Dense and Convolution2D
  definitions between Keras and Caffe

  INPUT: "model": the keras model, "params": the caffe weights
  OUTPUT: "model": the keras model with weights
  '''
  
  weightlayers=[]
  layerindex = 0

  for layer in model.layers:
    if len(layer.get_weights()) > 0:
      weightlayers.append(layerindex)
    layerindex+=1
  print "There are "+str(len(weightlayers))+" layers in the model with weights"

  if len(weightlayers) != len(params):
    print "ERROR: caffe model and specified keras model do not match"
    return model 

  paramkeys = params.keys()

  for i in xrange(0,len(params)):
    layer = model.layers[weightlayers[i]]
    #print layer.input_shape
    #print layer.output_shape
    weights = params[paramkeys[i]]

    # Dense layers are specified as Input-Output in Keras
    if type(layer) is Dense:
      weights[0] = weights[0].transpose(1,0)
      weights[1] = weights[1]
    # Convolution 2D is specified as flip and then multiply
    elif type(layer) is Convolution2D:
      weights[0] = weights[0].transpose(0,1,2,3)[:,:,::-1,::-1]
    layer.set_weights(weights)
    
  return model


def caffe2keras( caffemodel, caffeparams, kerasmodel ):
  '''
 	Transfer caffe network to keras by extracting caffe weights, instantiating the keras model, 
 	and transferring caffe weights to keras model, then compiling.

  INPUT: "caffemodel": the caffe model, "caffeparams": the caffe weights, "kerasmodel": the keras model
  OUTPUT: "kerasmodel": the keras model with weights applied, "net": the caffemodel
  '''
  params,net = get_caffe_params(caffemodel,caffeparams)
  kerasmodel = set_keras_params(kerasmodel, params)
  kerasmodel.compile(loss='categorical_crossentropy', optimizer='sgd')

  print "Finished compiling categoral crossentropy on VGG network."

  return kerasmodel,net


if __name__ == '__main__':

  MODEL_DIR = '/home/ubuntu/agenet/model/'
  IMAGE_DIR = '/home/ubuntu/agenet/images/'
  RESULTS_DIR = '/home/ubuntu/agenet/results/'
  FILTER_TYP = 'Orig'

  netname = 'deploy_age.prototxt'
  paramname = 'age_net.caffemodel'

  keras_agemodel = keras_agenet(drop=0)

  params, net = get_caffe_params(MODEL_DIR + netname, MODEL_DIR + paramname)
  kmodel, caffenet = caffe2keras(MODEL_DIR + netname, MODEL_DIR + paramname, keras_agemodel)

  '''
  #CAN'T PICKLE MODEL ANYMORE

  config = kmodel.get_config()
  pickle.dump(config, open("keras_agemodel.p", "wb"))
  config = pickle.load(open("/home/ubuntu/agenet/keras_agemodel.p", "rb"))
  kmodel = Sequential.from_config(config)

  # CHECK WEIGHTS
  caffe_weights = params['fc8']
  keras_weights = kmodel.layers[-1].get_weights()
  print "CAFFE WEIGHTS============"
  print np.array(caffe_weights[0].transpose(1,0)).shape
  caffe_coeff = np.array(caffe_weights[0].transpose(1,0))

  #print keras_weights.shape
  print "KERAS WEIGHTS============"
  print np.array(keras_weights[0]).shape
  keras_coeff = np.array(keras_weights[0])

  print npt.assert_allclose(caffe_coeff, keras_coeff, atol=.00000001)
  diff = caffe_coeff - keras_coeff
  print diff[np.argmin(diff)]
  print diff[np.argmax(diff)]

  #LOAD MEAN IMAGE
  mean_filename=os.path.join(IMAGE_DIR,'mean.binaryproto')
  proto_data = open(mean_filename, "rb").read()
  a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
  mean  = caffe.io.blobproto_to_array(a)[0]

  #CHECK RESIZING IMAGE LOOKS GOOD => LOOKS GOOD
  test = misc.imread(os.path.join(IMAGE_DIR, FILTER_TYP, img_list[0]))
  test = test[:,:,[2,1,0]]
  test = misc.imresize(test, (227, 227)).astype(np.float32)
  misc.imsave(os.path.join(IMAGE_DIR,'test.jpg'), test)

  '''
  img_list = [f for f in os.listdir(IMAGE_DIR+FILTER_TYP) if re.match(r'[a-zA-Z0-9\-]*.jpg', f)]
  results = []
  age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

  for img in img_list:
    im = imgprocess.transform_image_RGB(os.path.join(IMAGE_DIR,FILTER_TYP,img), target_size=227)
    print im.shape
    #im = imgprocess.load_img(IMAGE_DIR+FILTER_TYP+'/'+img, grayscale=False, target_size=227)
    #im = imgprocess.img_to_array(im, dim_ordering='th')
    #im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
    prediction = kmodel.predict(im, batch_size=1, verbose=1)
    range_predict = age_list[np.argmax(prediction)] 
    results.append(np.hstack((prediction[0], range_predict)))

  age_list.append('prediction')
  pred_results = pd.DataFrame(results, index=img_list, columns=age_list)
  pred_results.to_csv(RESULTS_DIR+FILTER_TYP+'.csv')

  #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  #kmodel.compile(optimizer=sgd, loss='categorical_crossentropy')

  print "done"


