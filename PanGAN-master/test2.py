import tensorflow as tf
import os 
import numpy as np
#import gdal
import cv2
from PanGan_2 import PanGan
from DataSet import DataSet
from config import FLAGES
import scipy.io as scio
import time
import os
import tifffile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''定义参数'''
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('pan_size',
                           default_value=None,
                           docstring='pan image size')
tf.app.flags.DEFINE_string('ms_size',
                           default_value=None,
                           docstring='ms image size')
# tf.app.flags.DEFINE_string('batch_size',
#                            default_value=1,
#                            docstring='img batch')
# tf.app.flags.DEFINE_string('num_spectrum',
#                            default_value=4,
#                            docstring='spectrum num')
# tf.app.flags.DEFINE_string('ratio',
#                            default_value=4,
#                            docstring='pan image/ms img')
tf.app.flags.DEFINE_string('model_path',
                           default_value='./model/qk/Generator-107000',

                           docstring='pan image/ms img')
tf.app.flags.DEFINE_string('test_path',
                           default_value='./data/test_gt',
                           docstring='test img data')
tf.app.flags.DEFINE_string('result_path',
                           default_value='./result',
                           docstring='result img')
# tf.app.flags.DEFINE_string('norm',
#                            default_value=True,
#                            docstring='if norm')


                           
def main(argv):
    if not os.path.exists(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)
    model=PanGan(FLAGS.pan_size,FLAGS.ms_size, 1, 4, 4,0.001, 0.99, 1000,False)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.model_path)
        ms_test_path= FLAGS.test_path + '/lrms'
        pan_test_path=FLAGS.test_path + '/pan'
        for img_name in os.listdir(ms_test_path):
            start=time.time()
            print(img_name)
            pan, ms = read_img(pan_test_path, ms_test_path, img_name,FLAGS)
            start=time.time()
            PanSharpening,error,error2= sess.run([model.PanSharpening_img,model.g_spectrum_loss,model.g_spatial_loss], feed_dict={model.pan_img:pan, model.ms_img:ms})
            PanSharpening=PanSharpening*127.5+127.5
            PanSharpening=PanSharpening.squeeze()
            PanSharpening=PanSharpening.astype('uint8')
            end=time.time()
            print(end-start)
            save_name=img_name.split('.')[0] + '.TIF'
            save_path=os.path.join(FLAGS.result_path,save_name)
            #cv2.imwrite(save_path, PanSharpening)
            #img_write(PanSharpening,save_path)
            # PanSharpening=cv2.cvtColor(PanSharpening[:,:,0:3], cv2.COLOR_BGR2RGB)
            # cv2.imwrite(save_path, PanSharpening)
            tifffile.imsave(save_path, PanSharpening)
            print(img_name + ' done.' + 'spectrum error is ' + str(error) + 'spatial error is ' + str(error2))
            
def read_img(pan_test_path, ms_test_path, img_name, FLAGS):
    pan_img_path=os.path.join(pan_test_path, img_name)
    ms_img_path=os.path.join(ms_test_path, img_name)
    #pan_img=cv2.imread(pan_img_path, -1)
    #pan_img=gdal_read(pan_img_path,'pan')
    pan_img=read8bit(pan_img_path,'pan')
    h,w=pan_img.shape
    pan_img=pan_img.reshape((1,h,w,1))
    #ms_img=cv2.imread(ms_img_path, -1)
    #ms_img=gdal_read(ms_img_path,'ms')
    ms_img=read8bit(ms_img_path,'ms')
    h,w,c=ms_img.shape
    ms_img=cv2.resize(ms_img,(4*w,4*h),interpolation=cv2.INTER_CUBIC)
    h,w,c=ms_img.shape
    
    # ms_img=np.array(ms_img)
    # h,w,c=ms_img.shape
    # ms_img=cv2.resize(ms_img,(4*w,4*h),interpolation=cv2.INTER_CUBIC)
    ms_img=ms_img.reshape((1,h,w,c))
    return pan_img, ms_img
    
def gdal_read(path,name):
    data=gdal.Open(path)
    w=data.RasterXSize
    h=data.RasterYSize
    img=data.ReadAsArray(0,0,w,h)
    if name == 'ms':
        img=np.transpose(img,(1,2,0))
    img=(img-1023.5)/1023.5
    return img
    
def read8bit(path,name):
    if name=='ms':
        v='src'
    else:
        v='pan'
    v='I'
    #img=scio.loadmat(path)[v]
    img=np.load(path)
    img=(img-127.5)/127.5
    return img
    
def img_write(img_array,save_path):
    datatype=gdal.GDT_UInt16
    h,w,c=img_array.shape
    driver=gdal.GetDriverByName('GTiff')
    data=driver.Create(save_path, w, h, c, datatype)
    for i in range(c):
        data.GetRasterBand(i+1).WriteArray(img_array[:,:,i])
    del data
if __name__ == '__main__':
    tf.app.run()
    
      
    
