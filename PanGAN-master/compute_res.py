#from PIL import Image,ImageFilter
import os
import numpy as np
import cv2
#import gdal
import scipy.io as scio


def read_img(path,name):
    data=gdal.Open(path)
    w=data.RasterXSize
    h=data.RasterYSize
    img=data.ReadAsArray(0,0,w,h)
    if name == 'ms':
        img=np.transpose(img,(1,2,0))
    return img
    
def read_img2(path):
    img=scio.loadmat(path)['I']
        
    return img
    
def img_write(img_array,save_path):
    datatype=gdal.GDT_UInt16
    h,w=img_array.shape
    c=1
    driver=gdal.GetDriverByName('GTiff')
    data=driver.Create(save_path, w, h, c, datatype)
    for i in range(c):
        data.GetRasterBand(i+1).WriteArray(img_array)
    del data

ms_source_path='./data/test/5_ms_crop_8bit'
pan_source_path='./data/test/5_pan_crop_8bit'
result_path='./result'
save_path='./result/ms_res'

if not os.path.exists(save_path):
    os.makedirs(save_path)

data=os.listdir(result_path)
sum=0
for img_name in data:
    if '.TIF' in img_name:
        pan_sharp=cv2.imread(os.path.join(result_path,img_name))
        pan_sharp=cv2.cvtColor(pan_sharp, cv2.COLOR_BGR2RGB)/1.0
        #pan_sharp=cv2.GaussianBlur(pan_sharp, (7,7), 1);
        #print(pan_sharp[:,:,3])
        img_name_sr=img_name.split('.')[0]+'.mat'
        ms=read_img2(os.path.join(ms_source_path,img_name_sr))/1.0
        h,w,c=ms.shape
        pan_sharp=cv2.resize(pan_sharp,(w,h),interpolation=cv2.INTER_LINEAR)
        res=(np.abs(pan_sharp-ms[:,:,0:3])).astype('uint8')
        res=cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        sum=sum+np.mean(res)
        print(img_name+' '+str(np.mean(res)))
        cv2.imwrite(os.path.join(save_path,img_name),res)
print('mean is : '+ str(sum/len(data)))
    

         
