import numpy as np
import os
import h5py
#import gdal
import scipy.io as scio


class DataSet(object):
    def __init__(self,pan_size,ms_size,source_path,data_save_path,batch_size, stride, category='train'):
        self.pan_size=pan_size
        self.ms_size=ms_size
        self.batch_size=batch_size
        if not os.path.exists(data_save_path):
            self.make_data(source_path,data_save_path,stride)
        self.pan,self.ms=self.read_data(data_save_path,category)
        self.data_generator=self.generator()
        
    def generator(self):
        num_data=self.pan.shape[0]
        while True:
            batch_pan=np.zeros((self.batch_size,self.pan_size,self.pan_size,1))
            batch_ms=np.zeros((self.batch_size,self.ms_size,self.ms_size,4))
            for i in range(self.batch_size):
                random_index=np.random.randint(0,num_data)
                batch_pan[i]=self.pan[random_index]
                batch_ms[i]=self.ms[random_index]
            yield batch_pan, batch_ms
    
    def read_data(self,path,category):
        f=h5py.File(path, 'r')
        if category == 'train':
            pan=np.array(f['pan_train'])
            ms=np.array(f['ms_train'])
        else:
            pan=np.array(f['pan_valid'])
            ms=np.array(f['ms_valid'])
        return pan,ms
        
    def make_data(self, source_path, data_save_path, stride):
        # source_ms_path=os.path.join(source_path, 'MS','1.TIF')
        # source_pan_path=os.path.join(source_path, 'Pan','1.TIF')
        source_ms_path=os.path.join(source_path, 'MS','1.mat')
        source_pan_path=os.path.join(source_path, 'Pan','1.mat')
        all_pan=self.crop_to_patch(source_pan_path, stride, name='pan')
        all_ms=self.crop_to_patch(source_ms_path, stride, name='ms')
        print('The number of ms patch is: ' + str(len(all_ms)))
        print('The number of pan patch is: ' + str(len(all_pan)))
        pan_train, pan_valid, ms_train, ms_valid=self.split_data(all_pan,all_ms)
        print('The number of pan_train patch is: ' + str(len(pan_train)))
        print('The number of pan_valid patch is: ' + str(len(pan_valid)))
        print('The number of ms_train patch is: ' + str(len(ms_train)))
        print('The number of ms_valid patch is: ' + str(len(ms_valid)))
        pan_train=np.array(pan_train)
        pan_valid=np.array(pan_valid)
        ms_train=np.array(ms_train)
        ms_valid=np.array(ms_valid)
        f=h5py.File(data_save_path,'w')
        f.create_dataset('pan_train', data=pan_train)
        f.create_dataset('pan_valid', data=pan_valid)
        f.create_dataset('ms_train', data=ms_train)
        f.create_dataset('ms_valid', data=ms_valid)
        
    def crop_to_patch(self, img_path, stride, name):
        #img=(cv2.imread(img_path,-1)-127.5)/127.5
        img=self.read_img2(img_path)
        h=img.shape[0]
        w=img.shape[1]
        print(h)
        print(w)
        all_img=[]
        if name == 'ms':
            for i in range(0, h-self.ms_size, stride):
                for j in range(0, w-self.ms_size, stride):
                    img_patch=img[i:i+self.ms_size, j:j+self.ms_size,:]
                    all_img.append(img_patch)
                    if i + self.ms_size >= h:
                        img_patch=img[h-self.ms_size:, j:j+self.ms_size,:]
                        all_img.append(img_patch)
                img_patch=img[i:i+self.ms_size, w-self.ms_size:,:]
                all_img.append(img_patch)
        else:
            for i in range(0, h-self.pan_size, stride*4):
                for j in range(0, w-self.pan_size, stride*4):
                    img_patch=img[i:i+self.pan_size, j:j+self.pan_size].reshape(self.pan_size,self.pan_size,1)
                    all_img.append(img_patch)
                    if i + self.pan_size >= h:
                        img_patch=img[h-self.pan_size:, j:j+self.pan_size].reshape(self.pan_size,self.pan_size,1)
                        all_img.append(img_patch)
                img_patch=img[i:i+self.pan_size, w-self.pan_size:].reshape(self.pan_size,self.pan_size,1)
                all_img.append(img_patch)
        return all_img
        
    def split_data(self,all_pan,all_ms):
        ''' all_pan和all_ms均为list'''
        pan_train=[]
        pan_valid=[]
        ms_train=[]
        ms_valid=[]
        for i in range(len(all_pan)):
            rand=np.random.randint(0,100)
            if rand <=10:
                pan_valid.append(all_pan[i])
                ms_valid.append(all_ms[i])
            else:
                ms_train.append(all_ms[i])
                pan_train.append(all_pan[i])
        return pan_train, pan_valid, ms_train, ms_valid
        
    def read_img(self,path,name):
        data=gdal.Open(path)
        w=data.RasterXSize
        h=data.RasterYSize
        img=data.ReadAsArray(0,0,w,h)
        if name == 'ms':
            img=np.transpose(img,(1,2,0))
        img=(img-1023.5)/1023.5
        return img
        
    def read_img2(self, path):
        img=scio.loadmat(path)['I']
        img=(img-127.5)/127.5
        
        return img
        
      
        
                    
         
    
    
 
                
