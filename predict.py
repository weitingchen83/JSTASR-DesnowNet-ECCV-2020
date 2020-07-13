import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from argparse import ArgumentParser
from keras.preprocessing.image import ImageDataGenerator

#custom
import model.model 


print('import end')

def parse_args():
    parser = ArgumentParser(description='Predict')
    parser.add_argument(
        '-dataroot', '--dataroot',
        type=str, default='./testImg',
        help='root of the image'
    )
    parser.add_argument(
        '-datatype', '--datatype',
        type=str, default=['jpg','tif','png'],
        help='type of the image'
    )
    parser.add_argument(
        '-predictpath', '--predictpath',
        type=str, default='./predictImg',
        help='root of the output'
    )
    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int, default=3,
        help='batch_size'
    )
    return  parser.parse_args()

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    if count != total:
        sys.stdout.flush()
    else:
        print()
    
def generate_data_generator(datagenerator, X,BATCHSIZE):
    genX1 = datagenerator.flow(X,batch_size = BATCHSIZE,shuffle=False)
    count = 0
    while True:
            Xi1 = genX1.next()
            
            # /255 And Change to RGB
            Xi1 = Xi1/255
            Xi1 = Xi1[:,:,:,::-1]
            yield [Xi1]

if __name__== '__main__':
    
    args = parse_args()

    #read test data
    selectNames = []

    data=[]
    print('Read img from:' , args.dataroot)
    fnames=os.listdir(args.dataroot)
    print('Len of the file:',len(fnames))
    count = 1
    for f in fnames:
        progress(count,len(fnames),'Loading data...')
        count+=1
        if f.split('.')[-1] in args.datatype:
            tmp=cv2.imread(args.dataroot+'/'+f)
            selectNames.append(f)
            if tmp.shape[1]<tmp.shape[0]:
                tmp=np.rot90(tmp)
            if tmp.shape[0]!=480 or tmp.shape[1]!=640:
                tmp=cv2.resize(tmp, (640, 480), interpolation=cv2.INTER_CUBIC)
            data.append(tmp)
    data=np.array(data)
    
    print(data.shape,'data shape')
    
    if not os.path.exists(args.predictpath):
        os.mkdir(args.predictpath)
    
    #BUILD COMBINE MODEL
    modelRecoverCombine = model.model.build_combine_model()
    print('LogPath:',args.predictpath)

    val_data_gen = ImageDataGenerator(featurewise_center=False,
                         featurewise_std_normalization=False)
    
    pred=modelRecoverCombine.predict_generator(generate_data_generator(val_data_gen,data,args.batch_size),steps = data.shape[0]/args.batch_size,verbose=1)
    
    print('Save Output')
    for i in range(pred.shape[0]):
        progress(i+1,data.shape[0],'Saving output...')
        pred[i]=np.clip(pred[i],0.0,1.0)
        cv2.imwrite(args.predictpath+'/'+os.path.splitext(selectNames[i])[0]+'.jpg', (pred[i]*255).astype(np.uint8))
