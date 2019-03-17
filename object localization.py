import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#from sklearn.metrics import roc_auc_score


ant_path=r'C:\Users\Anu\Downloads\Compressed\stanford-dogs-dataset\Annotation'
img_path=r'C:\Users\Anu\Downloads\Compressed\stanford-dogs-dataset\Images'

images=[]
annotations=[]
true_class=[]

classes=os.listdir(ant_path)

#Load Images,Classes,Annotations
for index,sample in enumerate(classes):
    print('Processing Sample: %d'%(index+1))
    ant_files=os.listdir(os.path.join(ant_path,sample))
    files=glob.glob(os.path.join(img_path,sample,'*.jpg'))
    for i in range(len(files)):
        print('Loading Class %d and Annotation File: %d'%(index+1,i+1))
        img=cv2.imread(files[i])
        img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        img=cv2.resize(img,(175,175),cv2.INTER_AREA)
        images.append(img)
        
        #Annotations
        ant_file_path=os.path.join(ant_path,sample,ant_files[i])
        tree=ET.parse(ant_file_path)
        root=tree.getroot()
        objects=root.findall('object')
        for o in objects:
            tmp=[]
            bndbox=o.find('bndbox')
            tmp.append(int(bndbox.find('xmin').text))
            tmp.append(int(bndbox.find('ymin').text))
            tmp.append(int(bndbox.find('xmax').text))
            tmp.append(int(bndbox.find('ymax').text))
        annotations.append(tmp)
        true_class.append(index+1)

#Reshape Annotations
annotations=np.reshape(annotations,(20580,4))

#One hot Encode Labels
class_labels=np.zeros((len(images),120))
for index,class_ in enumerate(true_class):
    class_labels[index][class_-1]=1.0
    
#Split Dataset
train_images,val_images,train_labels,val_labels,train_annotations,val_annotations=train_test_split(images,class_labels,annotations,test_size=0.2)
    
        
              
#Placeholder          
inputs_=tf.placeholder(tf.float32,[None,175,175,3],name='Input_images')
cls_input=tf.placeholder(tf.float32,[None,120],name='Class_inputs')
bbx_input=tf.placeholder(tf.float32,[None,4],name='bbx_input')



def conv_layer(x,filter_size,kernel_size,strides,padding):
    return tf.layers.conv2d(x,filters=filter_size,kernel_size=kernel_size,activation=tf.nn.relu,strides=strides,
                            padding=padding,kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),use_bias=True)
def max_pool(x,pool_size,strides):
    return tf.layers.max_pooling2d(x,pool_size=pool_size,strides=strides)


###VGG16 inspired
#64
x=conv_layer(inputs_,64,3,1,'valid')
x=conv_layer(x,64,3,1,'valid')
x=tf.layers.batch_normalization(x)
x=max_pool(x,2,2)

#128
x=conv_layer(x,128,3,1,'valid')
x=conv_layer(x,128,3,1,'valid')
x=tf.layers.batch_normalization(x)
x=max_pool(x,2,2)

#256
x=conv_layer(x,256,3,1,'valid')
x=conv_layer(x,256,3,1,'valid')
x=conv_layer(x,256,3,1,'valid')
x=tf.layers.batch_normalization(x)
x=max_pool(x,2,2)

#512
x=conv_layer(x,512 ,3,1,'valid')
x=conv_layer(x,512,3,1,'valid')
x=conv_layer(x,512,3,1,'valid')
x=tf.layers.batch_normalization(x)
x=max_pool(x,2,2)

#Fully Connected
flat=tf.layers.flatten(x)
FC1=tf.layers.dense(flat,1024,activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
drop1=tf.layers.dropout(FC1,0.4)
FC2=tf.layers.dense(drop1,4096,activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
drop2=tf.layers.dense(FC2,0.5)

clf_logits=tf.layers.dense(drop2,120,activation=None,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
reg_logits=tf.layers.dense(drop2,4,activation=None,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())


clf_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=clf_logits,labels=cls_input))
reg_loss=tf.reduce_mean(tf.square(reg_logits - bbx_input))

total_loss=clf_loss+reg_loss

train=tf.train.AdamOptimizer(0.001).minimize(total_loss)

acc=tf.metrics.accuracy(labels=tf.argmax(cls_input,1),predictions=tf.argmax(clf_logits,1))


config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.allocator_type='BFC'

tf.add_to_collections('Classification_head',clf_logits)
tf.add_to_collections('Regression_head',reg_logits)

epochs = 20
batch_size=5

acc_list=[]
auc_list=[]
loss_list=[]
val_accuracy=[]
val_losses=[]
regression_list=[]
saver=tf.train.Saver()

train_batches=len(train_images)//batch_size
val_batches=len(val_images)//batch_size

train_clf_loss,train_reg_loss,train_acc=[],[],[]
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(epochs):
        print('In Training..')
        for batch in tqdm(range(train_batches+1)):
            if batch==3292:
                tmp=train_images[batch*batch_size:]
                img_batch=np.reshape(tmp,(len(tmp),175,175,3))
                lbl_batch=train_labels[batch*batch_size:]
                bbx_batch=train_annotations[batch*batch_size:]
            else:
                tmp=train_images[batch*batch_size:(batch+1)*batch_size]
                img_batch=np.reshape(tmp,(len(tmp),175,175,3))
                lbl_batch=train_labels[batch*batch_size:(batch+1)*batch_size]
                bbx_batch=train_annotations[batch*batch_size:(batch+1)*batch_size]
            total_loss_,clf_loss_,reg_loss_,acc_,_=sess.run([total_loss,clf_loss,reg_loss,acc,train],feed_dict=
                                                {inputs_:img_batch,cls_input:lbl_batch,bbx_input:bbx_batch})
        
        
        print('In Validation')
        for batch_val in tqdm(range(val_batches)):
            temp_batch=val_images[batch*batch_size:(batch+1)*batch_size]
            val_image_batch=np.reshape(temp_batch,(len(temp_batch),175,175,3))
            val_label_batch=val_labels[batch*batch_size:(batch+1)*batch_size]
            val_acc,val_loss=sess.run([acc,total_loss],feed_dict={inputs_:val_images,
                                      cls_input:val_labels,
                                      bbx_input:val_annotations})
        val_accuracy.append(val_acc)
        val_losses.append(val_loss)
        train_clf_loss.append(clf_loss_)
        train_reg_loss.append(reg_loss_)
        train_acc.append(acc_)
        print('=======================================')
        print('Epoch: ',(epoch+1))
        print('Train Accuracy: ',(acc_))
        print('Total Train Loss: ',(total_loss_))
        print('Classifier Loss: ',(clf_loss_))
        print('Regression Loss: ',(reg_loss_))
        print('Validation')
        print('Accuracy: ',(val_acc))
        print('Loss: ',val_loss)
        print('=======================================')
    
    print('-----Training Finished\n----')
    saver.save(sess, os.path.join(os.getcwd(),"CNN_OL.ckpt"))            
            