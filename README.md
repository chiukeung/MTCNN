# MTCNN
MTCNN on caffe
Description
This work is about MTCNN implemented on Caffe and it has function of drawing boundingbox without five landmarks. 

Prerequisities
1. GPUs to train models
2. Download WIDER FACE and FDDB for training and testing model.

Dependencies
1. Caffe
2. Python 2.7
3. Ubuntu 16.04
4. CUDA 8.0


Prepare:
1. add mtcnn_euclidean_loss_layer.hpp to CAFFE_ROOT/include/caffe/layers
2. add mtcnn_euclidean_loss_layer.cpp and mtcnn_euclidean_loss_layer.cu to CAFFE_ROOT/src/caffe/layers
3. sudo make clean
4. sudo make -j8

Training process:
train pnet:
1. run gen_12net_data.py to generate pos, part, neg data 
Then you can choose one of two paths({1},{2}) to generate hd5 file:
{1}2. run extract_pnet.py and mtcnnform_pnet.py
   3. run gen_hdf5_pnet.py to generate hd5 file for training
{2}2. run merge_pnet.py
   3. run gen_hdf5_pnet_new.py
4. run train/Pnet_train/PNet_train.sh to train the pnet

train rnet:
1. run gen_12net_hard_example.py to generate neg data
2. run gen_24net_data.py to generate pos, part data
3. run merge_rnet.py to generate image list and label and bbox for generating hdf5 file
4. run gen_hdf5_rnet_new.py to generate hd5 file for training
5. run train/Rnet_train/RNet_train.sh to train the rnet

train onet:
1. run gen_24net_hard_example.py to generate neg data
2. run gen_48net_data.py to generate pos, part data
3. run merge_onet.py to generate image list and label and bbox for generating hdf5 file
4. run gen_hdf5_onet.py to generate hd5 file ofr training
5. run train/Rnet_train/ONet_train.sh to train the onet

Test process:
use FDDB dataset to calculate Recall
1. run testonFDDB/runFDDB_xnet.py which use the model we generate to generate predict.txt in fold FDDB-folds
2. modify detFile in evaluation/evaluate.cpp to the predict we generate
3. sudo make clean
4. sudo make
5. run the command in aboutFDDB/evaluate.sh(plz remember to add sudo before command) to generate \
   ContROC.txt and DiscROC.txt where we can get the Recall from the first colomn in DiscROC.txt

Additionally, we can use testonFDDB/demo.py to test the real picture.

Some details:
1. hd5 file contains three parts of data(pos,part,neg) and the ratio is pos:part:neg=1:1:3. for pnet, pos data is 19w+. 
for rnet, pos data is 40w+. for onet, pos data is 19K+.
2. the format of data is:
[image_path][class_label][bbox_label];
pos example:  class_label=1, bbox_label=[0.1,0.2,0.3,0.4];
part example: class_label=-1,bbox_label=[0.2,0.3,0.4,0.5];
neg example:  class_label=0, bbox_label=[0,0,0,0];
 

Reference:
1. kpzhang
2. foreverYoungGitHub
3. samylee
4. dlunion
5. AITTSMD
6. hualitlc
7. DuinoDu
8. CongWeilin

