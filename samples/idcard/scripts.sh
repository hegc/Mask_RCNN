# train
hpython idcard.py train --dataset=../../datasets/idcard --weights=coco --gpu=0,1,2,3

hpython idcard.py splash --weights=../../logs/idcard20181030T1823/mask_rcnn_idcard_0030.h5 --image=../../datasets/idcard/val/images/1006854_B.jpg
