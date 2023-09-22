#!/bin/bash

# Fixed Params
PRETRAINED_DATASET="imagenet"
EPSILON=0.03922
LOSS_FNS='neg_ce bounded_logit_neg cos_sim r_ce cell feature_layers'
CONFIDENCE=10
BATCH_SIZE=20
LEARNING_RATE=0.005
NUM_EPOCHS=5
WORKERS=4
NGPU=1
SUBF="imagenet_untargeted"

TARGET_NETS="vgg16 vgg19 resnet50"
for target_net in $TARGET_NETS; do
  for ls in $LOSS_FNS; do
   CUDA_VISIBLE_DEVICES=0 python3 train_uap.py \
      --dataset coco \
      --pretrained_dataset $PRETRAINED_DATASET --pretrained_arch $target_net \
      --epsilon $EPSILON \
      --loss_function $ls \
      --num_epochs $NUM_EPOCHS \
      --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
      --workers $WORKERS --ngpu $NGPU \
      --result_subfolder $SUBF \
	  --postfix $ls

	CUDA_VISIBLE_DEVICES=0 python3 train_uap.py \
      --dataset imagenet \
      --pretrained_dataset $PRETRAINED_DATASET --pretrained_arch $target_net \
      --epsilon $EPSILON \
      --loss_function $ls \
      --num_epochs $NUM_EPOCHS \
      --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
      --workers $WORKERS --ngpu $NGPU \
      --result_subfolder $SUBF \
	  --postfix $ls
	done
done
