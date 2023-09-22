#!/bin/bash

# Fixed Params
PRETRAINED_DATASET="imagenet"
EPSILON=0.03922
LOSS_FN="bounded_logit_fixed_ref ce t_r_ce"
CONFIDENCE=10
BATCH_SIZE=20
LEARNING_RATE=0.005
NUM_EPOCHS=5
WORKERS=2
NGPU=1
SUBF="imagenet_targeted"

TARGET_CLASSES="24 99 245 344 471 555 661 701"

TARGET_NETS="resnet50 vgg16 vgg19"
for net in $TARGET_NETS; do
  for ls in $LOSS_FN; do
      for target_class in $TARGET_CLASSES; do
      CUDA_VISIBLE_DEVICES=1 python3 train_uap.py \
     --dataset imagenet \
     --pretrained_dataset $PRETRAINED_DATASET --pretrained_arch $net \
     --target_class $target_class --targeted\
     --epsilon $EPSILON \
     --loss_function $ls --confidence $CONFIDENCE \
      --num_epochs $NUM_EPOCHS \
     --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
     --workers $WORKERS --ngpu $NGPU \
     --result_subfolder $SUBF \
     --postfix $ls
      done
    done
done


TARGET_NETS="vgg16 googlenet resnet18 densenet121"
for net in $TARGET_NETS; do
    for ls in $LOSS_FN; do
      for target_class in $TARGET_CLASSES; do
      CUDA_VISIBLE_DEVICES=0 python3 eval.py \
      --dataset coco \
      --source_arch vgg19 \
      --target_class $target_class --targeted\
      --pretrained_dataset $PRETRAINED_DATASET --pretrained_arch $net \
      --epsilon $EPSILON \
      --loss_function $ls \
      --num_epochs $NUM_EPOCHS \
      --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
      --workers $WORKERS --ngpu $NGPU \
      --result_subfolder $SUBF \
	    --postfix $ls
	  done
	done
done

