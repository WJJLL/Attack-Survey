# # TO train target attackers
# well-trained generator is saved to args.expname
# training data is coco (args.train_dir)
# source model is args.foolmodel
TARGET_NETS="vgg16 vgg19 resnet50"
TargetCls='24 99 245 344 471 555 661 701'
LOSS_FN="bounded_logit_fixed_ref ce t_r_ce"
for net in $TARGET_NETS; do
  for ls in $LOSS_FN; do
    for tg in $TargetCls; do
        python3 GAP_clf.py \
        --train_dir coco \
        --loss_function $ls \
        --expname targeted/coco_${net}_${ls} \
        --batchSize 20 --testBatchSize 10 --mag_in 10 --foolmodel $net --mode train \
        --nEpochs 5 \
        --MaxIter 1000 \
        --perturbation_type imdep --target $tg --gpu_ids 0
    done
  done
done


# # evaluate performance of targeted attacks
# target_model : args.foolmodel
# expname: path of well-trained generator
LOSS_FN="bounded_logit_fixed_ref ce t_r_ce"
TARGET_NETS="resnet18"
for ls in $LOSS_FN; do
  for target_model in $TARGET_NETS ; do
    python3 eval_all.py --expname targeted/imagenet_vgg19_${ls} \
--batchSize 20 --testBatchSize 10 --mag_in 10 --mode test \
--nEpochs 5 --foolmodel $target_model \
--MaxIter 1000  --checkpoint \
--perturbation_type imdep --target -1 --gpu_ids 0 >img_vgg19_${target_model}_${ls}.txt
  done
done


# # TO train non-target attackers
TARGET_NETS="vgg16 vgg19 resnet50"
LOSS_FN='neg_ce bounded_logit_neg cos_sim r_ce cell feature_layers'
for net in $TARGET_NETS; do
  for ls in $LOSS_FN; do
      python3 GAP_clf.py \
        --train_dir imagenet \
        --loss_function $ls \
        --expname untargeted/imagenet_${net}_${ls} \
        --batchSize 20 --testBatchSize 10 --mag_in 10 --foolmodel $net --mode train \
        --nEpochs 5 \
        --MaxIter 1000 \
        --perturbation_type imdep --target -1 --gpu_ids 1  >untargeted/imagenet_${net}_${ls}.txt
   done
done

# # evaluate performance of non-targeted attacks
# expname: path of well-trained generator
LOSS_FN='neg_ce bounded_logit_neg cos_sim r_ce cell feature_layer'
TARGET_NETS='resnet50'
for ls in $LOSS_FN; do
  for target_model in $TARGET_NETS ; do
  python3 GAP_clf.py \
  --expname untargeted/imagenet_resnet50_${ls} \
  --batchSize 20 --testBatchSize 10 --mag_in 10 --mode test \
  --nEpochs 5 --foolmodel $target_model \
  --MaxIter 1000  --checkpoint  --nrp \
  --perturbation_type imdep --target -1 --gpu_ids 0  >img_resnet50_${target_model}_${ls}.txt
  done
done

