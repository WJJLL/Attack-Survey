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
