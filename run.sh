# 1
echo "* 1 ************************************************************************************************************************************************************************"
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR "./output/t1"

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --config-file ./configs/OWOD/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OWOD.TEMPERATURE 2.0 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "/home/tjl/OWOD/output/t1/model_final.pth" 

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0005 OUTPUT_DIR "./output/t1_final" MODEL.WEIGHTS "/home/tjl/OWOD/output/t1/model_final.pth"

# 2 
echo "* 2 ************************************************************************************************************************************************************************"
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR "./output/t1"
mkdir /home/tjl/OWOD/output/t2
cp -r /home/tjl/OWOD/output/t1/model_final.pth /home/tjl/OWOD/output/t2/model_final.pth
cp -r /home/tjl/OWOD/output/t1/last_checkpoint /home/tjl/OWOD/output/t2/last_checkpoint
cp -r /home/tjl/OWOD/output/t1/metrics.json /home/tjl/OWOD/output/t2/metrics.json

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1  --resume --config-file ./configs/OWOD/t2/t2_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR "./output/t2" MODEL.WEIGHTS "/home/tjl/OWOD/output/t2/model_final.pth"

mkdir /home/tjl/OWOD/output/t2_ft
cp -r /home/tjl/OWOD/output/t2/model_final.pth /home/tjl/OWOD/output/t2_ft/model_final.pth
cp -r /home/tjl/OWOD/output/t2/last_checkpoint /home/tjl/OWOD/output/t2_ft/last_checkpoint
cp -r /home/tjl/OWOD/output/t2/metrics.json /home/tjl/OWOD/output/t2_ft/metrics.json

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t2/t2_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR "./output/t2_ft" MODEL.WEIGHTS "/home/tjl/OWOD/output/t2_ft/model_final.pth"

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --config-file ./configs/OWOD/t2/t2_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OWOD.TEMPERATURE 2.0 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "/home/tjl/OWOD/output/t2_ft/model_final.pth"

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t2/t2_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0005 OUTPUT_DIR "./output/t2_final" MODEL.WEIGHTS "/home/tjl/OWOD/output/t2_ft/model_final.pth"

# 3
echo "* 3 ************************************************************************************************************************************************************************"
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR "./output/t1"
mkdir /home/tjl/OWOD/output/t3
cp -r /home/tjl/OWOD/output/t2_ft/model_final.pth /home/tjl/OWOD/output/t3/model_final.pth
cp -r /home/tjl/OWOD/output/t2_ft/last_checkpoint /home/tjl/OWOD/output/t3/last_checkpoint
cp -r /home/tjl/OWOD/output/t2_ft/metrics.json /home/tjl/OWOD/output/t3/metrics.json

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1  --resume --config-file ./configs/OWOD/t3/t3_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR "./output/t3" MODEL.WEIGHTS "/home/tjl/OWOD/output/t3/model_final.pth"

mkdir /home/tjl/OWOD/output/t3_ft
cp -r /home/tjl/OWOD/output/t3/model_final.pth /home/tjl/OWOD/output/t3_ft/model_final.pth
cp -r /home/tjl/OWOD/output/t3/last_checkpoint /home/tjl/OWOD/output/t3_ft/last_checkpoint
cp -r /home/tjl/OWOD/output/t3/metrics.json /home/tjl/OWOD/output/t3_ft/metrics.json

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t3/t3_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR "./output/t3_ft" MODEL.WEIGHTS "/home/tjl/OWOD/output/t3_ft/model_final.pth"

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --config-file ./configs/OWOD/t3/t3_val.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OWOD.TEMPERATURE 2.0 OUTPUT_DIR "./output/t3_final" MODEL.WEIGHTS "/home/tjl/OWOD/output/t3_ft/model_final.pth"

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t3/t3_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0005 OUTPUT_DIR "./output/t3_final" MODEL.WEIGHTS "/home/tjl/OWOD/output/t3_ft/model_final.pth"

# 4
echo "* 4 ************************************************************************************************************************************************************************"
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR "./output/t1"
mkdir /home/tjl/OWOD/output/t4
cp -r /home/tjl/OWOD/output/t3_ft/model_final.pth /home/tjl/OWOD/output/t4/model_final.pth
cp -r /home/tjl/OWOD/output/t3_ft/last_checkpoint /home/tjl/OWOD/output/t4/last_checkpoint
cp -r /home/tjl/OWOD/output/t3_ft/metrics.json /home/tjl/OWOD/output/t4/metrics.json

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1  --resume --config-file ./configs/OWOD/t4/t4_train.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR "./output/t4" MODEL.WEIGHTS "/home/tjl/OWOD/output/t4/model_final.pth"

mkdir /home/tjl/OWOD/output/t4_ft
cp -r /home/tjl/OWOD/output/t4/model_final.pth /home/tjl/OWOD/output/t4_ft/model_final.pth
cp -r /home/tjl/OWOD/output/t4/last_checkpoint /home/tjl/OWOD/output/t4_ft/last_checkpoint
cp -r /home/tjl/OWOD/output/t4/metrics.json /home/tjl/OWOD/output/t4_ft/metrics.json

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --resume --config-file ./configs/OWOD/t4/t4_ft.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.001 OUTPUT_DIR "./output/t4_ft" MODEL.WEIGHTS "/home/tjl/OWOD/output/t4_ft/model_final.pth"


CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t4/t4_test.yaml SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0005 OUTPUT_DIR "./output/t4_final" MODEL.WEIGHTS "/home/tjl/OWOD/output/t4_ft/model_final.pth"