#!/bin/bash
echo "3d inference"
OUTPUT_DIR="/home/eric/human-body-pose/video_output"
VIDEO_INPUT="/home/eric/human-body-pose/video_input"

rm /home/eric/Detectron/output/*

cd /home/eric/Detectron
python tools/infer_video.py \
    --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml \
    --output-dir output \
    --image-ext mp4 \
    --wts https://dl.fbaipublicfiles.com/detectron/37698009/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml.08_45_57.YkrJgP6O/output/train/keypoints_coco_2014_train:keypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    $VIDEO_INPUT

cd /home/eric/VideoPose3D/data
python prepare_data_2d_custom.py -i /home/eric/Detectron/output -o myvideos


cd /home/eric/VideoPose3D
for video in ${VIDEO_INPUT}/*
do
	THIS_VIDEO=${video##*/}
	echo THIS_VIDEO
	python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject $THIS_VIDEO --viz-action custom --viz-camera 0 --viz-video /home/eric/Detectron/input/$THIS_VIDEO --viz-output ${OUTPUT_DIR}/${THIS_VIDEO%.*}_video_output.mp4 --viz-export ${OUTPUT_DIR}/${THIS_VIDEO%.*}_numpy_output.npy --viz-size 6
done

cd /home/eric/human-body-pose
python3 main.py --height 185