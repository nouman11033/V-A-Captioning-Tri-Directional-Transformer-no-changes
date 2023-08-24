make a folder called submodels in this repo after clone
within make another 2 folders :
1. "pycocoevalcap"
   https://github.com/salaniz/pycocoevalcap/tree/ca1b05fa0e99f86de2259f87d43c72b322523731
2. "video features"
   https://github.com/v-iashin/video_features.git


This is a repo with 3 transformer/layers :
a. based on : https://github.com/v-iashin/BMT.git , https://github.com/mbzuai-oryx/Video-ChatGPT , 
b. Trial Changed (colab) : 
c. Bi directional Trial (colab) : https://colab.research.google.com/github/v-iashin/BMT/blob/master/colab_demo_BMT.ipynb


Start by extracting audio and visual features from your video using video_features repository. This repo is also included in ./submodules/video_features (commit 662ec51caf591e76724237f0454bdf7735a8dcb1).

Extract I3D features

# run this from the video_features folder:
cd ./submodules/video_features
conda deactivate
conda activate i3d
python main.py \
    --feature_type i3d \
    --on_extraction save_numpy \
    --device_ids 0 \
    --extraction_fps 25 \
    --video_paths ../../sample/women_long_jump.mp4 \
    --output_path ../../sample/
Extract VGGish features (if ValueError, download the vggish model first--see README.md in ./submodules/video_features)

conda deactivate
conda activate vggish
python main.py \
    --feature_type vggish \
    --on_extraction save_numpy \
    --device_ids 0 \
    --video_paths ../../sample/women_long_jump.mp4 \
    --output_path ../../sample/
Run the inference

# run this from the BMT main folder:
cd ../../
conda deactivate
conda activate bmt
python ./sample/single_video_prediction.py \
    --prop_generator_model_path ./sample/best_prop_model.pt \
    --pretrained_cap_model_path ./sample/best_cap_model.pt \
    --vggish_features_path ./sample/women_long_jump_vggish.npy \
    --rgb_features_path ./sample/women_long_jump_rgb.npy \
    --flow_features_path ./sample/women_long_jump_flow.npy \
    --duration_in_secs 35.155 \
    --device_id 0 \
    --max_prop_per_vid 100 \
    --nms_tiou_thresh 0.4
Expected output

[
  {'start': 0.1, 'end': 4.9, 'sentence': 'We see a title screen'},
  {'start': 5.0, 'end': 7.9, 'sentence': 'A large group of people are seen standing around a building'},
  {'start': 0.7, 'end': 11.9, 'sentence': 'A man is seen standing in front of a large crowd'},
  {'start': 19.6, 'end': 33.3, 'sentence': 'The woman runs down a track and jumps into a sand pit'},
  {'start': 7.5, 'end': 10.0, 'sentence': 'A large group of people are seen standing around a building'},
  {'start': 0.6, 'end': 35.1, 'sentence': 'A large group of people are seen running down a track while others watch on the sides'},
  {'start': 8.2, 'end': 13.7, 'sentence': 'A man runs down a track'},
  {'start': 0.1, 'end': 2.0, 'sentence': 'We see a title screen'}
]
Note that in our research we avoided non-maximum suppression for computational efficiency and to allow the event prediction to be dense. Feel free to play with --nms_tiou_thresh parameter: for example, try to make it 0.4 as in the provided example.

The sample video credits: Women's long jump historical World record in 1978

If you are having an error

RuntimeError: Vector for token b'<something>' has <some-number> dimensions, but previously read vectors
have 300 dimensions.
try to remove *.txt and *.txt.pt from the hidden folder ./.vector_cache/ and check if you are not running out of disk space (unpacking of glove.840B.300d.zip requires extra ~8.5G). Then run single_video_prediction.py again.