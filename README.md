# Video interpolation with Super SloMo

This TF2 implementation of [Super SloMo](https://arxiv.org/abs/1712.00080) is heavily inspired by another [PyTorch implementation](https://github.com/avinashpaliwal/Super-SloMo).

The following packages are required
 - tensorflow
 - tensorflow-addons
 - opencv-python

## Training

To train the models, run the following command

`python train.py `

with the following arguments:
 - -v: path to the folder containing the training videos (don't specify if the dataset has already been generated)
 - -d: path to the dataset containing the decompressed videos
 - -w: path to the weights (this folder contains the checkpoints of the models, saved after each epoch)
 - -e: number of epochs
 - -b: batch size (default: 7)
 - -l: learning rate (default: 0.00002)
 - -i: in between frames between the high and low frame-rate videos (default: 7)
 - -c: crop size of the random crop used for data augmentation if specified (optional)
 - -p: the last checkpoint to load the weights from (optional)
 - -s: the number of samples per sequence, used to avoid over-fitting on longer sequences (default: 50)

The weights used are saved into `weights`. Note: the weights of a pre-trained VGG16 are required for the training phase and should be placed in the `weights` folder as `weights_vgg16.h5`. A pre-trained model and the weights of a VGG16 can be found [here](https://drive.google.com/file/d/1WFaWKFBUemx_tsX2wS7fwpzzBFfgZ5m0/view?usp=share_link).

## Up-sample

To up-sample a video, run the following command:

`python upsample.py`

with the following arguments
 - -i: the path to the input video
 - -o: the path to the output video
 - -f: the temporary frame folder (leave unspecified if you don't need to retrieve the frames)
 - -mf: the number of frame to extract from the input video
 - -p: the model checkpoint to use
 - -b: batch size
 - -r: the up-sampling rate
 - -fps: the frame rate of the output video
