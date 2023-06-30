# MAP-Inspired Deep Unfolding Network for Distributed Compressive Video Sensing

This is the PyTorch implementation of the IEEE SIGNAL PROCESSING LETTERS paper: MAP-Inspired Deep Unfolding Network for Distributed Compressive Video Sensing (https://ieeexplore.ieee.org/abstract/document/10079086) by Xin Yang, Chunling Yang.

South China University of Technology

## Description

(1) The training models of each sampling rate are saved in the check_point folder

(2) The dataloder1.py file is the data loading code for keyframe network training

        1. Gop_size is set to 1, which means that each training only takes a random frame in a sequence
        
        2. image_size indicates the image block size
        
        3. The load_filename parameter represents the parameter path list of the training data
        
(3) The dataloder.py and dataloder2.py files are data loading codes for non-keyframe network training

        1. gop_size means GOP size + 1, if it is 8, then set the value to 9
        
        2. image_size indicates the image block size
        
        3. The load_filename parameter represents the parameter path list of the training data

(4) fnames_shuffle_test.npy, fnames_shuffle_train.npy and fnames_shuffle_val.npy represent the data path list of the training, testing and verification set data of the UCF-101 dataset

(5) The model_dumhan.py file implements the code for each model

        1. The sample_and_inirecon_key class is the key frame sampling and reconstruction network, num_filters indicates the number of sampling points, and B_size is the sampling block size
        
        2. The sample_and_inirecon_nonkey class is a non-key frame sampling and reconstruction network, num_filters indicates the number of sampling points, and B_size is the sampling block size
        
        3. num_of, num_of1, and num_of2 represent the number of hypotheses in the overall hypothesis set, reference frame 1 hypothesis set, and reference frame 2 hypothesis set, respectively
        
        4. backWarp_MH is multi-hypothesis reverse alignment
        
        5. synthes_net is a fusion network
        
        6. FLOWNET1_key and FLOWNET2_key are the key frame optical flow residual estimation network
        
        7. FLOWNET1_nonkey and FLOWNET2_nonkey are non-key frame optical flow residual estimation networks
        
        8. basic_block_b_key and basic_block_a_key are key frame basic modules
        
        9. basic_block_b_nonkey and basic_block_a_nonkey are non-key frame basic modules

(6) The test_dumhan.py file is the test code

        1. rgb indicates whether to test color images, if yes, set it to True
        
        2. flag indicates whether to load the trained model, set to True
        
        3. block_size indicates the sampling block size
        
        4. gop_size indicates the GOP size
        
        5. image_width and image_height indicate that when the resolution cannot be divisible by the sampling block size, fill with 0, and the image size after filling
        
        6. img_w and img_h represent the size of the original image
        
        7. num_gop is the number of GOP
        
        8. test_video_name is the test folder, which can be placed in the same level directory as the current py file
        
        9. sr means non-key frame sampling rate
        
(7) train_key_dumhan.py is the key frame training code

(8) train_nonkey_dumhan.py is the non-key frame training code

## Citation

If you find the code helpful in your resarch or work, please cite the following papers.

    @ARTICLE{10079086,
        author={Yang, Xin and Yang, Chunling},
        journal={IEEE Signal Processing Letters}, 
        title={MAP-Inspired Deep Unfolding Network for Distributed Compressive Video Sensing}, 
        year={2023},
        volume={30},
        number={},
        pages={309-313},
        doi={10.1109/LSP.2023.3260707}}
