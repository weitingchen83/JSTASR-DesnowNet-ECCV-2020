# JSTASR: Joint Size and Transparency-AwareSnow Removal Algorithm Based on ModifiedPartial Convolution and Veiling Effect Removal (Accepted by ECCV-2020)
**Wei-Ting Chen, Hao-Yu Feng, Jian-Jiun Ding, Chen-Che Tsai and Sy-Yen Kuo**  

* Wei-Ting Chen and Hao-Yu Feng share the equal contribution in this paper.
<br>

[[Paper Download]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660749.pdf)
[[Code Download]](https://github.com/weitingchen83/JSTASR-DesnowNet-ECCV-2020)  


![image](folder/ex.png)


We also develop a large-scale snow dataset which contains the veiling effect called "Snow Removal in Realistic Scenario (SRRS)". You can download from this [link](https://drive.google.com/file/d/1GX3e0ziUzBXnDtgeB5sHkgeWP1wUg5td/view?usp=sharing)


***
You can also refer our related works on dehazing:  

1."PMS-Net: Robust Haze Removal Based on Patch Map for Single Images" which has been published in **CVPR 2019**.

[[Paper Download]](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_PMS-Net_Robust_Haze_Removal_Based_on_Patch_Map_for_Single_CVPR_2019_paper.html)
[[Code Download]](https://github.com/weitingchen83/PMS-Net)
<br><br>and

2."PMHLD: Patch Map Based Hybrid Learning DehazeNet for Single Image Haze Removal" which has been published in **TIP 2020**.

[[Paper Download]](https://ieeexplore.ieee.org/document/9094006)
[[Code Download]](https://github.com/weitingchen83/Dehazing-PMHLD-Patch-Map-Based-Hybrid-Learning-DehazeNet-for-Single-Image-Haze-Removal-TIP-2020)


# Abstract:

Snow removal usually affects the performance of computer vision. Comparing with other atmospheric phenomenon (e.g., haze and rain), snow is more complicated due to its transparency, various size, and accumulation of veiling effect, which make single image de-snowing more challenging. In this paper, first, we reformulate the snow model. Different from that in the previous works, in the proposed snow model, the veiling effect is included. Second, a novel joint size and transparency-aware snow removal algorithm called JSTASR is proposed. It can classify snow particles according to their sizes and conduct snow removal in different scales. Moreover, to remove the snow with different transparency, the transparency-aware snow removal is developed. It can address both transparent and non-transparent snow particles by applying the modified partial convolution. Experiments show that the proposed method achieves significant improvement on both synthetic and real-world datasets and is very helpful for object detection on snow images.


# Setup and environment

To generate the recovered result you need:

1. Python 3 
2. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 9.0)
3. tensorflow 1.6.0
4. keras 2.2.0
5. cv2 3.4.4

Testing
```
$ python ./predict.py -dataroot ./your_dataroot -datatype datatype -predictpath ./output_path -batch_size batchsize
```
*datatype default: tif, jpg ,png

Example:
```
$ python ./predict.py -dataroot ./testImg -predictpath ./p -batch_size 3
$ python ./predict.py -dataroot ./testImg -datatype tif -predictpath ./p -batch_size 3
```
The pre-trained model can be found at this [website](https://drive.google.com/drive/folders/1GcJr4gOqO6ltjJY0Q94wQn-Ru4O09OK5?usp=sharing).
Please download three pre-trained models to the folder "modelParam".

# Citations
Please cite this paper in your publications if it helps your research.  

Bibtex:
```
@inproceedings{JSTASRChen,
  title={JSTASR: Joint Size and Transparency-Aware Snow Removal Algorithm Based on Modified Partial Convolution and Veiling Effect Removal},
  author={Chen, Wei-Ting and Fang, Hao-Yu and Ding, Jian-Jiun and Tsai, Chen-Che and Kuo, Sy-Yen},
  booktitle={European Conference on Computer Vision},
  year={2020}
}

```
