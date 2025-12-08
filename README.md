# PFF-Net: Patch Feature Fitting for Point Cloud Normal Estimation (TVCG 2025)

### **[Paper (arXiv)](https://arxiv.org/abs/2511.21365)**

Estimating the normal of a point requires constructing a local patch to provide center-surrounding context, but determining the appropriate neighborhood size is difficult when dealing with different data or geometries. Existing methods commonly employ various parameter-heavy strategies to extract a full feature description from the input patch. However, they still have difficulties in accurately and efficiently predicting normals for various point clouds. In this work, we present a new idea of feature extraction for robust normal estimation of point clouds. We use the fusion of multi-scale features from different neighborhood sizes to address the issue of selecting reasonable patch sizes for various data or geometries. We seek to model a patch feature fitting (PFF) based on multi-scale features to approximate the optimal geometric description for normal estimation and implement the approximation process via multi-scale feature aggregation and cross-scale feature compensation. The feature aggregation module progressively aggregates the patch features of different scales to the center of the patch and shrinks the patch size by removing points far from the center. It not only enables the network to precisely capture the structure characteristic in a wide range, but also describes highly detailed geometries. The feature compensation module ensures the reusability of features from earlier layers of large scales and reveals associated information in different patch sizes. Our approximation strategy based on aggregating the features of multiple scales enables the model to achieve scale adaptation of varying local patches and deliver the optimal feature description. Extensive experiments demonstrate that our method achieves state-of-the-art performance on both synthetic and real-world datasets with fewer network parameters and running time.

## Dataset
We train our network model on the PCPNet dataset.
The used datasets can be downloaded from [here](https://drive.google.com/drive/folders/1eNpDh5ivE7Ap1HkqCMbRZpVKMQB1TQ6H?usp=share_link).
Unzip them to a folder `***/Dataset/` and set the value of `dataset_root` in `run.py`.
The dataset is organized as follows:
```
│Dataset/
├──PCPNet/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
│
├──FamousShape/
│  ├── list
│      ├── ***.txt
│  ├── ***.xyz
│  ├── ***.normals
│  ├── ***.pidx
```

## Train
Our trained model is provided in `./log/000/ckpts/ckpt_800.pt`.
To train a new model on the PCPNet dataset, simply run:
```
python run.py --gpu=0 --mode=train
```
Your trained model will be save in `./log/***/`.

## Test
You can use the provided model for testing:
- PCPNet dataset
```
python run.py --gpu=0 --mode=test --data_set=PCPNet
```
- FamousShape dataset
```
python run.py --gpu=0 --mode=test --data_set=FamousShape
```
The evaluation results will be saved in `./log/000/results_***/ckpt_800/`.
To test with your trained model, simply run:
```
python run.py --gpu=0 --mode=test --data_set=*** --ckpt_dirs=*** --ckpt_iters=***
```
To save the normals of the input point cloud, you need to change the variables in `run.py`:
```
save_pn = True          # to save the point normals as '.normals' file
sparse_patches = False  # to output sparse point normals or not
```

## Citation
If you find our work useful in your research, please cite our paper:

    @article{li2025pffnet,
      author    = {Li, Qing and Feng, Huifang and Shi, Kanle and Gao, Yue and Fang, Yi and Liu, Yu-Shen and Han, Zhizhong},
      title     = {{PFF-Net}: Patch Feature Fitting for Point Cloud Normal Estimation},
      booktitle = {IEEE Transactions on Visualization and Computer Graphics (TVCG)},
      year      = {2025},
    }