# NCTU_CS_T0828_HW3-Instance segmentation
## Introduction
The proposed challenge is Tiny VOC dataset contains only 1,349 training images, 100 test images with 20 common object classes.
Train dataset | Test dataset
------------ | ------------- |
1.349 images | 100 images
## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-7500U CPU @ 2.90GHz
- 2x NVIDIA 2080Ti
## Data pre-process
I used data downloaded from TA’s drive containing 1,349 training data and 100 testing data. Somehow, I need validation data to train the model but ‘test.json’ has no image id and annotation key. So, I took only 10 training data wich has annotation key as validation data to train faster and set the validation step in config.py. Futher, I fixed the ‘pascal_train.json’ format into coco annotation format. All data conversion processed are in the **jsonconversion.py** file.
 ```
 $ jsonconversion.py
 ```
After transform, the structure becomes like this:
```
+- hw3
|  +- samples
|    +- coco 
|      +- pascal
|        +- annotations
|        +- train2017
|        +- val2017
|      coco.py
|    demo.py
|  +- mrcnn
|    +- _pycache_
|    __init__.py 
|    config.py
|    model.py  	 
|    parallel_model.py
|    utils.py
|    visualize.py
|  README.md
|  requirements.txt
|  setup.py
```

## Training
You can set the config in **coco.py** (class CoCoConfig). The config template is in **mrcnn/config.py**.
### Configure the environment
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.

I used cuda 10.0 with tensorflow-gpu==1.14 and keras==2.2.5
```
$ conda create -n mrnn python=3.6
$ source activate mrnn
$ pip install -r requirements.txt
```
### Model architecture(Callbacks, Optimizer, Regularizer)
You can set them in **coco.py** or **model.py(recommend)**
### Train models
To train models, run following commands.
```
$ python coco.py train --dataset pascal --model imagenet
```
### Pretrained models
Only can use **ImageNet** pretrained model for the fairness.
### Training weights
The training weights are saved in **logs** directory.
## Testing
### Test model
Using the **demo.py** to test your trained model.
```
$ python3 demo.py
```
You will get a test_result json file.
## Submission
Submit the test_result json file, get the score.
