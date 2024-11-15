# Project Description

### Inference results for USA downstreet video streaming

<a href="https://imgflip.com/gif/3hoy1q"><img src="https://i.imgflip.com/3hoy1q.gif" title="made at imgflip.com"/></a>
<a href="https://imgflip.com/gif/3howsk"><img src="https://i.imgflip.com/3howsk.gif" title="made at imgflip.com"/></a>

### Feature extraction and measurement of butterflies and orchids

- Extraction for average blackness of butterfly eyespot through Mask RCNN.

![擷取1](https://user-images.githubusercontent.com/31026907/82220040-b74e0b00-9950-11ea-91df-419cf123ddf2.PNG)

- Extraction of orchid featuremap from different layers of Residual Network (ResNet).

![擷取5](https://user-images.githubusercontent.com/31026907/82219899-853ca900-9950-11ea-8ff9-b1f024486807.PNG)

- Extraction of orchid featuremap from different layers of Feature Pyramid Network (FPN).

![擷取4](https://user-images.githubusercontent.com/31026907/69710629-b1facb00-113a-11ea-808b-15d45ac37bb1.PNG)

- The Average-Precision (AP) and training time for different Mask RCNN models (orchids).

![擷取6](https://user-images.githubusercontent.com/31026907/82221465-9dadc300-9952-11ea-879b-a3925689addd.PNG)

### Extraction for orchid root length and leaf width using skeleton extraction algorithm and min-rectangle area method via output masks

- Skeleton extraction algorithm (image morphology).

![擷取新](https://user-images.githubusercontent.com/31026907/215658947-0ad23de9-fe06-47f1-9746-93e2c995944c.PNG)

- Min-rectangle area method.

![擷取8](https://user-images.githubusercontent.com/31026907/82225372-b371b700-9957-11ea-888c-58620629b87d.PNG)

### Inference results

![image](https://user-images.githubusercontent.com/31026907/69708103-fa63ba00-1135-11ea-807f-002e5ea3c021.png)

### Different values for the features of orchids measured via output boundingboxes, masks, and classes

![擷取2](https://user-images.githubusercontent.com/31026907/69709188-ed47ca80-1137-11ea-814c-f2fd75cd9940.PNG)

# Setup & run code

### Getting started

- Clone this repo to your local

```
git clone https://github.com/simonyang0608/Mask_RCNN_Project
cd Mask_RCNN_Project
```

### Computer equipments

- System: Ubuntu16.04
- Python version: Python 3.5 or higher
- Keras version: Keras 1.8.4
- Tensorflow version: Tensorflow 1.8.0
- Training: \
  CPU: Intel Xeon E5-2698 Dual 20 cores @ 42.2 GHz \
  GPU: NVIDIA Tesla V100 16GB*8
- Testing & inference: \
  CPU: Intel(R) Core(TM)i5-7300HQ CPU @ 2.5.0 GHz \
  GPU: NVIDIA GeForce GTX 1050 2GB
  
### Training

- You should download the released pretrained models. And put the model on the folder ./mrcnn_model to train related models continously.

```
python3 mrcnn_model/parallel_model.py
```

### Testing & inference

- You should download the released pretrained models. And put the model on the folder ./mrcnn_model to inference the test set on the folder ./pic.
  
- Orchids dataset or video streaming
  
```
python3 mask_rcnn_create_live_stream.py
```

- Butterfly dataset

```
python3 mask_rcnn_detect_butterfly.py
```

# To get the released pretrained model
  https://github.com/simonyang0608/Mask_RCNN_Project/releases/tag/1

