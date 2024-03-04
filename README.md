# Patch auxiliary generative adversarial network (PAGAN)


---
# Abstract
Hip fractures represent a significant public health concern, especially for the elderly. To make a diagnosis, pelvic radiographs (PXRs) are crucial and frequently employed for evaluating hip fractures. In order to enhance the accuracy of hip fracture diagnosis, we propose a classification model in conjunction with an auxiliary module, which called the Patch-Auxiliary Generative Adversarial Network (PAGAN), for hip fracture classification. PAGAN combines both global information, the full PXR image, and local information, the hip region patch, for learning. Additionally, any state-of-the-art (SOTA) classification model, such as EfficientNetB0, ResNet50, and DenseNet121, can be embedded in PAGAN for hip fracture detection. Our results demonstrate that the performances of these models improves after adopting PAGAN. Furthermore, in order to evaluate the explainability of the classification model after integrating PAGAN, we propose a GradCAM based method for quantifying the model attention region. The results also support the idea that PAGAN increases the model's attention to the region of interest.




<!-- # Framework

<p align="center">
  <img src="https://github.com/NYCUciflab/PAGAN_PXR/blob/main/figure/framework.png" />
</p> -->



# GradCAM for visualization

<p align="center">
  <img src="https://github.com/NYCUciflab/PAGAN_PXR/blob/main/figure/PXR_heatmap.png" />
</p>




# Usage

#### 1. Data Folder Structure: Before training, we need to arrange the image folders in the below Configuration.
```
Dataset.
       |- Train.
               |- fracture
               |- non-fracture

        - Valid.
               |- fracture
               |- non-fracture
 
        - Test.
              |- fracture
              |- non-fracture

        - Crop_data

```

#### 2. Install the requirements
```
pip install -r requirements.txt
```

#### 3. Modify the folder path and hyperparameters in "config.py"

The parameters are listed in file "config.py", you can modify the hyperparameters and the image folder here.

```
i.g.

##### INFORMATION #####
testing_code     = False
annotaton        = 'test_1'
device           = 'cuda:0'
model_name       = 'efficientnet_b0'
epochs           = 100
learning_rate    = 0.0001
batch_size       = 8
image_size       = 512

# image folder
train_dir   = 'ENTER YOUR TRAINING FORDER PATH'
valid_dir   = 'ENTER YOUR VALIDATION FORDER PATH'
test_dir    = 'ENTER YOUR TEST FORDER PATH'
crop_folder = 'ENTER YOUR CROP FOLDER PATH'

```

#### 4. Training and testing

Run the training process
```
python train.py
```

#### 5. Testing
In "visualization.ipynb", you can calculate and visualize the testing performance.