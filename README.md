# LeukoVision: Identifying white blood cells from microscopic images

This project aims to build a deep learning model to identify different types of WBCs. 
The data that use as a referance for our model is introduced in the following publication:
https://doi.org/10.1016/j.cmpb.2019.105020 & https://www.nature.com/articles/s42256-019-0101-9
and the data is avaliable to download by the following link:
https://www.kaggle.com/datasets/ahmedali42/dataset-spain-germany

## What each folder do?

The folder structure is as follow:

    root/ ->contains Readme, gitignore, requirements and all subfolders below
    |_src
    |_Notebooks
    |   |_inceptionv3
    |   |_resnet50
    |   |_vgg16
    |_Streamlit

## How to use our models?

To use our model to predict your WBCs, first, we recommend you to use our `requirements.txt` file to install all the dependant 
python packages on a new enviroment to make sure that you don't run into any technical problems. 

Second, you can download our models (h5 or pth file format) and loaded to identify your images.