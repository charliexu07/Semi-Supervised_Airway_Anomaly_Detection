# Semi-Supervised Airway Anomaly Detection

This repository introduces semi-supervised learning to the method proposed in the paper ["Airway Anomaly Detection by Prototype-based Graph Neural Network"](https://drive.google.com/file/d/18dsJLTJGDQf9x0yMYefQkMaCoMQjTLQE/preview)

[comment]: <> (## Abstract)

[comment]: <> (Detecting the airway anomaly can be an essential part to aid the lung disease diagnosis. Since normal human airways share an anatomical structure, we design a graph prototype whose structure follows the normal airway anatomy. Then, we learn  the prototype and a graph neural network from a weakly-supervised airway dataset, i.e., only the holistic label is available, indicating if the airway has anomaly or not, but which bronchus node has the anomaly is unknown. During inference, the graph neural network predicts the anomaly score at both the holistic level and node-level of an airway. We initialize the airway anomaly detection problem by creating a large airway dataset with 2589 samples, and our prototype-based graph neural network shows high sensitivity and specificity on this new benchmark dataset. )


[comment]: <> (## Reference)

[comment]: <> (    @paper{zhao2018pyramid,)

[comment]: <> (    title={Airway Anomaly Detection by Prototype-based Graph Neural Network},)

[comment]: <> (    author={Zhao, Tianyi and Yin, Zhaozheng},)

[comment]: <> (    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention}, )

[comment]: <> (    year={2021},)

[comment]: <> (    organization={Springer})

[comment]: <> (    })

## Environment
```
Python 3.8.5
PyTorch 1.10.0
CUDA 10.2
```
 
[comment]: <> (## Data)

[comment]: <> (The segmented airway mask is given in the data/nii folder, in the NifTi format.)

[comment]: <> (The processed feature vectors are saved in the data folder, in the npy format.)

## Command
First, split the training and validation dataset
```
python split_dataset.py
```

Pre-train the model
```
python train.py
```

Fine-tune the model
```
python finetune.py
```
 
Evaluate the model
```
python test.py
```

After train/finetune/test, rename the "model" folder to a specific name and create a new "model" folder for the next round of training

Optional: supervised training
```
python supervised.py
```
