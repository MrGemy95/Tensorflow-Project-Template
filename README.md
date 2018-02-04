# Tensorflow Project Templete
A simple and well designed OOP structure is essential for any Deep Learning project, so after a lot of practice and contributing in tensorflow projects here's a tensorflow project templete that combines   **simplcity**, **best practice for folder structure** and **good OOP design**.
The main idea is that there's much stuff you do every time you start your tensorflow project so wrapping all this shared stuff will help you to get into the main project faster.

## Table Of Content
-  [Project architecture](https://github.com/Mrgemy95/Tensorflow-Project-Templete#project-architecture)
-  [Folder structure](https://github.com/Mrgemy95/Tensorflow-Project-Templete#folder-structure)
-  [ Main Components](https://github.com/Mrgemy95/Tensorflow-Project-Templete#main-components)
    -  [Model](https://github.com/Mrgemy95/Tensorflow-Project-Templete#model)
    -  [Trainer](https://github.com/Mrgemy95/Tensorflow-Project-Templete#ptrainer)
    -  [Data Loader](https://github.com/Mrgemy95/Tensorflow-Project-Templete#data-loader)
    -  [Logger](https://github.com/Mrgemy95/Tensorflow-Project-Templete#logger)
-  [references](https://github.com/Mrgemy95/Tensorflow-Project-Templete#references)

Project architecture 
--------------


Folder structure
--------------

```
├──  base
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── ease_trainer.py - this file contains the abstract class of the trainer.
│
│
├── model               -This folder contains any model of your project.
│   └── xample_model.py
│
│
├── trainer             -this folder contains trainers of your project.
│   └── example_trainer.py
│   
├──  mains              - here's the main/s of your project (you may need more than one main.
│                         
│  
├──  data _loader  
│    └── data_generator.py  - here's the data_generator that responsiple for all data handling.
│ 
└── utils
     ├── logger.py
     └── llny_other_utils_you_need

```

--------------
## Main Components
--------------
### Models
--------------
##### base model
--------------
Base model is an abstract class that must be inherted by any model you create, the idea behind this is that there's much shared stuff between all models.
The base model contains:
1. **Save** -This function to save a checkpoint to the desk. 
2. **Load** -This function to load a checkpoint from the desk.
3. **Cur_epoch, Global_step counters** -These variables to keep track of the curerent epoch and global step.
4. **Init_Saver** An abstract function to inialize the saver used for saving and loading the checkpoint, **Note**: override this function in the model you want to implement.
5. **Build_model** Here's an abstract function to define the model, **Note**: override this function in the model you want to implement.
##### Your model
Here's where you implement your model.
So you should :
1. 
######
#### Trainer
#### Data Loader
#### Logger


### References

