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
│   ├── base_model.py  - this file contains the abstract class of the model
│   └── ease_trainer.py - this file contains the abstract class of the trainer
│
│
├── model               <-This folder contains any model of your project
│   └── xample_model.py
│
│
├── trainer           <-this folder contains trainers of your project
│   └── example_trainer.py
│   
├──  mains       <- here's the main/s of your project (you may need more than one main
│                         
│  
├──  data _loader  
│    └── data_generator.py
│ 
└── utils
     ├── logger.py
     └── llny_other_utils_you_need

```

Main Components
--------------
#### Model

#### Trainer
#### Data Loader
#### Logger


### References
