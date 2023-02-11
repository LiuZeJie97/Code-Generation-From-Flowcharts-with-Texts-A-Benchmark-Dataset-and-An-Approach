# FC2Code
Code for the paper ["Code Generation From Flowcharts with Texts: A Benchmark Dataset and An Approach"](https://aclanthology.org/2022.findings-emnlp.449/). 
The presentation is [here](https://s3.amazonaws.com/pf-user-files-01/u-59356/uploads/2022-11-17/ps13uor/presentation2124.mp4).

Our model is developed based on [TRANX](https://github.com/pcyin/tranX), please cite their [paper](https://aclanthology.org/D18-2002/) if using our model.

## Dataset Format
The folder "FC2Code" contains the following files: 
### code.txt
&emsp; We obtained 320 code from [__LeetCode__](https://leetcode.com/problemset/all/)

### flowchart.txt
&emsp;We manually drew the flowchart for each code. The first part are the basic information of each node:
        
    [node id] => [node type]: [the text contained within the node]

&emsp;The second part are the associations between nodes:
        
    [node id] ( [yes\no\None] ) -> [node id]

&emsp;You can visit http://flowchart.js.org and translate the flowchart.txt into pictures.

### mapping_relations.txt : 

&emsp; The relationships between the nodes and the code snippets, can only be used in the training phase.

### sequence.txt

&emsp;We sort the nodes according to the code, can only be used in the training phase.


## Two-Stage Code Generation Model

Our model can be divided into 2 stages:
### The First Stage

1. convert flowchart into pseudo_code

    https://github.com/LiuZeJie97/flowchart-to-code

### The Second Stage

1. convert (pseudo_code, code) pairs into pickled files: 

    model\second_stage\datasets\FC2Code\fc2code_dataset.py 

2. train or test on the pickled files: 

    model\second_stage\run_batch.py

