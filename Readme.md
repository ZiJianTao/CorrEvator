# CorrEvator

## CorrEvator: Evaluating Patch Correctness in Automated Program Repair

Python library dependencies are listed in requirements.txt
you can choose to run `pip install -r requirements.txt` to quickly install the dependencies.

---
Dataset:

Source data comes from [Quatrain]: Reference paper: Tian, et al. "Is this Change the Answer to that Problem?: Correlating Descriptions of Bug and Code Changes for Evaluating Patch Correctness" 
Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering (link:[https://api.semanticscholar.org/CorpusID:251403079])
<including: 7544 incorrect BugReport-Patch pairs and 1591 correct BugReport-Patch pairs>

The data were then pre-processed and divided into 10 groups, resulting in 10-group cross-validation. (The data processing tools we use are inside 'pre-process' )

---

#### **IN order to evaluate, the following 5 steps are required**

+ getGraph_train_data.py
    
    `python getGraph_train_data.py N`
    
    It will take `xxx_pull_info_X.txt` and `xxx_pull_info_y.txt` files from `DATA/GROUPi/train-dataset` and `DATA/GROUPi/test-dataset` directories to build the training and testing data graph. 
    N1 indicates the size of the various parameters used to build the graph, which you can set and modify in the code according to your own experimental needs.
    Then three graph files will be generated in the same directory, as follows:
   
    `vsBATS_N.train_graphs`, `vsBATS_N.val_graphs`, `vsBATS_N.test_graphs`


+ gmn/train.py  

    `python gmn/train.py filename`
    
    It will take `filename.train_graphs` and `filename.val_graphs` in the `DATA/GROUPi/train-dataset` directory to train the model.

     
+ gmn/getResult.py

     `python gmn/getResult.py filename modelname repoid`
    
    It will take `filename.test_graphs` in each repository directory to test the model in the `modelname` directory.


+ getresult.bat or run_script.sh

     `./run_script.sh` in Linux or `./run_script.sh` in Windows
   
    Here are two simple batch codes to get the results of all the checkpoints to select a better model.
    

+ evaluation.py

     `python evaluation.py`
  
    It will take `data_file_BATS0.8.pkl` to evaluate the final result.


---

The overall process is roughly as follows: 

1. Set various parameters for constructing the feature graph in 'getGraph_train_data.py' according to the requirements.
2. Use `python getGraph_train_data.py N` to build graphs for ten sets of data.
3. Use `python gmn/train.py filename` to train the model.
4. Use `gmn/getResult.py` to directly obtain results, or use `getresult.bat` or `run_script.sh` to find the best checkpoint model.
5. Modify the code in lines 389-399 of 'getResult.py' to test the best model on each set of data.
6. Finally, use 'evaluation.py' to obtain the overall results.

---

nlp.py: Natural Language Processing model for calculating the text similarity.
```
m = Model(texts)
text_sim = query_sim_tfidf(tokens1, tokens2)
``` 

comp.py: Calculate the similarity for feature extraction.
``` 
# Set up the params of compare (different metrics).
# Check for init NLP model.
feature_vector = get_pr_sim_vector(pull1, pull2)
```



