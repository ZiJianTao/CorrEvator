# CorrEvator

## CorrEvator: Evaluating Patch Correctness in Automated Program Repair

Python library dependencies are listed in requirements.txt
you can choose to run `pip install -r requirements.txt` to quickly install the dependencies.

---
Dataset:

Source data comes from [Quatrain]: Reference paper: Tian, et al. "Is this Change the Answer to that Problem?: Correlating Descriptions of Bug and Code Changes for Evaluating Patch Correctness" 
Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering (link:[https://api.semanticscholar.org/CorpusID:251403079])
<including: 7544 incorrect BugReport-Patch pairs and 1591 correct BugReport-Patch pairs>

The data were then pre-processed and divided into 10 groups, resulting in 10-group cross-validation.


---

#### **IN order to evaluate, the following 5 steps are required**

+ getGraph_train_data.py.py
    
    `python getGraph_train_data.py N`
    
    It will take `xxx_pull_info_X.txt` and `xxx_pull_info_y.txt` files from `DATA/GROUPi/train-dataset` directory to build the training data graph. 
    The N1 indicates the sliding window size used to build the graph, 
    then two graph files will be generated in the same directory, as follows:
   
    `remove_XXX_N.train_graphs`, `remove_XXX_N.val_graphs`
    
    >Note: XXX can be ` `, `title`, `body`. Where N represents the sliding window size.

+ getGraph_repo_test.py
   
    `python getGraph_repo_test_data.py N1 N2`
    
    It will take `xxx_pull_info_X.txt` and `xxx_pull_info_y.txt` files from each repository directory to build the testing data graph. 
    The N1 indicates the sliding window size used to build the graph. 
    The N2 indicates the graph building of which repository. 
    Then `remove_XXX_N.test_graphs` will be generated in the each repository directory.

+ gmn/train.py  

    `python gmn/train.py filename`
    
    It will take `filename.train_graphs` and `filename.train_val_graphs` in the `data/train-dataset/` directory to train the model.
     
+ gmn/getResult.py

     `python gmn/getResult.py filename modelname repoid`
    
    It will take `filename.test_graphs` in each repository directory to test the model in the `modelname` directory. 
    Repoid indicates how many testing data sets will be processed.

+ getresult.bat or run_script.sh

     `./run_script.sh` in Linux or `./run_script.sh` in Windows to get the results of all the checkpoints to select a better model.

+ gmn/getResult.py

     `python gmn/getResult.py filename modelname repoid`
    
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



