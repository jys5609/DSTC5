# Dialog State Tracking Challenge 5 (DSTC5)

## Neural Dialog State Tracker for Large Ontologies by Attention Mechanism

## Getting started

**STEP 1: Generate data for training** </br>

Generate the training data.

    $ python data_generator.py  

After then, there will be created 5 pkl files.

* dstc5_general.pkl: dumping the slot, slot value vectors </br>
* dstc5_train.pkl: dumping the training data </br>
* dstc5_dev.pkl: dumping the validation data </br>
* dstc5_dev_acc.pkl: dumping the accumulated validation data (accumulate the previous utterance) </br>
* dstc5_test_acc.pkl: dumping the accumulated test data (accumulate the previous utterance) </br>

**STEP 2: Training** </br>

It takes 7 arguments. Example of the implementing code is like below. 

    $ python train.py -l 100 -lr 0.005 -e 100

*  -l: the number of lstm units (default = 100) </br>
* -lr: learning rate (default = 0.005) </br>
* -dr1: first dropout parameter (default = 0) </br>
* -dr2: second dropout parameter (default = 0) </br>
*  -e: the number of epoch (default = 300) </br>
*  -t: type of dstc (4 or 5, default = 5) </br>
*  -c: criteria of finding threshold (accuracy or fscore, default = accuracy) </br>

After then, there will be created weight file with named 'dstc5_lstm#l_lr#lr_dr#dr1_#dr2.h5' (ex. dstc5_lstm100_lr005_dr0_0.h5)

**STEP 3: Predict with finding threshold** </br>

It takes 7 arguments. Example of the implementing code is like below. 

    $ python predict.py -l 100 -lr 0.005 -e 100 -th

These arguments are same with STEP 2.

*   -l: the number of lstm units (default = 100) </br>
*  -lr: learning rate (default = 0.005) </br>
* -dr1: first dropout parameter (default = 0) </br>
* -dr2: second dropout parameter (default = 0) </br>
*   -e: the number of epoch (default = 300) </br>
*   -t: type of dstc (4 or 5, default = 5) </br>
*   -c: criteria of finding threshold (accuracy or fscore, default = accuracy) </br>

One more argumets here.

*   -th: to decide the threshold and make a file for threshold (default is no -th) </br>
  (You have to add -th when you implement predict.py first time.)

After then, there will be created 2 json files like below.

* dev_dstc5_lstm100_lr005_dr0_0_accuracy.json </br>
* test_dstc5_lstm100_lr005_dr0_0_accuracy.json </br>

**STEP 4: Make a result** </br>

For validation result,

    $ bash dev_run.sh dev_dstc5_lstm100_lr005_dr0_0_accuracy.json

For test result,

    $ bash test_run.sh test_dstc5_lstm100_lr005_dr0_0_accuracy.json

Then, you can see the result and there will be created the result files.

* dev_dstc5_lstm100_lr005_dr0_0_accuracy.score.csv </br>
* test_dstc5_lstm100_lr005_dr0_0_accuracy.score.csv
