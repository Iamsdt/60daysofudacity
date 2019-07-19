Today I create a helper class.

You may notice, most of the projects like image classification you have same types of code, I don't know how you handle all the same code. Actually, I copy and paste all of those codes. To get rid of this problem I create this helper class with the proper documentation for every function.

List of functions:
- prepare loader() -> for creating date loader
- visualize() -> for visualizing data from data loader
- load_latest_model() -> load last save model
- save_current_model() -> save current model
- save_check_point() -> save model completely
- load_checkpoint() -> load complete model
- freeze_parameters() -> freeze parameters
- unfreeze() -> unfreeze parameters
- train() -> train model
- train_faster_log() -> train model with log print after certain interval in every epoch.
- check_overfitted() -> check train loss and valid loss
- test_per_class() -> test result per class
- test() -> total test result

To load this class you can use this command on kaggle or Colab

```!wget https://raw.githubusercontent.com/Iamsdt/60daysofudacity/master/day22/Helper.py```

Example Notebook

https://www.kaggle.com/iamsdt/kernel6beca45512