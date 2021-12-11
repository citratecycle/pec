### python environment:
I've configured the conda environment under my user directory "/home/louchenfei", so directly typing `python` should envoke the correct python package.  
to verify that the correct environment is used, there should be '(/home/louchenfei/anaconda3)' appearing in front of the bash prompt. And by typing `which python` in terminal, it should return `/home/louchenfei/anaconda3/bin/python`

### File Structure

- nikolaos/*
- global_param.py
- inference_new.py
- main_new.py
- model.py
- train_new.py
- utils_new.py

when it is finished, we can directly call  
`python main_new.py --model_name MODEL_NAME --pretrained_file PRETRAINED_FILE_NAME --optimizer OPTIM_NAME --train_mode TRAIN_MODE --evaluate_mode EVALUATE_MODE --task TASK --device DEVICE --save SAVE`  
Their meanings are specified as below. Note that not all arguments are meaningful at the same time, for example, when the `TASK` is 'train', then `EVALUATE_MODE` does not have any meaning.
- `MODEL_NAME`: the name of the model, either 'cifar', 'vgg' or 'vgg19'. 'cifar' is a smaller model with only 4 conv layers, and 'vgg' is a deeper model with 6 conv layers, and 'vgg19' is an extremely deep convolutional network with 16 layers. It achieves the best accuracy, and we hope that early-exit could show its advantage on this model.
- `PRETRAINED_FILE_NAME`: the file that stores the pretrained model. They are all stored under the directory 'models_new'
    - '[model_name]_normal_default.pt' is the model for vanilla training
    - '[model_name]_original_default.pt' is the model with early-exit layers implemented but only original layers pre-trained.
    - '[model_name]_exits_default.pt' means the model with early-exit layers implemented and well trained.
- `OPTIM_NAME`: the name of the optimizer. Either 'sgd' or 'adam'. Usually by using 'adam' the training can be much more efficient.
- `TRAIN_MODE`: the mode for training. 
    - 'normal': train the model according to the most vanilla setting ---- without any early-exit schemes
    - 'original': train the model with early-exit layers implemented but only train the original layers and ignore the early-exit layers. This mode stores a pre-trained file that is necessary for the following 'exits' mode
    - 'exits': train the early-exit layers of the model, assuming that the original layers have already been trained.
- `EVALUATE_MODE`: the mode for evaluation.
    - 'normal': evaluate the normal mode without any implementation of early exit layers
    - 'exits': evaluate the model with early exit layers and adaptive inference methods
- `TASK`: the task type, either 'train' or 'evaluate'. 'train' means to train and store a model based on training dataset, while 'evaluate' means to inference a model based on testing dataset
- `DEVICE`: the device where the training and inference is executed. Either 'cpu' or 'cuda'
- `SAVE`: if non-zero, then save the model, otherwise does not save the model

Finally, we can change the setting of hyperparameters by making corresponding changes in `global_param.py`.


### the process for training:
1. configure the hyperparameters in `global_param.py`. E.g. for normal training mode of vgg network, configure the parameters in the namespace `vgg_normal_train_hyper`
2. type the command. E.g. for training vgg using 'normal' mode on gpu, type:  
`python main_new.py --model_name vgg --optimizer adam --train_mode normal --task train --device cuda --pretrained_file models_new/cifar_exits_default.pt > experimental_results_new/vgg_train_normal.txt`