### old files: 

can run but the codes are a little bit messy  
just type `python main.py`, but if you want to change some settings (such as switch to a different model or do inference instead of training, you must go deep into the codes to make the changes, which is extremely prone to bugs)

- nikolaos/*
- inference.py
- main.py
- train.py
- utils.py


### new files:

still in progress but will be definitely more elegant

- json_files/*
- nikolaos/*
- global_param.py
- inference_new.py
- main_new.py
- model.py
- train_new.py
- utils_new.py

when it is finished, we can directly call  
`python main_new.py --model_name MODEL_NAME --pretrained_file FILE_NAME --early_exit_json JSON_NAME --optimizer OPTIM_NAME --train_mode TRAIN_MODE --task TASK --device DEVICE`  
and we can change the setting of hyperparameters by making corresponding changes in `global_param.py`; and if we want to change the early-exit structures of the network, we can directly change it by changing the json values in `json_files/*`
