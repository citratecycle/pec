# train the model cifar and do the inference after that
# model_name=cifar
model_name=vgg19
suffix=update_1


python main_new.py --model_name ${model_name} --optimizer adam --train_mode normal --task train --device cuda --trained_file_suffix ${suffix} > experimental_results_new/${model_name}_train_normal_${suffix}.txt

python main_new.py --model_name ${model_name} --optimizer adam --train_mode original --task train --device cuda --trained_file_suffix ${suffix} > experimental_results_new/${model_name}_train_original_${suffix}.txt

python main_new.py --model_name ${model_name} --optimizer adam --train_mode exits --task train --device cuda --pretrained_file models_new/${model_name}_train_original_${suffix}.pt --trained_file_suffix ${suffix} > experimental_results_new/${model_name}_train_exits_${suffix}.txt

python main_new.py --model_name ${model_name} --optimizer adam --train_mode exits --task evaluate --device cuda --pretrained_file models_new/${model_name}_train_exits_${suffix}.pt --trained_file_suffix ${suffix} --beta 1.5 > experimental_results_new/${model_name}_inference_exits_${suffix}.txt