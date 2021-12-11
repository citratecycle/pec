model_name=cifar
save=0
suffix=update_1

sudo python3 main_new.py --model_name ${model_name} --optimizer adam --train_mode exits --task evaluate --device cuda --pretrained_file models_new/${model_name}_train_normal_${suffix}.pt --trained_file_suffix ${suffix} --beta 1 --save ${save} --evaluate_mode normal > experimental_results_new_1/${model_name}_inference_exits_normal.txt