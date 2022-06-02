python ./transformers-main/examples/research_projects/seq2seq-distillation/make_student.py google/mt5-small mt5_small_6_3 6 3

python ./transformers-main/examples/research_projects/seq2seq-distillation/finetune.py \
--model_name_or_path mt5_small_6_3 --data_dir data/cmu_hinglish_dog \
--learning_rate=3e-4 --freeze_encoder \
--do_train --train_batch_size 32 \
--do_predict --n_train 100 \
--model_name_or_path mt5_small_6_3 --eval_beams 2 --eval_max_gen_length 142 \
--val_check_interval 0.25 --n_val 1000 \
--output_dir distil_student_mt5_cmu_hinglish --gpus 1 --logger_name wandb \
--task translation --overwrite_output_dir

# Maybe include --freeze_embeds again, resulted in an error for the mT5 model.
