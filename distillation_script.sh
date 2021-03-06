python ./transformers-main/examples/research_projects/seq2seq-distillation/distillation.py --teacher t5-small --data_dir data/cmu_hinglish_dog_distillation \
--student_decoder_layers 3 --student_encoder_layers 6 --tokenizer_name t5-small \
--learning_rate=3e-4 --freeze_encoder --no_teacher \
--do_train --train_batch_size 32 \
--do_predict --n_train 100 \
--model_name_or_path t5-small --eval_beams 2 --eval_max_gen_length 142 \
--val_check_interval 0.25 --n_val 1000 \
--output_dir distil_t5_cmu_hinglish --gpus 1 --logger_name wandb \
--task translation --overwrite_output_dir
