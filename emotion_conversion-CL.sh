
                
python train.py  --num_f 128 \
                --train_A_dir './data/neutral/'  \
                --train_B_dir './data/target/' \
                --model_dir './model/' --model_name 'model.ckpt' --random_seed 0  \
                --validation_A_dir './data/test_neutral/' --validation_B_dir './data/test_target/' \
                --output_dir './validated/' --tensorboard_log_dir './log/' 
                
                
python train.py  --num_f 256 \
                --train_A_dir './data/neutral/'  \
                --train_B_dir './data/target/' \
                --model_dir './model/' --model_name 'model.ckpt' --random_seed 0  \
                --validation_A_dir './data/test_neutral/' --validation_B_dir './data/test_target/' \
                --output_dir './validated/' --tensorboard_log_dir './log/'
                
python train.py  --num_f 380 \
                --train_A_dir './data/neutral/'  \
                --train_B_dir './data/target/' \
                --model_dir './model/' --model_name 'model.ckpt' --random_seed 0  \
                --validation_A_dir './data/test_neutral/' --validation_B_dir './data/test_target/' \
                --output_dir './validated/' --tensorboard_log_dir './log/' 
                
                
python train_f0.py
                --train_A_dir './data/neutral/'  \
                --train_B_dir './data/target/' \
                --model_dir './model/' --model_name 'model.ckpt' --random_seed 0  \
                --validation_A_dir './data/test_neutral/' --validation_B_dir './data/test_target/' \
                --output_dir './validated/' --tensorboard_log_dir './log/' 
                

                
# ... (add more if you want)
                
     