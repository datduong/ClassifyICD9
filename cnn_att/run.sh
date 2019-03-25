
## complete data, init label 
CUDA_VISIBLE_DEVICES=4 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder m3icd8996FixW2v --num_of_filter 50 --label_encoded_dim 50 --init_label_emb --w2v_emb 100 --batch_size 12 --not_train_w2v_emb --top_k 8 --model_load /local/datdb/MIMIC3database/format10Jan2019/m3icd8996FixW2v/current_best.pytorch --epoch 0 --do_test > m3icd8996FixW2v_part3.txt 


CUDA_VISIBLE_DEVICES=7 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder m3icd50FixW2v --num_of_filter 50 --label_encoded_dim 50 --init_label_emb --w2v_emb 100 --batch_size 16 --not_train_w2v_emb --add_name _50icd > m3icd50FixW2v.txt 


CUDA_VISIBLE_DEVICES=1 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder m3icdManualSmall6FixW2v --num_of_filter 50 --label_encoded_dim 50 --init_label_emb --w2v_emb 100 --batch_size 16 --not_train_w2v_emb --add_name _manual_small_6icd --top_k 2 --epoch 200 > m3icdManualSmall6FixW2v.txt 

CUDA_VISIBLE_DEVICES=1 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder m3icdManualSmall2FixW2v --num_of_filter 50 --label_encoded_dim 50 --init_label_emb --w2v_emb 100 --batch_size 16 --not_train_w2v_emb --add_name _manual_small_2icd --top_k 1 --epoch 100 > /local/datdb/ICD9multitask/cnn_att/m3icdManualSmall2FixW2v.txt 


## 50 icd add full ancestors 
CUDA_VISIBLE_DEVICES=7 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder m3icd50AddAnc --num_of_filter 50 --label_encoded_dim 50 --init_label_emb --w2v_emb 100 --batch_size 16 --not_train_w2v_emb --add_name _full_tree_50icd --top_k 5 --epoch 0 --do_test > m3icd50FixW2vAddAnc2.txt 

## continue 
CUDA_VISIBLE_DEVICES=7 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder m3icd50 --num_of_filter 50 --label_encoded_dim 50 --init_label_emb --w2v_emb 100 --batch_size 32 --not_train_w2v_emb --add_name _50icd --model_load /local/datdb/MIMIC3database/format10Jan2019/m3icd50/current_best.pytorch --epoch 100 > m3icd50FixW2v_part2.txt 


CUDA_VISIBLE_DEVICES=7 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder m3icd5 --num_of_filter 50 --label_encoded_dim 50 --init_label_emb --w2v_emb 100 --batch_size 32 --not_train_w2v_emb --add_name _5icd > m3icd5FixW2v.txt 

## complete data, use label average 
CUDA_VISIBLE_DEVICES=7 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder m3icd8996icdAve --num_of_filter 100 --label_encoded_dim 100 --w2v_emb 100 --batch_size 12 > m3icd8996icdAve.txt 

## complete data, use label average + gcnn
CUDA_VISIBLE_DEVICES=6 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder m3icd8996icdAveGcnnLr0.001 --num_of_filter 100 --label_encoded_dim 100 --w2v_emb 100 --batch_size 12 --do_gcnn --gcnn_dim 100 --lr .001 > m3icd8996icdAveGcnnLr0.001.txt 



## run gcnn only 
CUDA_VISIBLE_DEVICES=5 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder m3icd8996icdGcnnOnly300FixW2v --num_of_filter 300 --label_encoded_dim 300 --w2v_emb 300 --batch_size 12 --do_gcnn --do_gcnn_only --gcnn_dim 300 --lr .001 --epoch 1000 --not_train_w2v_emb > m3icd8996icdGcnnOnly300FixW2v.txt 




## use 50 icd 

CUDA_VISIBLE_DEVICES=6 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/main.py --result_folder pilot --num_of_filter 50 --label_encoded_dim 50 --init_label_emb --w2v_emb 100 --batch_size 8 --add_name _50icd > pilot1.txt 

CUDA_VISIBLE_DEVICES=6 /home/datdb/anaconda3/bin/python3 -u /local/datdb/ICD9multitask/cnn_att/run_cnn_att.py --result_folder pilot --num_of_filter 50 --label_encoded_dim 50 --init_label_emb --w2v_emb 100 --batch_size 8 --add_name _50icd > pilot1.2.txt 

