export CUDA_VISIBLE_DEVICES=0

#python main.py --anomaly_ratio 0.5 --num_epochs 10 --batch_size 32  --mode all  --dataset SMD  --data_path dataset/SMD --input_c 38 --win_size 50 --k 5 --elayers 4 --dmodel 1024 --dff 1024 --patience 3
#python main.py --mode test --model_save_path result/SMD_2022_08_12_12_01_14 --data_path dataset/SMD
python main.py --mode test --model_save_path result/SMD_2022_08_12_12_03_28 --data_path dataset/SMD --win_size 25