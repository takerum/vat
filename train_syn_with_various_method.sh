#!/bin/sh

python train_syn.py --cost_type=MLE \
        --dataset_filename=syndata_1.pkl --save_filename=MLE_1.pkl
python train_syn.py --cost_type=L2  --lamb=0.0001\
        --dataset_filename=syndata_1.pkl --save_filename=L2_1.pkl
python train_syn.py --cost_type=dropout --dropout_rate=0.4\
        --dataset_filename=syndata_1.pkl --save_filename=dropout_1.pkl
python train_syn.py --cost_type=VAT --epsilon=1.8\
        --dataset_filename=syndata_1.pkl --save_filename=RP_1.pkl
python train_syn.py --cost_type=AT_L2 --num_power_iter=0 --epsilon=0.2\
        --dataset_filename=syndata_1.pkl --save_filename=AT_1.pkl
python train_syn.py --cost_type=VAT --num_power_iter=1 --epsilon=0.6\
        --dataset_filename=syndata_1.pkl --save_filename=VAT_1.pkl

python train_syn.py --cost_type=MLE \
        --dataset_filename=syndata_2.pkl --save_filename=MLE_2.pkl
python train_syn.py --cost_type=L2  --lamb=0.0001\
        --dataset_filename=syndata_2.pkl --save_filename=L2_2.pkl
python train_syn.py --cost_type=dropout --dropout_rate=0.75\
        --dataset_filename=syndata_2.pkl --save_filename=dropout_2.pkl
python train_syn.py --cost_type=VAT --epsilon=2.8\
        --dataset_filename=syndata_2.pkl --save_filename=RP_2.pkl
python train_syn.py --cost_type=AT_L2 --num_power_iter=0 --epsilon=0.65\
        --dataset_filename=syndata_2.pkl --save_filename=AT_2.pkl
python train_syn.py --cost_type=VAT --num_power_iter=1 --epsilon=0.4\
        --dataset_filename=syndata_2.pkl --save_filename=VAT_2.pkl