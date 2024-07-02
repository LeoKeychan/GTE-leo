# GTE-leo

#### Environment

Python version 3.9.12
torch==1.12.0+cu113
numpy==1.21.5
tqdm==4.64.0

#### How to run the GTE

python main.py --data [yelp/gowalla/amazon_book]

#### LightGCL to compare with GTE
#### How to run the LightGCL

python main.py --data yelp
python main.py --data gowalla --lambda2 0
python main.py --data amazon_book --gnn_layer 1 --lambda2 0 --temp 0.1
