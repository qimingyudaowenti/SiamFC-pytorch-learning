The contents in this directory is extracted from [bilylee/**SiamFC-TensorFlow**](https://github.com/bilylee).

Follow below steps like steps in [here](https://github.com/bilylee/SiamFC-TensorFlow#training):

```python
cd SiamFC-pytorch-learning

# DATASET is your ILSVRC2015 dataset path
mkdir -p data
ln -s DATASET data/ILSVRC2015

python data_prepare/preprocess_VID_data.py

python data_prepare/build_VID2015_imdb.py
```

Now, the files in *SiamFC_pytorch_learning/data* should be:

```
ILSVRC2015
ILSVRC2015-VID-Curation
train_imdb.pickle
data/validation_imdb.pickle
```

