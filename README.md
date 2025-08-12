# Seeking Proxy Point via Stable Feature Space for Noisy Correspondence Learning (IJCAI-2025)

This is the official implementation for Seeking Proxy Point via Stable Feature Space for Noisy Correspondence Learning (IJCAI-2025) by Yucheng Xie, Songyue Cai, Tao Tong, Ping Hu, Xiaofeng Zhu


# Requirements
- Python 3.7
- PyTorch ~1.7.1
- numpy
- scikit-learn
- Punkt Sentence Tokenizer

# Datasets

We follow NCR [https://github.com/XLearning-SCU/2021-NeurIPS-NCR/](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) to obtain image features and vocabularies. After downloading the data, you need to place the folders as follows:

    |--data
       |--data
          |--cc152k_precomp
          |--coco_precomp
          |--f30k_precomp
       |--vocab
          |--cc152k_precomp_vocab.json
          |--coco_precomp_vocab.json
          |--f30k_precomp_vocab.json
          
# Training
python run.py --data_name=f30k_precomp --noise_ratio=0.2 --num_epochs=40

You can change --noise_ratio=0.2 to 0.4 | 0.6 | 0.8 to conduct more experiments on Flickr30K.

python run.py --data_name=cc152k_precomp --num_epochs=40

As CC152K is a real-world dataset, there is no need to set --noise_ratio.

python run.py --data_name=coco_precomp --noise_ratio=0.2 --num_epochs=20

You can change --noise_ratio=0.2 to 0.4 | 0.6 | 0.8 to conduct more experiments on MS-COCO.

# Evaluating
python evaluation.py

This will evaluate all the models in the model_path="./model_ckpt/cream_models/". If you need to evaluate one model, just change model_path in evaluation.py.

# Citation
If you find this work useful or interesting, please consider citing it.

The reference link is still in preparation.
