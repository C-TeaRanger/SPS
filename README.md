# Seeking Proxy Point via Stable Feature Space for Noisy Correspondence Learning (IJCAI-2025)

[![IJCAI 2025](https://img.shields.io/badge/IJCAI-2025-blue)](https://ijcai.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.7.1-orange)](https://pytorch.org/)

This repository contains the official implementation of the IJCAI-2025 paper: **"Seeking Proxy Point via Stable Feature Space for Noisy Correspondence Learning"**.

**Authors:** Yucheng Xie, Songyue Cai, Tao Tong, Ping Hu, Xiaofeng Zhu.

## 🛠 Requirements

We recommend using Anaconda to manage the environment.
- Python 3.7
- PyTorch ~1.7.1
- numpy
- scikit-learn
- Punkt Sentence Tokenizer (nltk)

```bash
# Example installation commands
conda create -n proxy python=3.7
conda activate proxy
pip install torch==1.7.1 numpy scikit-learn
python -c "import nltk; nltk.download('punkt')"
```

## 📂 Datasets

We follow [NCR (NeurIPS 2021)](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) to obtain image features and vocabularies.

After downloading the data, please organize the folders as follows:

```text
|-- data
    |-- data
    |   |-- cc152k_precomp
    |   |-- coco_precomp
    |   |-- f30k_precomp
    |-- vocab
        |-- cc152k_precomp_vocab.json
        |-- coco_precomp_vocab.json
        |-- f30k_precomp_vocab.json
```

## 🚀 Training

### 1. Flickr30K (Synthetic Noise)
Run the following command to train on Flickr30K. You can modify `--noise_ratio` to `0.2`, `0.4`, `0.6`, or `0.8` to conduct experiments with different noise levels.

```bash
python run.py --data_name=f30k_precomp --noise_ratio=0.2 --num_epochs=40
```

### 2. MS-COCO (Synthetic Noise)
Similar to Flickr30K, you can adjust the `--noise_ratio` (0.2 | 0.4 | 0.6 | 0.8).

```bash
python run.py --data_name=coco_precomp --noise_ratio=0.2 --num_epochs=20
```

### 3. CC152K (Real-world Noise)
Since CC152K is a real-world noisy dataset, no `noise_ratio` argument is needed.

```bash
python run.py --data_name=cc152k_precomp --num_epochs=40
```

## 📊 Evaluating

To evaluate the models, run:

```bash
python evaluation.py
```

**Note:**
* By default, this script evaluates all models located in `./model_ckpt/cream_models/`.
* To evaluate a specific model, please modify the `model_path` variable in `evaluation.py`.

### Pre-trained Models

We provide the pre-trained models used for the paper experiments. You can download them from the following link:

- **Google Drive:** [Download Link](https://drive.google.com/drive/folders/1SuudNBWTCcetkYFAW9Ha84tX8SCF8zZF?usp=sharing)

## 📝 Citation

If you find this work useful or interesting for your research, please consider citing:

```bibtex
@inproceedings{xie2025seeking,
  title={Seeking proxy point via stable feature space for noisy correspondence learning},
  author={Xie, Yucheng and Cai, Songyue and Tong, Tao and Hu, Ping and Zhu, Xiaofeng},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence},
  pages={2072--2080},
  year={2025}
}
```

## 📧 Contact
If you have any questions, please feel free to create an issue on this repository or contact us at **xyemrsnon@gmail.com**.
