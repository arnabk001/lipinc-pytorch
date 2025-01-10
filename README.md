# lipinc-pytorch

This repository contains the PyTorch implementation of the paper "[Exposing Lip-syncing Deepfakes from Mouth Inconsistencies](https://arxiv.org/abs/2401.10113)" by S Datta, S Jia, and S Lyu. The paper introduces LIPINC, a novel approach for detecting lip-syncing deepfakes by identifying temporal inconsistencies in the mouth region of videos.


## Introduction

Lip-syncing deepfakes are videos where a person's lip movements are manipulated to match altered or entirely new audio, making detection challenging. LIPINC (LIP-syncing detection based on mouth INConsistency) addresses this challenge by detecting temporal inconsistencies in the mouth region across video frames.

The original implementation is in Tensorflow, which can be hard to configure for training. The [official repository](https://github.com/skrantidatta/LIPINC) also misses out on training/evaluation code. This repository provides the PyTorch implementation of the LIPINC model, enabling researchers and practitioners to build models for lip-syncing deepfakes effectively.


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/arnabk001/lipinc-pytorch.git
   cd lipinc-pytorch
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure that you have PyTorch installed. You can install it from the [official website](https://pytorch.org/get-started/locally/).


## Dataset Preparation

1. **Download the dataset:**

   LIPINC is trained on the FakeAVCeleb_v1.2 dataset. You can request the download link for the dataset from [here](https://sites.google.com/view/fakeavcelebdash-lab/home?authuser=0)

2. **Extract frames:**

   Use the provided script to extract and preprocess frames from the videos:

   ```bash
   python preprocess_extract_lips.py
   ```

    We preprocess and save the extracted tensors to use later in trainng. We realized this is a much faster way to experiment with different training recipes. Please modify the required path/directories for your training.

## Training


1. **Start training:**

   ```bash
   python train_pipeline.py --config config.py
   ```

   The trained model checkpoints will be saved in the `outputs/` directory.

2. **Evaluate the model:**

   The training script (`train_pipeline.py`) will output the model's performance metrics on the test dataset, including additional evaluation metrics.

## Results

To be updated

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

We thank the authors of the original paper for their foundational work. For more information, refer to the [paper](https://arxiv.org/abs/2401.10113).

---

For any questions or issues, please open an issue on this repository