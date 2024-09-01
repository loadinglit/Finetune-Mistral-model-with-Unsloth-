## Fine-Tuning Mistral-7B Model with Alpaca Dataset

This repository contains the Jupyter Notebook used for fine-tuning the Mistral-7B model on the Alpaca dataset. The fine-tuned model, optimized for various NLP tasks, is available on Hugging Face.

### Model Details

- **Model Name:** Mistral-7B (4-bit Quantized)
- **Hugging Face Model Page:** [Mistral-7B (4-bit)](https://huggingface.co/unsloth/mistral-7b-v0.3-bnb-4bit)
- **Dataset:** Alpaca Cleaned [Link](https://huggingface.co/datasets/yahma/alpaca-cleaned)

### Notebook Highlights

- **Data Preparation:** Load and prepare the Alpaca dataset.
- **Model Training:** Fine-tune the Mistral-7B model with 4-bit quantization.
- **Evaluation:** Assess the model's performance.
- **Inference:** Generate predictions using the fine-tuned model.

### Installation Instructions

To set up the environment and run the notebook, follow these steps:

1. **Install Conda:**
   Ensure you have Conda installed. If not, download and install it from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create and Activate a New Conda Environment:**
   ```bash
   conda create -n mistral-finetune python=3.10
   conda activate mistral-finetune
   ```

3. **Install PyTorch and CUDA Dependencies:**
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. **Install Additional Packages:**
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
   ```

5. **Verify PyTorch and CUDA Installation:**
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

   For Pytorch 2.3.0 and newer RTX 30xx GPUs or higher, use:
   ```bash
   pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
   ```

6. **Install TensorFlow:**
   For GPU users:
   ```bash
   pip install tensorflow[and-cuda]
   ```

   For CPU users:
   ```bash
   pip install tensorflow
   ```

7. **Verify TensorFlow Installation:**
   ```bash
   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

   If a list of GPU devices is returned, TensorFlow is installed successfully.

### Running the Notebook

1. **Download the Notebook:**
   Clone or download this repository to your local machine.

2. **Open the Jupyter Notebook:**
   Launch Jupyter Notebook from the environment:
   ```bash
   jupyter notebook
   ```

3. **Execute the Cells:**
   Follow the instructions in the notebook to reproduce the fine-tuning process.

Feel free to explore the notebook, run experiments, and provide any feedback or contributions.

For questions or issues, please open an issue on this repository or contact me directly.

Happy experimenting! 
