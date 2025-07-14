import os
from modelscope import snapshot_download

def download_cosyvoice_model(model_id='iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B'):
    """
    Check if the model directory exists, and download it if not.
    
    Args:
        model_id: The model ID in ModelScope
        local_dir: The local directory to save the model
    """
    if not os.path.exists(local_dir):
        print(f"Model directory {local_dir} does not exist. Downloading...")
        snapshot_download(model_id, local_dir=local_dir)
        print(f"Model downloaded to {local_dir}")
    else:
        print(f"Model directory {local_dir} already exists.")

if __name__ == "__main__":
    download_cosyvoice_model()