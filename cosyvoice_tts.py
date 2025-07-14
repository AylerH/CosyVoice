import sys
import os
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from cosyvoice_download import download_cosyvoice_model

def generate_speech(text, prompt_wav_path, output_prefix='output', output_dir='.', model_id='iic/CosyVoice2-0.5B', 
                   model_dir='pretrained_models/CosyVoice2-0.5B', sample_rate=16000, 
                   load_jit=False, load_trt=False, fp16=False, stream=False):
    """
    Generate speech using CosyVoice2 model.
    
    Args:
        text: The text to convert to speech
        prompt_wav_path: Path to the prompt wav file
        output_prefix: Prefix for output wav files
        output_dir: Directory to save output files (default: current directory)
        model_id: The model ID in ModelScope
        model_dir: The local directory to save/load the model
        sample_rate: Sample rate for the prompt wav file
        load_jit: Whether to load JIT model
        load_trt: Whether to load TensorRT model
        fp16: Whether to use FP16 precision
        stream: Whether to stream the output
        
    Returns:
        List of paths to generated wav files
    """
    # Download model if it doesn't exist
    download_cosyvoice_model(model_id, model_dir)
    
    # Load prompt speech
    prompt_speech = load_wav(prompt_wav_path, sample_rate)
    
    # Initialize model
    cosyvoice = CosyVoice2(model_dir=model_dir, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate speech
    output_files = []
    for i, result in enumerate(cosyvoice.inference_cross_lingual(text, prompt_speech, stream=stream)):
        output_file = os.path.join(output_dir, f'{output_prefix}_{i}.wav')
        torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
        output_files.append(output_file)
    
    return output_files

if __name__ == "__main__":
    # Example usage
    model_id = 'iic/CosyVoice2-0.5B'
    model_dir = 'pretrained_models/CosyVoice2-0.5B'
    prompt_wav_path = 'prompt_wav_files/sichuan_dialet/000001.wav'
    text = '他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'
    
    output_files = generate_speech(
        text=text,
        prompt_wav_path=prompt_wav_path,
        output_prefix='fine_grained_control',
        output_dir='output_wavs',
        model_id=model_id,
        model_dir=model_dir
    )
    
    print(f"Generated {len(output_files)} audio files: {output_files}")
    
