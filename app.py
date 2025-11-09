import ollama
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
# --- Silence Warnings & Logs BEFORE importing chatty modules if possible ---
import warnings
import logging
import os
import re # Added for sentence splitting

# Suppress specific warnings (like the torch.cuda.amp.autocast one)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress lower-level logging
logging.getLogger("whisper").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow/generic backend noise if any

from bark import SAMPLE_RATE, generate_audio, preload_models
import time
import argparse
import google.generativeai as genai
import sys

# --- Configuration ---
AUDIO_FILE = "input.wav"
SAMPLE_RATE_REC = 16000  # Sample rate for recording (Whisper prefers 16kHz)
RECORD_SECONDS = 5        # Duration of audio to record
WHISPER_MODEL = "base"    # Model size ("tiny", "base", "small", "medium", "large")
OLLAMA_MODEL = "llama3.1:latest"
GEMINI_MODEL = "gemini-2.5-flash" # Fast and efficient for chat
BARK_MODEL_SIZE = "small" # Use "small" for lower VRAM, or "large" for better quality

# --- 1. Audio Input (STT) ---
def record_audio(filename, duration, sr):
    print(f"\nRecording for {duration} seconds...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(filename, sr, recording)  # Save as WAV file
    print(f"Recording complete. Saved to {filename}")

def transcribe_audio(filename):
    print("Transcribing audio...")
    # Load model here if not preloaded, but we preload in main() usually.
    model = whisper.load_model(WHISPER_MODEL)
    # verbose=False suppresses the Whisper progress bar
    result = model.transcribe(filename, verbose=False)
    return result["text"]

# --- 2. LLM Response Generation ---
def get_ollama_response(prompt_text):
    print(f"Sending prompt to Ollama ({OLLAMA_MODEL})...")
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{'role': 'user', 'content': prompt_text}]
    )
    return response['message']['content']

def get_gemini_response(prompt_text):
    print(f"Sending prompt to Gemini ({GEMINI_MODEL})...")
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"

# --- 3. Audio Output (TTS) ---
def generate_and_play_audio(text_prompt):
    if not text_prompt:
        return
        
    print("Synthesizing speech with Bark...")
    
    # 1. Clean up text (remove newlines that might confuse splitter)
    text_prompt = text_prompt.replace("\n", " ").strip()

    # 2. Split text into sentences using regex.
    # Looks for '.', '!', '?' followed by a space or end of string.
    sentences = re.split(r'(?<=[.!?])\s+', text_prompt)

    # 3. Generate and play each sentence sequentially
    for sentence in sentences:
        if len(sentence.strip()) < 2:
             continue # Skip empty or tiny weird chunks

        # print(f"Playing chunk: '{sentence[:20]}...'") # Optional debug
        
        # silence=True tries to suppress internal Bark progress bars
        audio_array = generate_audio(
            sentence,
            history_prompt="v2/en_speaker_6",
            text_temp=0.7,
            silent=True 
        )

        sd.play(audio_array, samplerate=SAMPLE_RATE)
        sd.wait() # Wait for this sentence to finish before starting the next one

    print("Audio playback complete.")

# --- Main Orchestration Loop ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Voice Assistant with Ollama or Gemini")
    parser.add_argument(
        "--llm", 
        choices=["ollama", "gemini"], 
        default="ollama", 
        help="Choose the LLM provider to use (default: ollama)"
    )
    args = parser.parse_args()

    # --- Gemini Setup Check ---
    if args.llm == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY environment variable not found.")
            print("Please set it using: export GEMINI_API_KEY='your_key'")
            sys.exit(1)
        genai.configure(api_key=api_key)

    # --- Pre-load models ---
    print("Pre-loading Whisper and Bark models... (this may take a moment)")
    
    whisper.load_model(WHISPER_MODEL)
    preload_models(
        text_use_gpu=True,
        text_use_small=BARK_MODEL_SIZE == "small",
        coarse_use_gpu=True,
        coarse_use_small=BARK_MODEL_SIZE == "small",
        fine_use_gpu=True,
        fine_use_small=BARK_MODEL_SIZE == "small",
        codec_use_gpu=True,
        force_reload=False
    )
    

    print(f"Models loaded. Starting agent using [{args.llm.upper()}]...")
    
    try:
        while True:
            # 1. Listen for user input
            input(f"\nPress Enter to start recording for {RECORD_SECONDS} seconds (or Ctrl+C to quit)...")
            record_audio(AUDIO_FILE, RECORD_SECONDS, SAMPLE_RATE_REC)
            
            # 2. Transcribe user input
            start_time = time.time()
            user_text = transcribe_audio(AUDIO_FILE)
            print(f" [Timing] Transcription: {time.time() - start_time:.2f}s")

            # Clean up transcription empty checks
            if not user_text.strip():
                print("No speech detected, trying again.")
                continue

            print(f"\nUser: {user_text}")

            if "exit" in user_text.lower() or "quit" in user_text.lower():
                print("Exiting.")
                break

            # 3. Get LLM response based on selected provider
            start_time = time.time()
            llm_response_text = ""
            if args.llm == "ollama":
                llm_response_text = get_ollama_response(user_text)
            elif args.llm == "gemini":
                llm_response_text = get_gemini_response(user_text)
            print(f" [Timing] LLM Response ({args.llm}): {time.time() - start_time:.2f}s")
            
            print(f"\nAgent ({args.llm}): {llm_response_text}\n")

            # 4. Synthesize and play response
            start_time = time.time()
            generate_and_play_audio(llm_response_text)
            print(f" [Timing] TTS Generation & Playback: {time.time() - start_time:.2f}s")

    except KeyboardInterrupt:
        print("\nExiting agent.")
    except Exception as e:
        # Print full error trace only if it's a real unexpected crash
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
