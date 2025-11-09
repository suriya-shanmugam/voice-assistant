import ollama
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.api import semantic_to_waveform
import time

# --- Configuration ---
AUDIO_FILE = "input.wav"
SAMPLE_RATE_REC = 16000  # Sample rate for recording (Whisper prefers 16kHz)
RECORD_SECONDS = 5        # Duration of audio to record
WHISPER_MODEL = "base"    # Model size ("tiny", "base", "small", "medium", "large")
LLAMA_MODEL = "llama3.1:latest"
BARK_MODEL_SIZE = "small" # Use "small" for lower VRAM, or "large" for better quality

# --- 1. Audio Input (STT) ---
def record_audio(filename, duration, sr):
    print("Recording...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(filename, sr, recording)  # Save as WAV file
    print(f"Recording complete. Saved to {filename}")

def transcribe_audio(filename):
    print("Transcribing audio...")
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(filename)
    return result["text"]

# --- 2. LLM Response Generation ---
def get_llm_response(prompt_text):
    print("Sending prompt to Llama 3.1...")
    response = ollama.chat(
        model=LLAMA_MODEL,
        messages=[{'role': 'user', 'content': prompt_text}]
    )
    return response['message']['content']

# --- 3. Audio Output (TTS) ---
def generate_and_play_audio(text_prompt):
    print("Synthesizing speech with Bark...")
    
    # generate_audio is the all-in-one function.
    # It will automatically use the "small" models you preloaded in main().
    audio_array = generate_audio(
        text_prompt,
        history_prompt="v2/en_speaker_6", # A sample voice
        text_temp=0.7
    )

    print("Playing audio...")
    sd.play(audio_array, samplerate=SAMPLE_RATE)
    sd.wait()
    print("Audio playback complete.")
# --- Main Orchestration Loop ---
def main():
    # Pre-load models to speed up first inference
    print("Pre-loading models...")
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
    print("All models loaded. Ready.")
    
    try:
        while True:
            # 1. Listen for user input
            input("Press Enter to start recording for {} seconds...".format(RECORD_SECONDS))
            record_audio(AUDIO_FILE, RECORD_SECONDS, SAMPLE_RATE_REC)
            
            # 2. Transcribe user input
            user_text = transcribe_audio(AUDIO_FILE)
            print(f"User: {user_text}")

            if "exit" in user_text.lower():
                print("Exiting.")
                break

            # 3. Get LLM response
            llm_response_text = get_llm_response(user_text)
            print(f"Agent: {llm_response_text}")

            # 4. Synthesize and play response
            # generate_and_play_audio(llm_response_text)

    except KeyboardInterrupt:
        print("\nExiting agent.")

if __name__ == "__main__":
    main()
