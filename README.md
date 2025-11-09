# AI Voice Assistant (Ollama & Gemini + Bark TTS)

This project is a local voice assistant that uses **Whisper** for speech-to-text (STT), **Ollama (Llama 3.1)** or **Google Gemini** for intelligence, and **Suno Bark** for text-to-speech (TTS) output.

## Prerequisites (System)

Before installing the Python packages, you need a few system-level dependencies for audio recording and processing.

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3-venv portaudio19-dev ffmpeg
```

  * `portaudio19-dev`: Required by `sounddevice` for microphone access.
  * `ffmpeg`: Required by `openai-whisper` for audio handling.

### MacOS

```bash
brew install portaudio ffmpeg
```

-----

## Installation

### 1\. Set up a Python Virtual Environment

It is highly recommended to use a virtual environment to keep dependencies isolated.

```bash
# Create the virtual environment named 'env'
python3 -m venv env

# Activate it
source env/bin/activate  # On Linux/macOS
# .\env\Scripts\activate # On Windows
```

### 2\. Install Python Dependencies

Install the required libraries within your active virtual environment.

```bash
pip install ollama openai-whisper sounddevice scipy numpy suno-bark google-generativeai
```

> **Note:** Installing `suno-bark` will also install PyTorch, which might be a large download.

-----

## Critical Manual Fix for Bark

Due to recent security changes in newer versions of PyTorch, the current `suno-bark` package may fail to load models with a "weights\_only" error. You must manually patch one file in the library.

1.  **Locate the file:** Find `generation.py` inside your virtual environment's site-packages.
      * Path usually looks like: `env/lib/python3.x/site-packages/bark/generation.py`
2.  **Open the file** in a text editor.
3.  **Go to line \~212**. You should see this line:
    ```python
    checkpoint = torch.load(ckpt_path, map_location=device)
    ```
4.  **Change it** to the following (add `weights_only=False`):
    ```python
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    ```
5.  Save and close the file.

-----

## Model Setup

### Option A: Setting up Ollama (Local)

If you want to use the local Llama 3.1 model, you need Ollama installed and running.

1.  **Install Ollama:** Follow instructions at [ollama.com](https://ollama.com).
2.  **Start the Server:**
    Open a separate terminal and run:
    ```bash
    ollama serve
    ```
3.  **Pull the Model:**
    In another terminal, download the model used in the script:
    ```bash
    ollama pull llama3.1:latest
    ```

### Option B: Setting up Gemini (Cloud)

If you want to use Google's Gemini model.

1.  Get an API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Export it as an environment variable:
    ```bash
    export GEMINI_API_KEY="your_actual_api_key_here"
    ```

-----

## Usage

Ensure your virtual environment is active (`source env/bin/activate`).

### Run with Ollama (Default)

Make sure `ollama serve` is running in another window.

```bash
python voice_agent.py
```

*Or explicitly:*

```bash
python voice_agent.py --llm ollama
```

### Run with Gemini

Make sure your `GEMINI_API_KEY` is set.

```bash
python voice_agent.py --llm gemini
```

## How it Works

1.  **Record:** Press Enter to record 5 seconds of audio.
2.  **Transcribe:** Whisper converts speech to text locally.
3.  **Think:** The text is sent to either Ollama or Gemini for a response.
4.  **Speak:** Bark synthesizes the response into speech (chunked sentence by sentence for longer responses).
