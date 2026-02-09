import os
import re
import base64
import struct
import wave
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API Key
# Make sure to set GOOGLE_API_KEY in your environment or .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoryRequest(BaseModel):
    prompt: str
    voice_name: str = "Kore"  # Options: Kore, Fenrir, Puck, etc.

class Segment(BaseModel):
    text: str
    audio_base64: Optional[str] = None

class StoryResponse(BaseModel):
    title: str
    image_base64: Optional[str] = None
    segments: List[Segment]

def pcm_to_wav_base64(pcm_data: bytes, sample_rate: int = 24000) -> str:
    """Converts raw PCM16 data to a base64 encoded WAV string."""
    header = struct.pack('<4sI4s', b'RIFF', 36 + len(pcm_data), b'WAVE')
    fmt = struct.pack('<4sIHHIIH', b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    data_header = struct.pack('<4sI', b'data', len(pcm_data))
    
    wav_bytes = header + fmt + data_header + pcm_data
    return base64.b64encode(wav_bytes).decode('utf-8')

@app.post("/generate", response_model=StoryResponse)
async def generate_story(request: StoryRequest):
    try:
        # 1. Generate the Story Text
        model_text = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
        story_prompt = f"Write a very short, engaging story (max 5 sentences) based on this prompt: {request.prompt}. Return ONLY the story text."
        
        story_res = model_text.generate_content(story_prompt)
        full_story_text = story_res.text.strip()
        
        # Extract a title (simple heuristic)
        title_res = model_text.generate_content(f"Give me a short 3-word title for this story: {full_story_text}")
        title = title_res.text.strip().replace('"', '')

        # 2. Generate Cover Image
        # Note: Using imagen-3.0 as it's the standard stable version in many environments, 
        # but you can switch to imagen-4.0-generate-001 if you have access.
        try:
            image_prompt = f"A storybook illustration for: {request.prompt}, {full_story_text[:50]}..."
            imagen_model = genai.GenerativeModel("imagen-3.0-generate-001")
            image_res = imagen_model.generate_content(image_prompt)
            image_base64 = image_res.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            print(f"Image generation failed: {e}")
            image_base64 = None # Fallback if image fails

        # 3. Process Audio Segments (Line-by-Line)
        # Split text into sentences for granular highlighting
        sentences = re.split(r'(?<=[.!?]) +', full_story_text)
        segments = []

        tts_client = genai.GenerativeModel("gemini-2.5-flash-preview-tts")
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            try:
                response = tts_client.generate_content(
                    sentence,
                    generation_config={
                        "response_modalities": ["AUDIO"],
                        "speech_config": {
                            "voice_config": {
                                "prebuilt_voice_config": {
                                    "voice_name": request.voice_name
                                }
                            }
                        }
                    }
                )
                
                # Extract PCM data
                pcm_data = response.candidates[0].content.parts[0].inline_data.data
                # Convert to WAV base64 for browser playback
                wav_b64 = pcm_to_wav_base64(base64.b64decode(pcm_data))
                
                segments.append(Segment(text=sentence, audio_base64=wav_b64))
            except Exception as e:
                print(f"TTS failed for segment: {e}")
                segments.append(Segment(text=sentence, audio_base64=None))

        return StoryResponse(
            title=title,
            image_base64=image_base64,
            segments=segments
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)