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
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoryRequest(BaseModel):
    prompt: str
    voice_name: str = "Kore"
    language: str = "en"  # Options: "en", "km"

class Segment(BaseModel):
    text: str
    audio_base64: Optional[str] = None

class StoryResponse(BaseModel):
    title: str
    image_base64: Optional[str] = None
    segments: List[Segment]

def pcm_to_wav_base64(pcm_data: bytes, sample_rate: int = 24000) -> str:
    header = struct.pack('<4sI4s', b'RIFF', 36 + len(pcm_data), b'WAVE')
    fmt = struct.pack('<4sIHHIIH', b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    data_header = struct.pack('<4sI', b'data', len(pcm_data))
    return base64.b64encode(header + fmt + data_header + pcm_data).decode('utf-8')

@app.post("/generate", response_model=StoryResponse)
async def generate_story(request: StoryRequest):
    try:
        model_text = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
        
        # 1. Generate Story Text based on Language
        if request.language == "km":
            # Khmer Prompt
            story_prompt = f"Write a very short, engaging story (max 5 sentences) in Khmer language based on this prompt: {request.prompt}. Return ONLY the story text."
        else:
            # English Prompt
            story_prompt = f"Write a very short, engaging story (max 5 sentences) based on this prompt: {request.prompt}. Return ONLY the story text."
        
        story_res = model_text.generate_content(story_prompt)
        full_story_text = story_res.text.strip()
        
        # Extract Title
        if request.language == "km":
            title_prompt = f"Give me a short 3-word title in Khmer for this story: {full_story_text}"
        else:
            title_prompt = f"Give me a short 3-word title for this story: {full_story_text}"
            
        title_res = model_text.generate_content(title_prompt)
        title = title_res.text.strip().replace('"', '')

        # 2. Generate Cover Image
        try:
            image_prompt = f"A storybook illustration for: {request.prompt}, {full_story_text[:50]}..."
            imagen_model = genai.GenerativeModel("imagen-3.0-generate-001")
            image_res = imagen_model.generate_content(image_prompt)
            image_base64 = image_res.candidates[0].content.parts[0].inline_data.data
        except Exception:
            image_base64 = None

        # 3. Process Audio Segments
        # Regex updated to include Khmer '។' delimiter and handle variable spacing
        # (?<=[...]) is a lookbehind assertion to keep the delimiter
        sentences = [s.strip() for s in re.split(r'(?<=[.!?។])\s*', full_story_text) if s.strip()]
        
        segments = []
        tts_client = genai.GenerativeModel("gemini-2.5-flash-preview-tts")
        
        for sentence in sentences:
            if not sentence: continue
                
            try:
                # Attempt to generate audio using Gemini TTS
                # Note: If Gemini TTS struggles with Khmer, the frontend will fallback to browser TTS
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
                pcm_data = response.candidates[0].content.parts[0].inline_data.data
                wav_b64 = pcm_to_wav_base64(base64.b64decode(pcm_data))
                segments.append(Segment(text=sentence, audio_base64=wav_b64))
            except Exception as e:
                print(f"TTS failed for segment (using fallback): {e}")
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