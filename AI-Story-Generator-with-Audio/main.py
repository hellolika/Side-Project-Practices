import os
import base64
import struct
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
        
        # 1. Generate Story Text
        if request.language == "km":
            story_prompt = f"Write a very short, engaging story (max 5 sentences) in Khmer language based on this prompt: {request.prompt}. Return ONLY the story text."
        else:
            story_prompt = f"Write a very short, engaging story (max 5 sentences) based on this prompt: {request.prompt}. Return ONLY the story text."
        
        story_res = model_text.generate_content(story_prompt)
        full_story_text = story_res.text.strip()
        print(" full story response: {story_res}")
        
        # Extract Title
        if request.language == "km":
            title_prompt = f"Give me a short 3-word title in Khmer for this story: {full_story_text}"
        else:
            title_prompt = f"Give me a short 3-word title for this story: {full_story_text}"
            
        title_res = model_text.generate_content(title_prompt)
        title = title_res.text.strip().replace('"', '')

        # 2. Generate Cover Image
        image_base64 = None
        # try:
        #     image_prompt = f"A storybook illustration for: {request.prompt}, {full_story_text[:50]}..."
        #     imagen_model = genai.GenerativeModel("imagen-3.0-generate-001")
        #     image_res = imagen_model.generate_content(image_prompt)
        #     image_base64 = image_res.candidates[0].content.parts[0].inline_data.data
        # except Exception as e:
        #     print(f"Image generation warning: {e}")

        # 3. Process Audio Segments
        # Regex to split by delimiters including Khmer '។'
        # sentences = [s.strip() for s in re.split(r'(?<=[.!?។])\s*', full_story_text) if s.strip()]
        
        segments = []
        tts_client = genai.GenerativeModel("gemini-2.5-flash-preview-tts")

        if not full_story_text:
            return StoryResponse(title=title, image_base64=image_base64, segments=segments)

        try:
            # Wrap the text in a directive to improve prosody and reduce robotic tone
            tts_prompt = f"Say in an engaging storytelling voice: {full_story_text}"
            if request.language == "km":
                tts_prompt = (
                    "Say this in an engaging, cinematic storytelling voice with a clear "
                    f"Khmer accent and intonation: {full_story_text}"
                )

            response = tts_client.generate_content(
                tts_prompt,
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

            # print("Raw TTS response:", response)

            # If available in your client version, this is often easier to inspect:
            # print("Response as dict:", response.to_dict())

            # Inspect candidates/parts before indexing
            # print("Candidates:", getattr(response, "candidates", None))
            if response.candidates:
                first = response.candidates[0]
                print("First candidate parts:", getattr(first.content, "parts", None))

            inline = response.candidates[0].content.parts[0].inline_data
            print("mime:", getattr(inline, "mime_type", None), "type:", type(inline.data), "len:", len(inline.data) if inline.data else None)
            mime = getattr(inline, "mime_type", None)
            data = inline.data
            wav_b64 = None

            # If SDK returns base64 string, decode it. If it returns bytes, keep as-is.
            if isinstance(data, str):
                data = base64.b64decode(data)

            if mime in ("audio/wav", "audio/x-wav"):
                wav_b64 = base64.b64encode(data).decode("utf-8")
            else:
                # assume raw PCM16 mono @ 24kHz
                wav_b64 = pcm_to_wav_base64(data, sample_rate=24000)
            segments.append(Segment(text=full_story_text, audio_base64=wav_b64))

        except Exception as e:
            print("Error during TTS, Process Audio:", repr(e))
            segments.append(Segment(text=full_story_text, audio_base64=None))

        return StoryResponse(
            title=title,
            image_base64=image_base64,
            segments=segments
        )

    except Exception as e:
        print(f"Critical Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)