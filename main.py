from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
import logging
import os
import openai
from utils import wait_and_check_status, get_lyrics_task_info, init_bucket, cleanup_songs_directory, openai_analysis, upload_to_s3, clip_audio, merge_songs
import json
import http.client
import asyncio
import uvicorn
from basic_pitch.inference import predict
import requests
from fastapi.responses import JSONResponse
from botocore.exceptions import NoCredentialsError
from song_merger import merge_songs
import uuid
from fastapi.middleware.cors import CORSMiddleware


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("song_generation.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

SUNOAI_API_KEY = os.getenv("SUNOAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


app = FastAPI()
app.state.s3_client = None  # Will be initialized in main

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    try:
        import boto3
        app.state.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        logging.info("S3 client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize S3 client: {e}")
        raise

# Pydantic models for request validation
class SongRequest(BaseModel):
    customMode: bool
    instrumental: bool
    prompt: str
    lyrics: Optional[str] # Optional parameter for lyrics
    style: Optional[str] = None
    title: Optional[str] = None
    model: str = "V4"  # Default value is "V4"
    clip: Optional[bool]  # Optional parameter for clipping
    clipLength: Optional[int]
    clipStart: Optional[int]  # Optional parameter for clipping start time
    lyricsStart: Optional[int]  # Correctly annotate lyricsStart
    promptStrength: Optional[bool]  # Correctly annotate lyricsStart
    lyricsStrength: Optional[bool]  # Correctly annotate lyricsStart
    clarity: Optional[bool]  # Correctly annotate lyricsStart
    negativeTags: Optional[str]  # Optional parameter for lyrics start time

class AudioExtensionRequest(BaseModel):
    defaultParamFlag: bool
    audioId: str
    prompt: Optional[str] = None
    style: Optional[str] = None
    title: Optional[str] = None
    continueAt: Optional[int] = None
    model: str = "V4"
    lyricsStart: Optional[int]
    promptStrength: Optional[bool]  # Correctly annotate lyricsStart
    lyricsStrength: Optional[bool]  # Correctly annotate lyricsStart
    clarity: Optional[bool]  # Correctly annotate lyricsStart
    songUrl: Optional[str]
    description: Optional[str]

class MusicPrompt(BaseModel):
    prompt: str

class UploadCoverRequest(BaseModel):
    prompt: str
    style: Optional[str] = None
    title: Optional[str] = None
    customMode: bool = True
    instrumental: bool = True
    model: str = "V4"
    negativeTags: Optional[str] = None

@app.post("/ai/generate-song/")
async def generate_song(request: SongRequest):
    if request.lyricsStart != 0:
        logging.info(f"lyricsStart detected: {request.lyricsStart}. Generating merged song...")
        try:
            conn = http.client.HTTPSConnection("apibox.erweima.ai")
            
            # First generation with instrumental=True
            payload_instrumental = {
                "prompt": request.prompt,
                "customMode": request.customMode,
                "instrumental": True,  # Instrumental mode
                "model": request.model,
                "negativeTags": request.negativeTags,
                "callBackUrl": "https://api.example.com/callback"
            }
            if request.style:
                payload_instrumental["style"] = request.style
            if request.title:
                payload_instrumental["title"] = request.title

            conn.request("POST", "/api/v1/generate", body=json.dumps(payload_instrumental), headers={
                'Content-Type': "application/json",
                'Accept': "application/json",
                'Authorization': f"Bearer {SUNOAI_API_KEY}"
            })
            res = conn.getresponse()
            data = res.read()
            response_data_instrumental = json.loads(data.decode("utf-8"))
            if response_data_instrumental.get("code") != 200:
                raise HTTPException(status_code=400, detail="Instrumental generation failed")
            task_id_instrumental = response_data_instrumental["data"].get("taskId")
            result_instrumental = await wait_and_check_status(
                task_id=task_id_instrumental,
                api_key=SUNOAI_API_KEY,
                s3_client=app.state.s3_client,
                bucket_name=AWS_BUCKET_NAME
            )
            song1_path = result_instrumental["data"][0]["stream_audio_url"]  # Instrumental URL

            # Second generation with instrumental=False
            payload_non_instrumental = {
                "prompt": request.prompt,
                "customMode": request.customMode,
                "instrumental": False,  # Non-instrumental mode
                "model": request.model,
                "negativeTags": request.negativeTags,
                "callBackUrl": "https://api.example.com/callback"
            }
            if request.style:
                payload_non_instrumental["style"] = request.style
            if request.title:
                payload_non_instrumental["title"] = request.title

            conn.request("POST", "/api/v1/generate", body=json.dumps(payload_non_instrumental), headers={
                'Content-Type': "application/json",
                'Accept': "application/json",
                'Authorization': f"Bearer {SUNOAI_API_KEY}"
            })
            res = conn.getresponse()
            data = res.read()
            response_data_non_instrumental = json.loads(data.decode("utf-8"))
            if response_data_non_instrumental.get("code") != 200:
                raise HTTPException(status_code=400, detail="Non-instrumental generation failed")
            task_id_non_instrumental = response_data_non_instrumental["data"].get("taskId")
            result_non_instrumental = await wait_and_check_status(
                task_id=task_id_non_instrumental,
                api_key=SUNOAI_API_KEY,
                s3_client=app.state.s3_client,
                bucket_name=AWS_BUCKET_NAME
            )
            song2_path = result_non_instrumental["data"][0]["stream_audio_url"]  # Non-instrumental URL

            # Download the audio files locally first
            import requests
            import tempfile
            
            # Create a temp directory for our files
            temp_dir = os.path.join(os.path.dirname(__file__), "temp_audio")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download song files
            song1_local_path = os.path.join(temp_dir, f"instrumental_{uuid.uuid4()}.mp3")
            song2_local_path = os.path.join(temp_dir, f"vocals_{uuid.uuid4()}.mp3")
            
            logging.info(f"Downloading instrumental from: {song1_path}")
            logging.info(f"Saving instrumental to: {song1_local_path}")
            # Download song1
            with requests.get(song1_path, stream=True) as r:
                r.raise_for_status()
                with open(song1_local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logging.info(f"Downloading vocals from: {song2_path}")
            logging.info(f"Saving vocals to: {song2_local_path}")
            # Download song2
            with requests.get(song2_path, stream=True) as r:
                r.raise_for_status()
                with open(song2_local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Create unique filename for the output
            unique_filename = f"merged_{uuid.uuid4()}.mp3"
            output_path = os.path.join(temp_dir, unique_filename)
            
            # Verify files exist before merging
            if not os.path.exists(song1_local_path):
                raise FileNotFoundError(f"Instrumental file not found: {song1_local_path}")
            if not os.path.exists(song2_local_path):
                raise FileNotFoundError(f"Vocals file not found: {song2_local_path}")
                
            logging.info(f"Files ready for merging:")
            logging.info(f"Instrumental: {song1_local_path} (exists: {os.path.exists(song1_local_path)})")
            logging.info(f"Vocals: {song2_local_path} (exists: {os.path.exists(song2_local_path)})")
            
            # Generate merged song using local files
            transition_time = request.lyricsStart * 1000  # Convert seconds to milliseconds
            merge_songs(song1_local_path, song2_local_path, output_path, transition_time, auto_align=True)
            logging.info(f"Songs merged successfully. Output saved at {output_path}")

            # Upload merged song to AWS S3
            s3_key = f"merged-songs/{unique_filename}"
            s3_url = upload_to_s3(app.state.s3_client, output_path, AWS_BUCKET_NAME, s3_key)
            if not s3_url:
                raise HTTPException(status_code=500, detail="Failed to upload merged song to AWS S3")
                
            # Clean up temporary files
            try:
                os.remove(song1_local_path)
                os.remove(song2_local_path)
                os.remove(output_path)
                logging.info("Temporary audio files cleaned up")
            except Exception as e:
                logging.warning(f"Error cleaning up temporary files: {e}")

            logging.info(f"Merged song uploaded to S3: {s3_url}")
            
            # Create response in the expected format
            response_data = {
                "status": "SUCCESS",
                "data": [
                    # Include details from the instrumental version
                    {
                        "id": str(uuid.uuid4()),
                        "title": f"{request.title or 'Merged Song'}_instrumental",
                        "stream_audio_url": song1_path,
                        "image_url": result_non_instrumental["data"][0].get("image_url", ""),
                        "prompt": request.prompt,
                        "duration": result_non_instrumental["data"][0].get("duration", 0),
                        "tags": result_instrumental["data"][0].get("tags", [])
                    },
                    # Include details from the vocal version but with merged song URL
                    {
                        "id": str(uuid.uuid4()),
                        "title": f"{request.title or 'Merged Song'}_merged",
                        "stream_audio_url": s3_url,
                        "image_url": result_non_instrumental["data"][0].get("image_url", ""),
                        "prompt": request.prompt,
                        "duration": result_non_instrumental["data"][0].get("duration", 0),
                        "tags": result_non_instrumental["data"][0].get("tags", [])
                    }
                ]
            }
            
            return response_data

        except Exception as e:
            logging.error(f"Error during merged song generation or upload: {e}")
            raise HTTPException(status_code=500, detail=f"Error during merged song generation or upload: {str(e)}")
    else:
        print("ClipStart Time:", request.clipStart)
        if request.promptStrength == True:
            openai.api_key = OPENAI_API_KEY
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": f"Please provide a prompt for the song with the following details: {request.prompt}.\nDo not extend the response more than 250 characters."}
                    ],
                    max_tokens=100
                )
                request.prompt = response.choices[0].message.content
            except Exception as e:
                logging.error(f"OpenAI analysis failed: {e}")
                raise HTTPException(status_code=500, detail=f"OpenAI analysis failed: {e}")
        if request.customMode:    
            conn = http.client.HTTPSConnection("apibox.erweima.ai")
            print("Custom Mode is enabled")
            payload = {
                "prompt": request.lyrics,
                "customMode": request.customMode,
                "instrumental": request.instrumental,
                "model": request.model,  # Use the model parameter from the request
                "negativeTags": request.negativeTags,
                "callBackUrl": "https://api.example.com/callback"
            }
        else:
            print("Custom Mode is disabled")
            conn = http.client.HTTPSConnection("apibox.erweima.ai")
            payload = {
                "prompt": request.prompt,
                "customMode": request.customMode,
                "instrumental": request.instrumental,
                "model": request.model,  # Use the model parameter from the request
                "negativeTags": request.negativeTags,
                "callBackUrl": "https://api.example.com/callback"
            }
        if request.style:
            payload["style"] = request.style
        if request.title:
            payload["title"] = request.title

        payload_json = json.dumps(payload)
        headers = {
            'Content-Type': "application/json",
            'Accept': "application/json",
            'Authorization': f"Bearer {SUNOAI_API_KEY}"
        }

        logging.info("Starting song generation process...")

        try:
            # Send the request
            conn.request("POST", "/api/v1/generate", body=payload_json, headers=headers)
            res = conn.getresponse()
            data = res.read()
            response_data = json.loads(data.decode("utf-8"))
            logging.info("Song generation request sent successfully.")

            if response_data.get("code") == 200:
                task_id = response_data["data"].get("taskId")
                if not task_id:
                    logging.error("Error: Task ID not found in the response.")
                    raise HTTPException(status_code=404, detail="Task ID not found in the response")

                # Backtrack the music creation process for the task ID
                logging.info(f"Backtracking music creation process for Task ID: {task_id}")
                result = await wait_and_check_status(
                    task_id=task_id,
                    api_key=SUNOAI_API_KEY,
                    s3_client=app.state.s3_client,
                    bucket_name=AWS_BUCKET_NAME
                )

                # Check if clipStart is provided
                if request.clipStart != 0:
                    logging.info(f"Clipping audio starting at {request.clipStart} seconds...")
                    try:
                        # Download the generated song locally
                        song_url = result["data"][0]["stream_audio_url"]
                        local_clipped_path, _ = clip_audio(song_url, clipStart=request.clipStart)

                        # Upload the clipped song to S3
                        clipped_s3_key = f"clipped-songs/{os.path.basename(local_clipped_path)}"
                        clipped_s3_url = upload_to_s3(app.state.s3_client, local_clipped_path, AWS_BUCKET_NAME, clipped_s3_key)
                        if not clipped_s3_url:
                            raise HTTPException(status_code=500, detail="Failed to upload clipped song to AWS S3")

                        # Clean up the local clipped file
                        if os.path.exists(local_clipped_path):
                            os.remove(local_clipped_path)
                            logging.info(f"Cleaned up local clipped file: {local_clipped_path}")

                        # Update the result with the clipped song URL
                        result["data"][0]["clipped_audio_url"] = clipped_s3_url
                        logging.info(f"Clipped song uploaded to S3: {clipped_s3_url}")
                    except Exception as e:
                        logging.error(f"Error during audio clipping or upload: {e}")
                        raise HTTPException(status_code=500, detail=f"Error during audio clipping or upload: {str(e)}")
                # For controlling the length of the song, i.e, 30 seconds
                if request.clip:
                    logging.info("Clipping is enabled")
                    for song in result["data"]:
                        try:
                            song_url = song["stream_audio_url"]
                            local_clipped_path, _ = clip_audio(song_url, clipStart=0, clipEnd= request.clipLength)
                            clipped_s3_key = f"clipped-songs/{os.path.basename(local_clipped_path)}"
                            clipped_s3_url = upload_to_s3(app.state.s3_client, local_clipped_path, AWS_BUCKET_NAME, clipped_s3_key)

                            if not clipped_s3_url:
                                raise HTTPException(status_code=500, detail="Failed to upload clipped song to AWS S3")

                            # Clean up the local clipped file
                            if os.path.exists(local_clipped_path):
                                os.remove(local_clipped_path)
                                logging.info(f"Cleaned up local clipped file: {local_clipped_path}")

                            # Overwrite stream_audio_url with the clipped song URL
                            song["stream_audio_url"] = clipped_s3_url
                            logging.info(f"Clipped song uploaded to S3 and stream_audio_url updated: {clipped_s3_url}")
                        except Exception as e:
                            logging.error(f"Error during audio clipping or upload for song {song['id']}: {e}")
                            raise HTTPException(status_code=500, detail=f"Error during audio clipping or upload: {str(e)}")
                    
                return result
            else:
                logging.error(f"Error: Received code {response_data.get('code')} - {response_data.get('msg')}")
                raise HTTPException(status_code=400, detail=response_data.get('msg'))

        except Exception as e:
            logging.error(f"Failed to send song generation request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
        
@app.post("/ai/extend-audio/")
async def extend_audio(request: AudioExtensionRequest):
    if request.promptStrength == True:
            openai.api_key = OPENAI_API_KEY
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": f"Please provide a prompt for the song with the following details: {request.prompt}"}
                    ],
                    max_tokens=200
                )
                request.prompt = response.choices[0].message.content
            except Exception as e:
                logging.error(f"OpenAI analysis failed: {e}")
                raise HTTPException(status_code=500, detail=f"OpenAI analysis failed: {e}")
    if request.defaultParamFlag:
        
        if not request.prompt or len(request.prompt) > 3000:
            raise HTTPException(status_code=400, detail="Prompt is required and must not exceed 3000 characters")
        if not request.style or len(request.style) > 200:
            raise HTTPException(status_code=400, detail="Style is required and must not exceed 200 characters")
        if not request.title or len(request.title) > 80:
            raise HTTPException(status_code=400, detail="Title is required and must not exceed 80 characters")

    conn = http.client.HTTPSConnection("apibox.erweima.ai")
    payload = {
        "defaultParamFlag": request.defaultParamFlag,
        "audioId": request.audioId,
        "model": request.model,
        "callBackUrl": "https://api.example.com/callback"
    }

    if request.defaultParamFlag:
        payload.update({
            "prompt": request.prompt,
            "style": request.style,
            "title": request.title
        })

    if request.continueAt is not None:
        payload["continueAt"] = request.continueAt

    payload_json = json.dumps(payload)
    headers = {
        'Content-Type': "application/json",
        'Accept': "application/json",
        'Authorization': f"Bearer {SUNOAI_API_KEY}"
    }

    logging.info("Starting audio extension process...")
    try:
        conn.request("POST", "/api/v1/generate/extend", body=payload_json, headers=headers)
        res = conn.getresponse()
        data = res.read()
        response_data = json.loads(data.decode("utf-8"))
        logging.info("Audio extension request sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send audio extension request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if response_data.get("code") == 200:
        task_id = response_data["data"].get("taskId")
        if task_id:
            logging.info(f"Task ID received for audio extension: {task_id}")
            result = await wait_and_check_status(
                task_id=task_id,
                api_key=SUNOAI_API_KEY,
                s3_client=app.state.s3_client,
                bucket_name=AWS_BUCKET_NAME
            )
            return result
        else:
            logging.error("Error: Task ID not found in the response")
            raise HTTPException(status_code=404, detail="Task ID not found")
    else:
        logging.error(f"Error: Received code {response_data.get('code')} - {response_data.get('msg')}")
        raise HTTPException(status_code=400, detail=response_data.get('msg'))

@app.post("/ai/generate-lyrics/")
async def generate_lyrics(music: MusicPrompt):
    try:
        conn = http.client.HTTPSConnection("apibox.erweima.ai")
        payload = json.dumps({
            "prompt": music.prompt,
            "callBackUrl": "https://api.example.com/callback"
        })

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {SUNOAI_API_KEY}'  
        }

        conn.request("POST", "/api/v1/lyrics", payload, headers)
        res = conn.getresponse()
        data = res.read()
        response_data = json.loads(data.decode('utf-8'))

        if res.status != 200:
            raise HTTPException(status_code=res.status, detail=f"Error: {data.decode('utf-8')}")

        task_id = response_data.get("data", {}).get("taskId")
        if not task_id:
            raise HTTPException(status_code=400, detail="Task ID not found in response")

        await asyncio.sleep(40)  
        result = await get_lyrics_task_info(task_id, SUNOAI_API_KEY)
        return {"status": "SUCCESS", "lyrics": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")


class URLPayload(BaseModel):
    url: str

@app.post("/ai/process-audio/")
async def process_audio(file: UploadFile = File(...)):

    # Step 1: Save the uploaded file locally
    local_audio_path = f"./{file.filename}"
    try:
        with open(local_audio_path, "wb") as audio_file:
            content = await file.read()
            audio_file.write(content)
        logging.info(f"Audio file saved locally: {local_audio_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Step 2: Use the saved file with the predict function
    try:
        model_output, midi_data, note_events = predict(local_audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during MIDI prediction: {str(e)}")

    # Step 3: Save the MIDI data to a file
    output_path = "./out.mid"
    try:
        midi_data.write(output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving MIDI file: {str(e)}")

    logging.info("Model Output:", model_output)
    logging.info("MIDI Data:", midi_data)
    logging.info("Note Events:", note_events)

    # Step 4: Analyze the MIDI file
    try:
        analysis_result = openai_analysis(output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing MIDI file: {str(e)}")

    # Step 5: Delete the locally saved audio and MIDI files
    if os.path.exists(local_audio_path):
        os.remove(local_audio_path)
        logging.info(f"Deleted the uploaded audio file: {local_audio_path}")
    if os.path.exists(output_path):
        os.remove(output_path)
        logging.info(f"Deleted the MIDI file: {output_path}")

    return {"analysis_result": analysis_result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



