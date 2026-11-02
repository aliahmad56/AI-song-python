import sqlite3
import logging
import http.client
import json
import asyncio
import requests
from fastapi import HTTPException
from typing import Dict, Any
import boto3
import os
from botocore.exceptions import ClientError
from typing import Optional
from music21 import converter, instrument, tempo, key, meter
import openai
from dotenv import load_dotenv
from pydub import AudioSegment
import librosa


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

async def wait_and_check_status(
    task_id: str, 
    api_key: str, 
    s3_client: boto3.client, 
    bucket_name: str
) -> Dict[str, Any]:
    retry_delay = 30
    terminal_states = [
        "CREATE_TASK_FAILED",
        "GENERATE_AUDIO_FAILED",
        "CALLBACK_EXCEPTION",
        "SENSITIVE_WORD_ERROR"
    ]
    attempt = 1
    
    while True:
        logging.info(f"Checking status attempt {attempt}")
        try:
            conn = http.client.HTTPSConnection("apibox.erweima.ai")
            headers = {
                'Accept': "application/json",
                'Authorization': f"Bearer {api_key}"
            }
            
            conn.request("GET", f"/api/v1/generate/record-info?taskId={task_id}", headers=headers)
            res = conn.getresponse()
            data = res.read()
            response = json.loads(data.decode("utf-8"))
            
            if response.get("code") == 200 and "data" in response:
                status = response["data"].get("status")
                suno_data = response["data"].get("response", {}).get("sunoData", [])
                
                if status == "SUCCESS" and suno_data:
                    logging.info("Song generation completed successfully.")
                    result = []
                    for index, item in enumerate(suno_data, 1):
                        result.append(await process_suno_data(item, index, s3_client, bucket_name))
                    return {"status": "SUCCESS", "data": result}
                elif status in terminal_states:
                    logging.error(f"Terminal error state reached: {status}")
                    return {"status": status}
                else:
                    logging.info(f"Task still processing (status: {status}), waiting {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    attempt += 1
                    continue
            else:
                logging.error(f"Error: Received code {response.get('code')} - {response.get('msg')}")
                raise HTTPException(status_code=400, detail=response.get('msg'))
                
        except Exception as e:
            logging.error(f"Failed to check status on attempt {attempt}: {e}")
            await asyncio.sleep(retry_delay)
            attempt += 1
            continue

async def get_lyrics_task_info(task_id: str, api_key: str) -> Dict[str, Any]:
    url = f"https://apibox.erweima.ai/api/v1/lyrics/record-info?taskId={task_id}"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Error: {response.text}")
        lyrics_data = response.json().get("data", {}).get("response", {}).get("data", [])
        return [{"title": item["title"], "text": item["text"]} for item in lyrics_data]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")

async def process_suno_data(item: Dict[str, Any], index: int, s3_client: boto3.client, bucket_name: str) -> Dict[str, Any]:
    ids = item.get("id")
    title = item.get("title")
    source_audio_url = item.get("sourceAudioUrl")
    image_url = item.get("imageUrl")
    prompt = item.get("prompt")
    duration = item.get("duration")
    tags = item.get("tags", [])

    # Process song storage
    s3_url = await process_song_storage(source_audio_url, ids, s3_client, bucket_name)

    # Remove database saving functionality
    # save_success = save_music_to_db(
    #     music_id=ids,
    #     title=f"{title}_version_{index}",
    #     version=index,
    #     audio_url=s3_url or source_audio_url,  # Use S3 URL if available, otherwise use original
    #     image_url=image_url
    # )
    # save_success = False # Set save_success to False since we are not saving to the database

    return {
        "id": ids,
        "title": f"{title}_version_{index}",
        "stream_audio_url": s3_url or source_audio_url,  # Use S3 URL if available
        "image_url": image_url,
        "prompt": prompt,
        "duration": duration,
        "tags": [tag for tag in tags.split(",")]
        # "saved_to_db": save_success
    }

def init_bucket(aws_access_key_id: str, aws_secret_access_key: str, bucket_name: str) -> boto3.client:
    """Initialize S3 client and ensure bucket exists"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logging.info(f"Bucket '{bucket_name}' exists and is accessible.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '403':
                logging.warning(f"Access denied to bucket '{bucket_name}'. Ensure the bucket exists and permissions are correct.")
            elif error_code == '404':
                logging.error(f"Bucket '{bucket_name}' does not exist, and creation is not permitted due to insufficient permissions.")
                raise
            else:
                logging.error(f"Unexpected error while accessing bucket '{bucket_name}': {e}")
                raise
        return s3_client
    except Exception as e:
        logging.error(f"Failed to initialize S3 bucket: {e}")
        raise

async def download_song(url: str, song_id: str) -> Optional[str]:
    """Download song from URL to local storage"""
    try:
        # Create songs directory if it doesn't exist
        songs_dir = os.path.join(os.getcwd(), 'songs')
        os.makedirs(songs_dir, exist_ok=True)
        
        # Define local file path
        local_path = os.path.join(songs_dir, f"{song_id}.mp3")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logging.info(f"Successfully downloaded song to {local_path}")
        return local_path
    except Exception as e:
        logging.error(f"Failed to download song: {e}")
        return None

def upload_to_s3(s3_client: boto3.client, file_path: str, bucket_name: str, s3_key: str) -> Optional[str]:
    """Upload file to S3 and return the URL"""
    try:
        # Upload file with content type
        s3_client.upload_file(
            file_path, 
            bucket_name, 
            s3_key,
            ExtraArgs={'ContentType': 'audio/mpeg'}
        )
        
        # Generate presigned URL that expires in 7 days
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': s3_key
            },
            ExpiresIn=604800  # 7 days in seconds
        )
        url = url.split('?')[0]  # Remove query parameters
        logging.info(f"Successfully uploaded file to S3: {url}")
        return url

    except Exception as e:
        logging.error(f"Failed to upload to S3: {e}")
        return None

async def process_song_storage(url: str, song_id: str, s3_client: boto3.client, bucket_name: str) -> Optional[str]:
    """Process song storage workflow"""
    try:
        # Download song to local songs directory
        local_path = await download_song(url, song_id)
        if not local_path:
            return None
        
        # Upload to S3 bucket
        s3_url = upload_to_s3(s3_client, local_path, bucket_name, song_id)
        
        # Clean up local file after upload
        if os.path.exists(local_path):
            os.remove(local_path)
            logging.info(f"Cleaned up local file: {local_path}")
        
        return s3_url
    except Exception as e:
        logging.error(f"Failed to process song storage: {e}")
        # Clean up local file in case of error
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
        return None

def cleanup_songs_directory():
    """Clean up any remaining files in the songs directory"""
    try:
        songs_dir = os.path.join(os.getcwd(), 'songs')
        if os.path.exists(songs_dir):
            for file in os.listdir(songs_dir):
                file_path = os.path.join(songs_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logging.info(f"Cleaned up file: {file_path}")
                except Exception as e:
                    logging.error(f"Error cleaning up file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error cleaning up songs directory: {e}")




def analyze_midi(file_path):
    score = converter.parse(file_path)
    instruments = set()
    tempos = set()
    keys = set()
    time_signatures = set()

    for part in score.parts:
        # Instruments
        instrs = part.getElementsByClass(instrument.Instrument)
        if instrs:
            instruments.add(instrs[0].instrumentName)

        # Tempos
        tempos.update(t.getQuarterBPM() for t in part.recurse().getElementsByClass(tempo.MetronomeMark))

        # Key signatures
        keys.update(str(k) for k in part.recurse().getElementsByClass(key.KeySignature))

        # Time signatures
        time_signatures.update(str(ts.ratioString) for ts in part.recurse().getElementsByClass(meter.TimeSignature))

    print("🎹 Instruments:", ', '.join(instruments))
    print("🎵 Tempos:", ', '.join(map(str, tempos)))
    print("🎼 Key Signatures:", ', '.join(keys))
    print("🕒 Time Signatures:", ', '.join(time_signatures))
    return instruments, tempos, keys, time_signatures


def openai_analysis(local_file_path):
    instruments, tempos, keys, time_signatures = analyze_midi(local_file_path)
    prompt = """Analyze the following information extracted from the .midi file and provide me with the prompt to generate the music with lyrics in suno ai api:
    Instruments: {instruments}
    Tempos: {tempos}
    Keys: {keys}
    Time Signatures: {time_signatures}""".format(
        instruments=', '.join(instruments),
        tempos=', '.join(map(str, tempos)),
        keys=', '.join(keys),  
        time_signatures=', '.join(time_signatures)
    )    
    try:
        result = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", 
                "content": "You are a music expert. I will provide you with a MIDI file, and you will analyze it and provide me with the prompt to generate the music with lyrics in suno ai api."
                "Rules:"
                "1. The text propmt should be able to generate the music in SUNO AI API."
                "2. I just need the text prompt to generate the music no extra music"
                "3. for your information, the text prompt will look like this:A gentle, flowing instrumental piece titled 'Waves of Serenity'. 120 BPM, 4/4 time. Soft ambient textures with a focus on calm piano melodies and subtle atmospheric layers and soulful lyrics. Ideal for meditation, relaxation, or cinematic background. Emphasize emotional warmth, spaciousness, and a sense of peaceful motion—like waves lapping on a tranquil shore."
                "4. Do not provide anything other than the text prompt."
                "5. It should not be more than 400 characters."
                },
                {
                "role": "user", 
                "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=100
        )
        print(result.choices[0].message['content'])
        gen_prompt = result.choices[0].message['content']
        return gen_prompt
    except Exception as e:
        logging.error(f"OpenAI analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI analysis failed: {str(e)}")

    

def clip_audio(audio_url: str, clipStart: int = 0, clipEnd: int = 32) -> tuple[str, float]:
    """
    Clips the audio from clipStart (in seconds) to clipEnd (in seconds).
    If clipEnd is None, defaults to 32 seconds after clipStart.
    Returns the path to the clipped audio and its duration in seconds.
    """
    try:
        # Download the audio file if it's a URL
        if audio_url.startswith("http"):
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            local_audio_path = "temp_audio.mp3"
            with open(local_audio_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            local_audio_path = audio_url

        # Load the audio file
        audio = AudioSegment.from_file(local_audio_path)

        # Convert start and end times to milliseconds
        start_ms = clipStart * 1000
        if clipEnd is None:
            end_ms = start_ms + (32 * 1000)
        else:
            end_ms = clipEnd * 1000

        # Ensure end_ms does not exceed audio length
        end_ms = min(end_ms, len(audio))

        # Clip the audio from start to end time
        clipped_audio = audio[start_ms:end_ms]

        # Generate the clipped audio file name
        if audio_url.startswith("http"):
            original_name = "temp_audio"
        else:
            original_name = os.path.splitext(os.path.basename(audio_url))[0]
        clipped_audio_path = f"{original_name}_clipped_{clipStart}to{int(end_ms/1000)}.mp3"

        # Save the clipped audio locally
        clipped_audio.export(clipped_audio_path, format="mp3")

        # Clean up the temporary file if downloaded
        if audio_url.startswith("http") and os.path.exists(local_audio_path):
            os.remove(local_audio_path)

        # Return the clipped audio path and its duration
        clipped_duration = (end_ms - start_ms) / 1000
        return clipped_audio_path, clipped_duration

    except Exception as e:
        logging.error(f"Failed to clip audio: {e}")
        return None, 0
    
def detect_vocals(audio_path):
    """
    Detect when vocals/lyrics start in a song using librosa
    Returns time in milliseconds
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Use librosa to detect vocals (simplified approach using harmonic separation)
    y_harmonic = librosa.effects.harmonic(y)
    mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
    
    # Simple heuristic: when MFCC variance increases significantly, vocals likely begin
    mfcc_var = mfccs.var(axis=1)
    threshold = mfcc_var.mean() * 1.5
    
    # Find first frame where variance exceeds threshold (potential vocals start)
    for i, frame in enumerate(mfccs.T):
        if frame.var() > threshold:
            # Convert frame index to milliseconds
            return int((i * 512 / sr) * 1000)
    
    # If no clear vocals detected, return 0
    return 0

def merge_songs(song1_path, song2_path, output_path, transition_time, auto_align=True):
    """
    Merge two songs where song1 plays until transition_time, then transitions to song2
    If auto_align is True, tries to align song2 starting at lyrics/vocals
    
    Parameters:
    - song1_path: Path to the first song
    - song2_path: Path to the second song
    - output_path: Path to save the merged song
    - transition_time: Time in milliseconds where song1 transitions to song2
    - auto_align: Whether to automatically detect and align with vocals in song2
    """
    print(f"Loading first song: {song1_path}")
    song1 = AudioSegment.from_file(song1_path)
    
    print(f"Loading second song: {song2_path}")
    song2 = AudioSegment.from_file(song2_path)
    
    # Ensure transition time doesn't exceed song1 length
    if transition_time > len(song1):
        print(f"Warning: Transition time exceeds song1 length. Using song1 length instead.")
        transition_time = len(song1)
    
    # Take part of song1 until transition point
    song1_part = song1[:transition_time]
    
    # Determine where to start song2
    if auto_align:
        print("Detecting vocals in song2...")
        vocals_start = detect_vocals(song2_path)
        print(f"Detected vocals at {vocals_start}ms in song2")
        
        # Take part of song2 starting at the vocals
        song2_part = song2[vocals_start:]
    else:
        # If not auto-aligning, just use the beginning of song2
        song2_part = song2
    
    # Create a crossfade between the songs (1 second)
    crossfade_duration = min(1000, len(song1_part), len(song2_part))
    
    # Merge songs with crossfade
    output = song1_part.append(song2_part, crossfade=crossfade_duration)
    
    # Export merged song
    print(f"Exporting merged song to {output_path}")
    output.export(output_path, format=os.path.splitext(output_path)[1][1:])
    print("Merge complete!")