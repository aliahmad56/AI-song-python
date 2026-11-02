from pydub import AudioSegment
import librosa
import os



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
    # Check if files exist
    if not os.path.exists(song1_path):
        raise FileNotFoundError(f"First song file not found: {song1_path}")
    if not os.path.exists(song2_path):
        raise FileNotFoundError(f"Second song file not found: {song2_path}")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
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
    output.export(output_path, format="mp3")
    print("Merge complete!")
    
    # Verify output file exists
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Failed to create output file: {output_path}")

