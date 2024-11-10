import os
import subprocess
import json
from tqdm import tqdm
import whisper
from moviepy.editor import VideoFileClip, AudioFileClip
import wave
import csv
from TTS.api import TTS

def transcribe_and_subtitle(input_dir, output_dir):
    # Load the Whisper model
    model = whisper.load_model("large")  # You can choose 'tiny', 'base', 'small', 'medium', 'large'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of MP4 and MOV files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.mov'))]

    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, f'transcribed_{video_file}')

        # Transcribe audio from the video
        print(f"Transcribing {video_file}...")
        result = model.transcribe(input_path, word_timestamps=True, fp16=False)

        # Save the transcription to an ASS file with grouped words
        ass_path = os.path.splitext(input_path)[0] + ".ass"
        generate_ass_subtitle(result, ass_path)

        # Burn subtitles into the video using FFmpeg
        print(f"Adding subtitles to {video_file}...")
        subprocess.run([
            'ffmpeg',
            '-i', input_path,
            '-vf', f"ass={ass_path}",
            '-c:a', 'copy',
            output_path
        ], check=True)

        # Optionally remove the ASS file
        os.remove(ass_path)

def generate_ass_subtitle(result, ass_path):
    """Generate an ASS subtitle file with grouped words in 2-second windows."""
    # ASS header with style definitions
    ass_header = """
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
Collisions: Normal
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,2,0,2,100,100,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(ass_path, "w", encoding="utf-8") as ass_file:
        ass_file.write(ass_header)

        # Flatten all words from all segments
        words = []
        for segment in result['segments']:
            words.extend(segment['words'])

        # Initialize grouping variables
        group_start_time = None
        group_end_time = None
        group_words = []
        max_group_duration = 1.2  # 1-second window

        for word_info in words:
            # Get timing and text for each word
            word_start = word_info['start']
            word_end = word_info['end']
            word_text = word_info['word'].replace('\n', ' ').strip()

            # Initialize group start and end times
            if group_start_time is None:
                group_start_time = word_start
                group_end_time = word_end

            # Check if adding the word exceeds the 2-second window
            if word_end - group_start_time > max_group_duration:
                # Write the current group to the ASS file
                write_ass_line(ass_file, group_start_time, group_end_time, group_words)
                # Start a new group
                group_start_time = word_start
                group_end_time = word_end
                group_words = [word_text]
            else:
                # Add word to the current group
                group_end_time = word_end
                group_words.append(word_text)

        # Write any remaining words in the last group
        if group_words:
            write_ass_line(ass_file, group_start_time, group_end_time, group_words)

def write_ass_line(ass_file, start_time, end_time, words):
    """Write a dialogue line to the ASS file with grouped words."""
    start_time_str = format_ass_timestamp(start_time)
    end_time_str = format_ass_timestamp(end_time)

    # Combine words into a single line
    text = ' '.join(words)
    # Escape special characters
    text = text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')

    # Write the dialogue line
    ass_line = f"Dialogue: 0,{start_time_str},{end_time_str},Default,,0,0,0,,{text}\n"
    ass_file.write(ass_line)

def format_ass_timestamp(seconds):
    """Convert seconds to ASS timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"

# Function to get video dimensions using ffprobe
def get_video_dimensions(filepath):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        filepath
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error getting dimensions for {filepath}")
        return None, None
    info = json.loads(result.stdout)
    width = info['streams'][0]['width']
    height = info['streams'][0]['height']
    return width, height

def crop_videos(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f'vertical_{filename}')
            command = f'ffmpeg -i "{input_path}" -vf "crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0, scale=1080:1920" -c:a copy "{output_path}"'
            subprocess.run(command, shell=True)

def trim_video_add_audio(video_path, audio_path, output_path):
    for i, vid_filename in enumerate(os.listdir(video_path)):
        for j, aud_filename in enumerate(os.listdir(audio_path)):
            if vid_filename.endswith('.mp4') and aud_filename.endswith('.wav'):
                # Get the duration of the .wav file
                with wave.open(os.path.join(audio_path, aud_filename), 'r') as audio_file:
                    frame_rate = audio_file.getframerate()
                    num_frames = audio_file.getnframes()
                    audio_duration = num_frames / float(frame_rate)
                
                # Load the video, remove audio, and trim to the audio duration
                with VideoFileClip(os.path.join(video_path, vid_filename)) as video, AudioFileClip(os.path.join(audio_path, aud_filename)) as audio:
                    video_no_audio = video.without_audio()  # Remove audio from video
                    trimmed_video = video_no_audio.subclip(0, audio_duration)

                    # Set the new audio from the .wav file
                    video_with_new_audio = trimmed_video.set_audio(audio)

                    # Write the result to the output path
                    output = os.path.join(output_path, f'trimmed_w_audio_{i}_{j}.mp4')
                    video_with_new_audio.write_videofile(output, codec="libx264", audio_codec="aac")

def create_story_audio(fun_facts, audio_files):

    with open(fun_facts, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Skip header
        next(reader, None)
        for i, row in enumerate(reader):
            fun_fact = str(row[0])
            # Load the multi-speaker VITS models
            tts = TTS(model_name="tts_models/en/vctk/vits")
            speaker = "p230"
            tts.tts_to_file(text=fun_fact, speaker=speaker, speed=1, pitch=1.2, file_path=os.path.join(audio_files, f'story_audio_{i}.wav'))

    

if __name__ == "__main__":
    fun_facts = "/Users/howardqian/Desktop/Youtube_Shorts/fun_facts.csv"
    raw_videos = '/Users/howardqian/Desktop/Youtube_Shorts/raw_videos'
    audio_files = '/Users/howardqian/Desktop/Youtube_Shorts/audio_files'
    trimmed_w_audio_videos = '/Users/howardqian/Desktop/Youtube_Shorts/trimmed_w_audio'
    cropped_videos = '/Users/howardqian/Desktop/Youtube_Shorts/cropped_videos'
    transcribed_videos = '/Users/howardqian/Desktop/Youtube_Shorts/transcribed_videos'

    print("CREATING AUDIO")
    create_story_audio(fun_facts, audio_files)

    print("TRIMMING RAW VIDEO AND ADDING AUDIO")
    trim_video_add_audio(raw_videos, audio_files, trimmed_w_audio_videos)

    print("CROPPING VIDEO")
    crop_videos(trimmed_w_audio_videos, cropped_videos)

    print("TRANSCRIBING AND ADDING SUBTITLES")
    transcribe_and_subtitle(cropped_videos, transcribed_videos)



