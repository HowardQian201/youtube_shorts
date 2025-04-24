#!/usr/bin/env python3

import os
import subprocess
import json
from tqdm import tqdm
import torch
import whisper
from moviepy.editor import VideoFileClip, AudioFileClip
import wave
from google_auth_oauthlib.flow import InstalledAppFlow
import googleapiclient.discovery
from api_keys import OPENAI_API_KEY, ELEVEN_LABS_KEYS
from openai import OpenAI
import random
from pydub import AudioSegment, effects
import wave
import io
import pickle
from google.auth.transport.requests import Request
from elevenlabs.client import ElevenLabs
from moviepy.video.fx.all import speedx
from elevenlabs.core.api_error import ApiError
import time
import delete_files as delete

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "large"           # try "large-v3" or "turbo" if VRAM <10 GB

def transcribe_and_subtitle(input_dir, output_dir):
    # Load the Whisper model
    model = whisper.load_model(MODEL_NAME, device=DEVICE)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of MP4 and MOV files in the input directory
    video_files = [
        f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".mov"))
    ]

    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(
            output_dir, f"{video_file[0]}_transcribed_{video_file}"
        )

        # Transcribe audio from the video
        print(f"Transcribing {video_file}...")
        result = model.transcribe(input_path, word_timestamps=True)


        # Save the transcription to an ASS file with grouped words
        ass_path = os.path.splitext(input_path)[0] + ".ass"
        safe = ass_path.replace('\\', '/')         # forward slashes are fine on Win
        safe = safe.replace(':', r'\:')             # <-- escape the colon!
        vf   = f"subtitles='{safe}'"      # quote so the drive-colon is literal

        generate_ass_subtitle(result, ass_path)

        # Burn subtitles into the video using FFmpeg
        print(f"Adding subtitles to {video_file}...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-vf",
                vf,
                "-c:a",
                "copy",
                output_path,
            ],
            check=True,
        )

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
Style: Default,Arial,72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,2,0,2,100,100,450,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(ass_path, "w", encoding="utf-8") as ass_file:
        ass_file.write(ass_header)

        # Flatten all words from all segments
        words = []
        for segment in result['segments']:
            words.extend(segment["words"])

        # Initialize grouping variables
        group_start_time = None
        group_end_time = None
        group_words = []
        max_group_duration = 0.75  # 0.75-second window

        for word_info in words:
            # Get timing and text for each word
            word_start = word_info["start"]
            word_end = word_info["end"]
            word_text = word_info["word"].replace("\n", " ").strip()

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
    """Write a dialogue line to the ASS file with grouped words and bounce effect."""
    start_time_str = format_ass_timestamp(start_time)
    end_time_str = format_ass_timestamp(end_time)

    # Combine words into a single line
    text = " ".join(words)
    # Escape special characters
    text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

    # Define the bounce effect using ASS override tags
    bounce_effect = r"{\shad5\bord5\fscx80\fscy80\t(0,50,\fscx100\fscy100)}" + text

    # Write the dialogue line with the bounce effect
    ass_line = (
        f"Dialogue: 0,{start_time_str},{end_time_str},Default,,0,0,0,,{bounce_effect}\n"
    )
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
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        filepath,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print(f"Error getting dimensions for {filepath}")
        return None, None
    info = json.loads(result.stdout)
    width = info["streams"][0]["width"]
    height = info["streams"][0]["height"]
    return width, height


def crop_videos(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(
                output_folder, f"{filename[0]}_vertical_{filename}"
            )
            command = f'ffmpeg -i "{input_path}" -vf "crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0, scale=1080:1920" -c:v libx264 -crf 18 -preset slow -c:a copy "{output_path}"'
            subprocess.run(command, shell=True)


def trim_video_add_audio(video_path, audio_path, output_path):
    for i, aud_filename in enumerate(os.listdir(audio_path)):
        visible_vid_files = [
            f
            for f in os.listdir(video_path)
            if not f.startswith(".") and os.path.isfile(os.path.join(video_path, f))
        ]
        vid_filename = random.choice(visible_vid_files)
        if vid_filename.endswith(".mp4") and aud_filename.endswith(".wav"):
            # Get the duration of the .wav file
            with wave.open(os.path.join(audio_path, aud_filename), "r") as audio_file:
                frame_rate = audio_file.getframerate()
                num_frames = audio_file.getnframes()
                audio_duration = num_frames / float(frame_rate)

            # Load the video, remove audio, and trim to the audio duration
            with VideoFileClip(
                os.path.join(video_path, vid_filename)
            ) as video, AudioFileClip(os.path.join(audio_path, aud_filename)) as audio:
                video_duration = video.duration
                # Set speed-up factor (e.g., 1.3x speed)
                speed_up_factor = 1.3
                # Calculate adjusted video duration after speed-up
                adjusted_video_duration = video_duration / speed_up_factor

                max_start_time = max(0, adjusted_video_duration - audio_duration)
                start_time = random.uniform(0, max_start_time)
                end_time = start_time + audio_duration

                video_no_audio = video.without_audio()  # Remove audio from video
                trimmed_video = video_no_audio.subclip(
                    start_time * speed_up_factor, end_time * speed_up_factor
                )

                # Apply speed-up effect
                sped_up_video = speedx(trimmed_video, factor=speed_up_factor)

                # Set the new audio from the .wav file
                video_with_new_audio = sped_up_video.set_audio(audio)

                # Write the result to the output path
                output = os.path.join(
                    output_path, f"{aud_filename[0]}_trimmed_w_audio.mp4"
                )
                video_with_new_audio.write_videofile(
                    output,
                    codec="h264_nvenc",
                    audio_codec="aac",
                    ffmpeg_params=[
                        "-pix_fmt", "yuv420p",   # <-- keeps colours correct everywhere
                        "-preset",  "p4",
                        "-cq",      "19",
                    ],
                )


def create_story_audio(facts, audio_files):

    for i, fact in enumerate(facts):
        audio = None
        try_num = -1
        while audio is None:
            try_num += 1
            print(f"Generating audio {i} try number {try_num}.")
            try:
                # Initialize the client with your API key
                client = ElevenLabs(api_key=ELEVEN_LABS_KEYS[try_num])
                # Generate speech
                audio = client.generate(
                    text=fact,
                    voice="Brian",  # Replace with your desired voice, Adam is the typical one, Michael is old but deep, Brian fits Demarcus, Liam may be better than Adam
                    model="eleven_turbo_v2_5",  # Replace with your desired model
                    stream=True,
                )
                # Combine the generator's output into bytes
                audio = b"".join(audio)
            except ApiError as e:
                print(f"API Error: {e}")
                audio = None

        # Load audio data into an AudioSegment
        audio = AudioSegment.from_file(io.BytesIO(audio), format="mp3")
        # Make louder by x decibles
        audio = audio.apply_gain(0)
        # Increase the speed by a factor (e.g., 1.1 times faster)
        audio = effects.speedup(audio, playback_speed=1.1)

        # Save audio as a .wav file
        output_file = f"{audio_files}/{i}_output_audio.wav"
        audio.export(output_file, format="wav")


def upload_to_youtube(video_dir, titles, descriptions):
    # Define constants
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    API_SERVICE_NAME = "youtube"
    API_VERSION = "v3"
    CLIENT_SECRET = r"C:\Users\hhqia\Code\Misc\youtube_shorts\client_secret_38628020910-hn37vtut0p1k8jnb65s2q5gpqes8335a.apps.googleusercontent.com.json"
    CREDENTIALS_FILE = "youtube_credentials.pkl"

    def authenticate():
        credentials = None
        SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
        CLIENT_SECRET = r"C:\Users\hhqia\Code\Misc\youtube_shorts\client_secret_38628020910-hn37vtut0p1k8jnb65s2q5gpqes8335a.apps.googleusercontent.com.json"
        CREDENTIALS_FILE = "youtube_credentials.pkl"

        # Check if credentials file exists
        if os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE, "rb") as token:
                credentials = pickle.load(token)

        # Refresh credentials if expired
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
                credentials = flow.run_local_server(port=0)

            # Save the new credentials
            with open(CREDENTIALS_FILE, "wb") as token:
                pickle.dump(credentials, token)

        return credentials

    def upload_video(
        credentials,
        file_path,
        title,
        description,
        category_id="22",
        privacy_status="public",
    ):

        youtube = googleapiclient.discovery.build(
            API_SERVICE_NAME, API_VERSION, credentials=credentials
        )

        request_body = {
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy_status,
            },
        }

        # Upload video file
        request = youtube.videos().insert(
            part="snippet,status",
            body=request_body,
            media_body=googleapiclient.http.MediaFileUpload(
                file_path, chunksize=-1, resumable=True
            ),
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Upload progress: {int(status.progress() * 100)}%")

        print("Video uploaded successfully!")
        return response

    # Set parameters for the video
    category_id = "27"  # 22 is for 'People & Blogs' 27 is for education
    privacy_status = "public"  # Options: "public", "private", or "unlisted"'
    credentials = authenticate()
    for filename in os.listdir(video_dir):
        if filename.endswith(".mp4"):
            file_path = os.path.join(video_dir, filename)
            upload_video(
                credentials=credentials,
                file_path=file_path,
                title=titles[int(filename[0])],
                description=descriptions[int(filename[0])],
                category_id=category_id,
                privacy_status=privacy_status,
            )


def create_script():
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    text = input("Please input what type of video you would like:\n")

    title_prompt = f"I would like to create 2 viral reddit-style story youtube videos \
        about {text}. Please make 2 great and specific titles using popular proper nouns for such a video \
        Also, give it a detailed and specific video topic so the viewers know exactly what the \
        videos are about. Please include proper nouns that normal people know about \
        for places and things in the titles. \
        I do not want a title that suggests there is a list in the video. \
        Please keep each title around 60 characters. \
        Please separate each title with a new line. \
        Please do not include any quotations or text other than the title within your output. I repeat, \
        Do not include beginning or ending quotations when outputting the fact."

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": title_prompt},
        ],
    )

    # Parse the response into lines
    titles = completion.choices[0].message.content.strip().split("\n")
    titles = [title.strip() for title in titles if title.strip() != ""]
    print("Video titles: ", titles)
    print("\n")

    script_prompt = f"I would like you to create a 2 fun and interesting Reddit \
        style stories that are realistic but kind of absurd to viewers about \
        2 different topics. Topic 1: {titles[0]}. Topic 2: {titles[1]}. \
        Each reddit style story should be unrelated to one another, roughly 500 characters \
        long (1000 characters total between the 2 stories) and should be fun, shocking, interesting, realistic, \
        and explained well. Please include proper nouns for places and things. \
        Please also make the story realistic and interesting to capture the attentions of listeners, as if it were posted on Reddit. \
        Please state the each story separated by a new line without any quotations or other text. I repeat, \
        DO NOT include beginning or ending quotations or any other text such as the title of the scripts when outputting each story. \
        DO NOT include any introductory words or sentences - just get right into the story. \
        Simply state the 2 scripts separated by a new line."

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": script_prompt},
        ],
    )

    facts = completion.choices[0].message.content.strip().split("\n")
    facts = [fact.strip() for fact in facts if fact.strip() != ""]
    print("Stories: ", facts)
    print("\n")

    description_prompt = f"Please create 2 descriptions for 2 youtube videos \
        based on each story I mention below. Please make this description as long as \
        possible (under 1000 characters) so that each video can easily \
        pop up in search results. Make sure to include many keywords \
        that are related to the video topics. Please output the 2 descriptions separated by '---'. \
        Please include hashtags in each descrition as well. \
        Do not include quotations or any other text in your output. \
        Video 1: {facts[0]}. Video 2: {facts[1]}."

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": description_prompt},
        ],
    )

    descriptions = completion.choices[0].message.content.strip().split("---")
    descriptions = [
        description.strip() for description in descriptions if description.strip() != ""
    ]
    print("Descriptions: ", descriptions)
    print("\n")

    return facts, titles, descriptions


if __name__ == "__main__":
    raw_videos = r"C:\Users\hhqia\Code\Misc\youtube_shorts\raw_videos"
    audio_files = r"C:\Users\hhqia\Code\Misc\youtube_shorts\audio_files"
    trimmed_w_audio_videos = r"C:\Users\hhqia\Code\Misc\youtube_shorts\trimmed_w_audio"
    cropped_videos = r"C:\Users\hhqia\Code\Misc\youtube_shorts\cropped_videos"
    transcribed_videos = r"C:\Users\hhqia\Code\Misc\youtube_shorts\transcribed_videos"

    start_time = time.time()

    print("CREATING SCRIPT")
    facts, titles, descriptions = create_script()

    print("CREATING AUDIO")
    create_story_audio(facts, audio_files)

    print("TRIMMING RAW VIDEO AND ADDING AUDIO")
    trim_video_add_audio(raw_videos, audio_files, trimmed_w_audio_videos)

    print("CROPPING VIDEO")
    crop_videos(trimmed_w_audio_videos, cropped_videos)

    print("TRANSCRIBING AND ADDING SUBTITLES")
    transcribe_and_subtitle(cropped_videos, transcribed_videos)

    print("UPLOADING TO YOUTUBE")
    upload_to_youtube(transcribed_videos, titles, descriptions)

    print("Deleting all files")
    delete.main()

    print("Finished in ", time.time() - start_time, "seconds.")
