#!/usr/bin/env python3

import os


def delete_all_files(directory):
    try:
        # List all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            # Check if it is a file and delete it
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"All files in '{directory}' have been deleted.")
    except Exception as e:
        print(f"Error: {e}")

def main():
    audio_files = r"C:\Users\hhqia\Code\Misc\youtube_shorts\audio_files"
    trimmed_w_audio_videos = r"C:\Users\hhqia\Code\Misc\youtube_shorts\trimmed_w_audio"
    cropped_videos = r"C:\Users\hhqia\Code\Misc\youtube_shorts\cropped_videos"
    transcribed_videos = r"C:\Users\hhqia\Code\Misc\youtube_shorts\transcribed_videos"

    delete_all_files(audio_files)
    delete_all_files(trimmed_w_audio_videos)
    delete_all_files(cropped_videos)
    delete_all_files(transcribed_videos)


if __name__ == "__main__":
    main()
