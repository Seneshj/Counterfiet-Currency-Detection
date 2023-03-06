import cv2

# List of video filenames to process
video_files = ['10a.mp4', '10b.mp4', '10c.mp4', 'cent_10b.mp4', 'cent_50.mp4', 'cent_50b.mp4']

# Loop through each video file
for video_file in video_files:
    # Open the video file
    cap = cv2.VideoCapture(video_file)
