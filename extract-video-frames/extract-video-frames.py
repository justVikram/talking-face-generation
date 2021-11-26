from cv2 import cv2
import time
import os
from pydub import AudioSegment
import moviepy.editor as mp


def extract_audio(video_path: str, audio_path: str) -> None:
    my_clip = mp.VideoFileClip(video_path)
    my_clip.audio.write_audiofile(audio_path+"/my_result.mp3")


def chunkify_audio(audio_path: str, interval) -> None:
    audio = AudioSegment.from_mp3(audio_path)

    # Length of the audiofile in milliseconds
    n = len(audio)

    # Variable to count the number of sliced chunks
    counter = 1

    # Interval length at which to slice the audio file.
    # If length is 22 seconds, and interval is 5 seconds,
    # The chunks created will be:
    # chunk1 : 0 - 5 seconds
    # chunk2 : 5 - 10 seconds
    # chunk3 : 10 - 15 seconds
    # chunk4 : 15 - 20 seconds
    # chunk5 : 20 - 22 seconds

    # Length of audio to overlap.
    # If length is 22 seconds, and interval is 5 seconds,
    # With overlap as 1.5 seconds,
    # The chunks created will be:
    # chunk1 : 0 - 5 seconds
    # chunk2 : 3.5 - 8.5 seconds
    # chunk3 : 7 - 12 seconds
    # chunk4 : 10.5 - 15.5 seconds
    # chunk5 : 14 - 19.5 seconds
    # chunk6 : 18 - 22 seconds
    overlap = 1.5 * 1000

    # Initialize start and end seconds to 0
    start = 0
    end = 0

    # Flag to keep track of end of file.
    # When audio reaches its end, flag is set to 1 and we break
    flag = 0

    # Iterate from 0 to end of the file,
    # with increment = interval
    for i in range(0, 2 * n, interval):

        # During first iteration,
        # start is 0, end is the interval
        if i == 0:
            start = 0
            end = interval

        # All other iterations,
        # start is the previous end - overlap
        # end becomes end + interval
        else:
            start = end - overlap
            end = start + interval

        # When end becomes greater than the file length,
        # end is set to the file length
        # flag is set to 1 to indicate break.
        if end >= n:
            end = n
            flag = 1

        # Storing audio file from the defined start to end
        chunk = audio[start:end]

        # Filename / Path to store the sliced audio
        filename = 'chunk' + str(counter) + '.wav'

        # Store the sliced audio file to the defined path
        chunk.export(filename, format="wav")
        # Print information about the current chunk
        print("Processing chunk " + str(counter) + ". Start = "
              + str(start) + " end = " + str(end))

        # Increment counter for the next chunk
        counter = counter + 1


def video_to_frames(input_loc, output_loc) -> int:
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print("Number of frames: ", video_length)
    count = 0
    print("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count + 1), frame)
        count = count + 1
        # If there are no more frames left
        if count > (video_length - 1):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print("Done extracting frames.\n%d frames extracted" % count)
            print("It took %d seconds forconversion." % (time_end - time_start))
            break

    return video_length


if __name__ == "__main__":
    input_loc = '/Users/avikram/Projects/Talking-Face-Generation/extract-video-frames/data/01.mp4'
    output_loc = '/Users/avikram/Projects/Talking-Face-Generation/extract-video-frames/data/output'
    n_frames = video_to_frames(input_loc, output_loc)
    extract_audio(input_loc, output_loc)
    chunkify_audio(os.path.join(output_loc, 'my_result.mp3'), 50)
