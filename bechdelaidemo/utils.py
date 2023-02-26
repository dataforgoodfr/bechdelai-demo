from pytube import YouTube
import moviepy.editor as mp


def download_youtube_video(link: str, filename: str, caption_language: str = "en") -> None:
    """Download a youtube video with captions given an id

    Parameters
    ----------
    link : str
        Youtube video link
    filename : str
        File name to save the video and the caption
    caption_language : str
        Language caption to download

    Raises
    ------
    TypeError
        url must be a string
    ValueError
        url must start with 'http'
    """
    try:
        yt = YouTube(link)
    except:
        print("Connection Error")
        return

    filename = filename if filename.endswith(".mp4") else filename + ".mp4"

    try:
        (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        ).download(filename=filename)

    except Exception as e:
        print("Could not download the video!", e)

    try:
        captions = {
            k: v
            for k, v in yt.captions.lang_code_index.items()
            if caption_language in k
        }
        for lang, caption in captions.items():
            caption.download(title=f"caption_{lang}", srt=False)
    except Exception as e:
        print("Could not download the caption!", e)
    print("Task Completed!")


def download_youtube_audio(link:str,filename:str = "audio.mp3") -> str:
    yt = YouTube(link)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename=filename)
    return filename


def import_as_clip(path_to_video: str) -> mp.VideoFileClip:
    """Imports a video file as a VideoFileClip object.
    
    Parameters:
        path_to_video (str): Path to a video file.
    
    Returns:
        mp.VideoFileClip: VideoFileClip object.
    """
    return mp.VideoFileClip(path_to_video)

def extract_audio_from_movie(file: str, extension: str = '.wav') -> None:
    """Extract the audio from a film and save it to a file.
    
    The audio is saved in the same directory as the film.
    
    Parameters:
        file (str): The name of the film file to extract the audio from.
        extension (str): The file extension of the audio file to save (default is ".wav").
    """
    clip = import_as_clip(file)
    filename = file.split(sep='.')[0] + extension
    clip.audio.write_audiofile(filename)
    return filename
