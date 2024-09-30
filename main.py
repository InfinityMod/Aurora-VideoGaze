import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from vidgear.gears import CamGear
import yt_dlp
import os


class YouTubeStreamProcessorHSVWithVidgear:
    """
    Fetches the YouTube live stream using yt-dlp, streams it using Vidgear, 
    and processes it to compute HSV histograms.
    """
    def __init__(self, youtube_url: str, buffer_size: int = 100, yt_dlp_path: str = None):
        """
        Initializes the YouTubeStreamProcessorHSVWithVidgear object.

        :param youtube_url: URL of the YouTube stream.
        :param buffer_size: Number of frames to average histograms over.
        :param yt_dlp_path: Optional path to the yt-dlp executable.
        """
        self.youtube_url = youtube_url
        self.buffer_size = buffer_size
        self.hist_buffer = deque(maxlen=buffer_size)
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        self._setup_plot()

        # Set custom yt-dlp path if provided
        if yt_dlp_path:
            self.set_yt_dlp_path(yt_dlp_path)

        # Get the video stream URL using yt-dlp
        self.stream_url = self.get_stream_url()

    def set_yt_dlp_path(self, yt_dlp_path: str):
        """
        Sets the path to the yt-dlp executable for extraction.

        :param yt_dlp_path: Path to the yt-dlp executable.
        """
        if os.path.exists(yt_dlp_path):
            os.environ['YTDLP_EXE'] = yt_dlp_path
        else:
            raise FileNotFoundError(f"The specified yt-dlp path does not exist: {yt_dlp_path}")

    def get_stream_url(self):
        """
        Uses yt-dlp to retrieve the best available video stream for the YouTube live stream.

        :return: Direct video stream URL.
        """
        try:
            ydl_opts = {
                'format': 'best',
                'noplaylist': True,
                'quiet': True,
                'geo_bypass': True    # Bypass geo restrictions
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(self.youtube_url, download=False)
                stream_url = next((format['url'] for format in info_dict['formats'] if format['ext'] in ['m3u8', 'mp4']), None)
                return stream_url
        except Exception as e:
            print(f"Failed to retrieve stream URL via yt-dlp: {e}")
            return None

    def _setup_plot(self):
        """
        Initializes the plot axes for displaying histograms of H, S, V channels.
        """
        self.axs[0].set_title('Hue Channel Histogram')
        self.axs[1].set_title('Saturation Channel Histogram')
        self.axs[2].set_title('Value Channel Histogram')
        for ax in self.axs:
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 2**16)

    def compute_hsv_histogram(self, frame: np.ndarray):
        """
        Computes the histogram for the H, S, V channels of a given frame.

        :param frame: The input image frame in BGR format.
        :return: Tuple of histograms for the H, S, and V channels.
        """
        # Convert the frame from BGR to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate histograms for the H, S, and V channels
        hist_h = cv2.calcHist([hsv_frame], [0], None, [256], [0, 256])  # Hue
        hist_s = cv2.calcHist([hsv_frame], [1], None, [256], [0, 256])  # Saturation
        hist_v = cv2.calcHist([hsv_frame], [2], None, [256], [0, 256])  # Value
        return hist_h, hist_s, hist_v

    def update_hist_buffer(self, hist_h: np.ndarray, hist_s: np.ndarray, hist_v: np.ndarray):
        """
        Updates the histogram buffer with the latest frame's histograms.

        :param hist_h: Hue channel histogram.
        :param hist_s: Saturation channel histogram.
        :param hist_v: Value channel histogram.
        """
        self.hist_buffer.append((hist_h, hist_s, hist_v))

    def average_histograms(self):
        """
        Averages the histograms stored in the buffer.

        :return: Tuple of averaged histograms for H, S, and V channels.
        """
        avg_hist_h = np.mean([h[0] for h in self.hist_buffer], axis=0)
        avg_hist_s = np.mean([h[1] for h in self.hist_buffer], axis=0)
        avg_hist_v = np.mean([h[2] for h in self.hist_buffer], axis=0)
        return avg_hist_h, avg_hist_s, avg_hist_v

    def display_histograms(self, avg_hist_h: np.ndarray, avg_hist_s: np.ndarray, avg_hist_v: np.ndarray):
        """
        Updates the live matplotlib plot with the averaged histograms.

        :param avg_hist_h: Averaged histogram for the hue channel.
        :param avg_hist_s: Averaged histogram for the saturation channel.
        :param avg_hist_v: Averaged histogram for the value channel.
        """
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()
        self._setup_plot()

        self.axs[0].plot(avg_hist_h, color='orange')  # Hue
        self.axs[1].plot(avg_hist_s, color='green')   # Saturation
        self.axs[2].plot(avg_hist_v, color='blue')    # Value

        plt.pause(0.01)  # Pause to update the plot

    def stream_video(self):
        """
        Opens the video stream URL using Vidgear and processes it for HSV histograms.
        """
        if not self.stream_url:
            print("Error: Unable to retrieve stream URL.")
            return

        # Use Vidgear's VideoGear to stream the video
        stream = CamGear(
            source=self.youtube_url, 
            stream_mode=True,
            logging=True
        ).start()
        plt.ion()  # Turn on interactive mode for live plot updates

        while True:
            frame = stream.read()
            if frame is None:
                print("Stream ended or frame capture failed.")
                break

            # Process the frame if successfully captured
            hist_h, hist_s, hist_v = self.compute_hsv_histogram(frame)
            self.update_hist_buffer(hist_h, hist_s, hist_v)

            if len(self.hist_buffer) == self.buffer_size:
                avg_hist_h, avg_hist_s, avg_hist_v = self.average_histograms()
                self.display_histograms(avg_hist_h, avg_hist_s, avg_hist_v)

            # Optionally display the original frame
            cv2.imshow('Stream Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        stream.stop()
        cv2.destroyAllWindows()

    def run(self):
        """
        Main entry point to process the video stream using yt-dlp and Vidgear.
        """
        self.stream_video()


# Example usage:
if __name__ == "__main__":
    youtube_url = r"https://www.youtube.com/watch?v=dJdpRO0BwA8"  # Replace with actual YouTube live stream URL
    yt_dlp_path = r"C:\Users\ge78rey\Downloads\yt-dlp.exe"  # Replace with the actual path to yt-dlp

    # Initialize the processor with YouTube URL and optional yt-dlp path
    processor = YouTubeStreamProcessorHSVWithVidgear(youtube_url, yt_dlp_path=yt_dlp_path)
    processor.run()