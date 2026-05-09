import subprocess
import base64
import numpy as np
import cv2


class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_audio(self) -> str:
        audio_path = self.video_path + ".mp3"
        command = [
            "ffmpeg", "-i", self.video_path,
            "-vn", "-acodec", "libmp3lame", "-y", audio_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
        return audio_path

    def _generate_thumbnail(self, image_b64: str, width=320):
        frame = cv2.imdecode(np.frombuffer(
            base64.b64decode(image_b64), np.uint8), cv2.IMREAD_COLOR)
        height = int(frame.shape[0] * (width / frame.shape[1]))
        resized = cv2.resize(frame, (width, height),
                             interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(
            '.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return base64.b64encode(buffer).decode('utf-8')

    def _bgr_to_base64(self, bgr_array) -> str:
        success, buffer = cv2.imencode(
            '.jpg', bgr_array, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            return base64.b64encode(buffer).decode("utf-8")
        return None

    def sample_video(self, scene_threshold=0.1) -> tuple[list[str], str]:
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", self.video_path
        ]
        width, height = map(int, subprocess.check_output(
            probe_cmd).decode().split(','))
        frame_size = width * height * 3

        cmd = [
            "ffmpeg", "-i", self.video_path,
            "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

        frames_b64 = []
        prev_frame_gray = None
        first_frame_b64 = None
        frame_count = 0

        try:
            while True:
                frame = process.stdout.read(frame_size)
                if not frame:
                    break

                frame_bgr = np.frombuffer(
                    frame, dtype=np.uint8).reshape((height, width, 3))

                # Convert to grayscale for scene change detection
                curr_frame_gray = cv2.cvtColor(
                    frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)

                if frame_count == 0:
                    first_frame_b64 = self._bgr_to_base64(frame_bgr)

                if prev_frame_gray is not None:
                    # Sum of Absolute Differences (SAD)
                    diff = np.abs(curr_frame_gray - prev_frame_gray)
                    score = np.mean(diff) / 255.0

                    if score > scene_threshold:
                        frames_b64.append(self._bgr_to_base64(frame_bgr))

                prev_frame_gray = curr_frame_gray
                frame_count += 1
        finally:
            process.terminate()

        if len(frames_b64) == 0:
            if first_frame_b64 is not None:
                frames_b64.append(first_frame_b64)
            else:
                raise ValueError(
                    "Could not extract any frames from the video.")

        return frames_b64, self._generate_thumbnail(frames_b64[0])
