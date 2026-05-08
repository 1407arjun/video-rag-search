import os
import subprocess
import tempfile
import base64
import shutil
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

    def _generate_thumbnail(self, frame_file: str, width=320):
        frame = cv2.imread(frame_file)
        height = int(frame.shape[0] * (width / frame.shape[1]))
        resized = cv2.resize(frame, (width, height),
                             interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(
            '.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return base64.b64encode(buffer).decode('utf-8')

    def sample_video(self, scene_threshold=0.3) -> tuple[list[str], str]:
        temp_dir = tempfile.mkdtemp()
        output_pattern = os.path.join(temp_dir, "keyframe_%04d.jpg")
        frames = []

        try:
            command = [
                "ffmpeg", "-i", self.video_path,
                "-vf", f"select='gt(scene,{scene_threshold})'",
                "-vsync", "vfr",
                "-q:v", "2",
                output_pattern
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)

            frame_files = sorted([os.path.join(temp_dir, f)
                                 for f in os.listdir(temp_dir)])

            if not frame_files:
                command = [
                    "ffmpeg", "-i", self.video_path, "-frames:v", "1", "-q:v", "2", output_pattern
                ]
                subprocess.run(
                    command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                frame_files = sorted([os.path.join(temp_dir, f)
                                     for f in os.listdir(temp_dir)])

            for f_path in frame_files:
                with open(f_path, "rb") as image_file:
                    b64_string = base64.b64encode(
                        image_file.read()).decode("utf-8")
                    frames.append(b64_string)

            return frames, self._generate_thumbnail(frame_files[0])
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
