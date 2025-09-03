# Standard Library Imports
import asyncio
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from datetime import datetime
from enum import IntEnum, auto
from importlib import import_module
from typing import List, Tuple, Optional

# Third-party Imports
import filetype
import moviepy.editor as mp
import numpy as np
from dotenv import load_dotenv
from instagrapi import Client
from instagrapi.exceptions import (
    LoginRequired,
    TwoFactorRequired,
    ChallengeRequired,
    BadPassword,
)
from moviepy.video.fx import all as vfx
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import sobel
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InputMediaPhoto, InputMediaVideo
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, ContextTypes
import nest_asyncio


class States(IntEnum):
    AUTH_2FA = auto()
    AUTH_SMS = auto()
    MEDIA_TYPE = auto()
    RECEIVE_MEDIA = auto()
    CONFIRM = auto()
    
    # Image Watermark (Step 10)
    ASK_IMAGE_WATERMARK = auto()
    RECEIVE_IMAGE_WATERMARK = auto()
    CHOOSE_IMG_WATERMARK_POSITION = auto()
    CHOOSE_IMG_WATERMARK_SCALE = auto()
    CHOOSE_IMG_WATERMARK_OPACITY = auto()
    CONFIRM_IMG_WATERMARK = auto()
    
    # Text Watermark (Step 11)
    ASK_TEXT_WATERMARK = auto()
    RECEIVE_TEXT = auto()
    CHOOSE_FONT = auto()
    CHOOSE_FONT_SIZE = auto()
    CHOOSE_COLOR = auto()
    CHOOSE_TEXT_POSITION = auto()
    CONFIRM_TEXT_WATERMARK = auto()
    
    # Music (Step 12)
    ASK_ADD_MUSIC = auto()
    RECEIVE_MUSIC = auto()
    RECEIVE_MUSIC_START_TIME = auto()
    CONFIRM_MUSIC = auto()
    
    # Combine (Step 13)
    CONFIRM_COMBINED_MEDIA = auto()

    # Final Processing (Steps 14 & 15)
    CONFIRM_FINAL_MEDIA = auto()

    # Video Effects (Step 16)
    ASK_VIDEO_EFFECTS = auto()
    CHOOSE_EFFECTS = auto()
    CONFIRM_EFFECTS = auto()
    
    # Finalize
    CAPTION = auto()


class FileValidator:
    """
    Validates files based on their type and extension as per Step 9 of the Holy Book.
    """
    IMAGE_EXTENSIONS: List[str] = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    VIDEO_EXTENSIONS: List[str] = ['.mp4', '.avi', '.flv', '.webm', '.mov', '.mkv', '.wmv']
    GIF_EXTENSIONS: List[str] = ['.gif']

    @classmethod
    def validate(cls, file_path: str) -> str:
        """
        Validates a single file to ensure it is a supported image, video, or gif.

        Args:
            file_path (str): The path to the file to validate.

        Returns:
            str: The type of the file ('image', 'video', 'gif').

        Raises:
            ValueError: If the file type is not supported or the file does not exist.
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found at path: {file_path}")

        # Primary validation using file extension as per Holy Book
        ext = os.path.splitext(file_path)[1].lower()

        if ext in cls.IMAGE_EXTENSIONS:
            logging.info(f"Validated {os.path.basename(file_path)} as 'image' based on extension.")
            return 'image'
        
        if ext in cls.VIDEO_EXTENSIONS:
            logging.info(f"Validated {os.path.basename(file_path)} as 'video' based on extension.")
            return 'video'
            
        if ext in cls.GIF_EXTENSIONS:
            logging.info(f"Validated {os.path.basename(file_path)} as 'gif' based on extension.")
            return 'gif'

        # Fallback to filetype library if extension is not recognized
        logging.warning(f"Extension '{ext}' not in known lists for {os.path.basename(file_path)}. Guessing with filetype library.")
        try:
            kind = filetype.guess(file_path)
            if kind:
                if kind.mime.startswith('image/gif'):
                    return 'gif'
                if kind.mime.startswith('image/'):
                    return 'image'
                if kind.mime.startswith('video/'):
                    return 'video'
        except Exception as e:
            logging.error(f"Could not use filetype library to guess type for {os.path.basename(file_path)}: {e}")

        # If all else fails, reject the file
        raise ValueError(f"Unsupported file type for file: {os.path.basename(file_path)}")


class MusicAdder:
    """
    Handles audio processing, specifically trimming audio to match video duration.
    """

    @staticmethod
    def _parse_time(time_str: str) -> float:
        """Converts MM:SS format to seconds."""
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return float(minutes * 60 + seconds)
        except ValueError:
            logging.error(f"Invalid time format for '{time_str}'. Must be MM:SS.")
            raise ValueError("Invalid time format. Please use MM:SS.")

    @staticmethod
    def trim_audio(audio_path: str, video_duration: float, start_time_str: str, output_path: str) -> None:
        """
        Trims an audio file to match the video's duration, starting from a specific time.

        Args:
            audio_path (str): Path to the input audio file.
            video_duration (float): Duration of the video in seconds.
            start_time_str (str): The start time for the audio in "MM:SS" format.
            output_path (str): Path to save the trimmed audio file.
        
        Raises:
            ValueError: If the start time is invalid or longer than the audio duration.
        """
        audio_clip = None
        try:
            start_time_sec = MusicAdder._parse_time(start_time_str)
            
            logging.info(f"Trimming audio '{audio_path}' to {video_duration}s, starting at {start_time_sec}s.")
            
            audio_clip = mp.AudioFileClip(audio_path)
            
            if start_time_sec >= audio_clip.duration:
                raise ValueError("The requested start time is after the audio clip ends.")

            # Trim the audio clip
            end_time_sec = min(start_time_sec + video_duration, audio_clip.duration)
            trimmed_clip = audio_clip.subclip(start_time_sec, end_time_sec)
            
            # If the trimmed audio is shorter than the video, it will just be that length.
            # The final combination step will handle looping or silence if needed,
            # but for now, we just provide the trimmed segment.
            
            trimmed_clip.write_audiofile(output_path, codec='mp3')
            logging.info(f"Trimmed audio saved to '{output_path}'.")

        finally:
            if audio_clip:
                audio_clip.close()
            if 'trimmed_clip' in locals() and trimmed_clip:
                trimmed_clip.close()


class EffectsEngine:
    """
    A class to apply various video effects to a video clip.
    It uses a dictionary-based approach to map effect names to their methods.
    """
    def _get_clean_clip(self, clip: mp.VideoClip) -> mp.VideoClip:
        """
        "Cleans" a clip by writing it to a temporary file and reading it back.
        This standardizes the clip's properties and can prevent codec/metadata issues.
        """
        temp_filename = f"temp_{uuid.uuid4()}.mp4"
        try:
            clip.write_videofile(temp_filename, codec='libx264', audio_codec='aac')
            clean_clip = mp.VideoFileClip(temp_filename)
            # Crucially, we need to carry over the original audio if it exists
            clean_clip.audio = clip.audio
            return clean_clip
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def __init__(self):
        """
        Initializes the EffectsEngine and the mapping of effect names to methods.
        """
        self.effects_map = {
            # Existing and Renamed Effects
            'Black & White': self.apply_black_and_white,
            'Fade In/Out': self.apply_fade_in_out,
            'Pixelated Effect': self.apply_pixelated,
            'Glitch': self.apply_glitch,
            'Neon Glow': self.apply_neon_glow,
            'VHS Look': self.apply_vhs_look,

            # New Effects from User List
            'Color Saturation': self.apply_color_saturation,
            'Contrast / Brightness': self.apply_contrast_brightness,
            'Chromatic Aberration': self.apply_chromatic_aberration,
            'Invert Colors': self.apply_invert_colors,
            'Speed Control': self.apply_speed_control,
            'Rotate': self.apply_rotate,
            'Film Grain': self.apply_film_grain,
            'Rolling Shutter': self.apply_rolling_shutter,
            'Cartoon / Painterly': self.apply_cartoon_painterly,
            'Vignette': self.apply_vignette,
        }

    def apply_effects_in_sequence(self, video_path: str, effects: list[str], output_path: str) -> str:
        """
        Applies a list of effects to a video in the specified order using a dictionary lookup.
        """
        clip = mp.VideoFileClip(video_path)
        
        for effect_name in effects:
            if effect_name in self.effects_map:
                # Look up the method in the dictionary and call it
                clip = self.effects_map[effect_name](clip)
            else:
                print(f"Warning: Effect '{effect_name}' not found.")

        clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        clip.close()
        return output_path

    # --- Effect Implementations (Placeholders and Existing) ---

    def apply_black_and_white(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Applies a black and white effect."""
        return clip.fx(vfx.blackwhite)

    def apply_color_saturation(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Applies color saturation effect."""
        # A factor of 2 doubles the saturation.
        return clip.fx(vfx.colorx, 2)

    def apply_contrast_brightness(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Adjusts contrast and brightness."""
        # Adds a moderate contrast boost.
        return clip.fx(vfx.lum_contrast, contrast=0.5)
        
    def apply_chromatic_aberration(self, clip: mp.VideoClip, shift: int = 5) -> mp.VideoClip:
        """Applies a chromatic aberration (RGB split) effect."""
        def effect(frame):
            # Create shifted versions of the R, G, B channels
            r = frame[:, :, 0]
            g = frame[:, :, 1]
            b = frame[:, :, 2]
            # Shift R to the left, B to the right
            r_shifted = np.roll(r, -shift, axis=1)
            b_shifted = np.roll(b, shift, axis=1)
            # Recombine the channels
            return np.stack([r_shifted, g, b_shifted], axis=-1).astype('uint8')
        return clip.fl_image(effect)

    def apply_pixelated(self, clip: mp.VideoClip, pixel_size: int = 10) -> mp.VideoClip:
        """Applies a pixelated effect by resizing down and then up."""
        return clip.fx(vfx.resize, 1/pixel_size).fx(vfx.resize, pixel_size)

    def apply_invert_colors(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Inverts the colors of the video."""
        return clip.fx(vfx.invert_colors)

    def apply_speed_control(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Changes the speed of the video (1.5x)."""
        return clip.fx(vfx.speedx, 1.5)

    def apply_rotate(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Rotates the video 90 degrees clockwise."""
        # Moviepy rotates counter-clockwise, so we use a negative angle
        return clip.fx(vfx.rotate, -90)

    def apply_vhs_look(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Applies a composite VHS tape look."""
        # 1. Lower saturation
        saturated_clip = clip.fx(vfx.colorx, 0.8)
        
        # 2. Add horizontal line noise and slight color shift
        def vhs_effect(frame):
            h, w, _ = frame.shape
            # Add horizontal lines
            lines = np.random.randint(0, h, size=h//20)
            frame[lines, :, :] //= 2 # Darken lines
            # Slight color shift
            b = frame[:, :, 2]
            b_shifted = np.roll(b, 2, axis=1)
            frame[:, :, 2] = b_shifted
            return frame
            
        processed_clip = saturated_clip.fl_image(vhs_effect)
        # 3. Add a subtle glitch
        return self.apply_glitch(processed_clip)

    def apply_film_grain(self, clip: mp.VideoClip, strength: float = 0.1) -> mp.VideoClip:
        """Adds film grain noise to each frame."""
        def effect(frame):
            # Generate noise with the same shape as the frame
            # Strength controls the intensity of the grain
            noise = np.random.randint(-25, 25, frame.shape) * strength
            # Add noise to the frame and clip values to stay within 0-255
            return np.clip(frame + noise, 0, 255).astype('uint8')
        return clip.fl_image(effect)

    def apply_glitch(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Applies an approximate digital glitch effect."""
        def effect(get_frame, t):
            frame = get_frame(t).copy() # Work on a copy to avoid modifying the original
            # 10% chance of a glitch appearing on any given frame
            if random.random() < 0.1:
                h, w, _ = frame.shape
                # Glitch a random horizontal strip
                glitch_height = h // 20
                if glitch_height == 0: glitch_height = 1
                
                y = random.randint(0, h - glitch_height)
                strip = frame[y:y+glitch_height, :, :]
                # Displace it horizontally
                displacement = random.randint(-w//4, w//4)
                strip = np.roll(strip, displacement, axis=1)
                # Zero out the part of the strip that was rolled over
                if displacement > 0:
                    strip[:, :displacement] = 0
                else:
                    strip[:, displacement:] = 0
                frame[y:y+glitch_height, :, :] = strip.astype('uint8')
            return frame.astype('uint8')
        return clip.fl(effect)

    def apply_rolling_shutter(self, clip: mp.VideoClip, intensity: int = 10, freq: float = 5) -> mp.VideoClip:
        """Applies a rolling shutter wobble effect."""
        def effect(get_frame, t):
            frame = get_frame(t)
            h, w, _ = frame.shape
            # Calculate a sinusoidal shift for each row
            shift = (intensity * np.sin(2 * np.pi * (freq * t + (np.arange(h) / h)))).astype(int)
            # Apply the shift to each row
            # Create an array of column indices
            cols = np.arange(w)
            # Repeat for each row and add the shift
            shifted_cols = cols[np.newaxis, :] + shift[:, np.newaxis]
            # Clip the indices to be within the frame width
            shifted_cols = np.clip(shifted_cols, 0, w - 1)
            # Use advanced indexing to create the wobbled frame
            return frame[np.arange(h)[:, np.newaxis], shifted_cols]
        return clip.fl(effect)

    def apply_neon_glow(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Applies an approximate neon edge effect."""
        def effect(frame):
            # Convert to grayscale for edge detection
            gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
            # Sobel edge detection to find edges
            sx = sobel(gray, axis=0, mode='constant')
            sy = sobel(gray, axis=1, mode='constant')
            edges = np.hypot(sx, sy)
            # Normalize and scale the edges
            edges = (edges / np.max(edges) * 255)
            # Create a neon color (e.g., cyan) and apply it
            neon_color = np.array([0, 255, 255])
            neon_frame = np.zeros_like(frame)
            neon_frame[edges > 50] = neon_color # Threshold for strong edges
            return neon_frame
        return clip.fl_image(effect)

    def apply_cartoon_painterly(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Applies a simplified cartoon/painterly effect using median filter and posterization."""
        def effect(frame):
            # Convert frame to PIL Image
            img = Image.fromarray(frame)
            # Apply a median filter to smooth textures and create a "smudged" look
            img = img.filter(ImageFilter.MedianFilter(size=5))
            # Posterize the image to reduce the color palette, enhancing the cartoon feel
            # The number (3) indicates the number of bits to keep for each channel
            img = img.quantize(colors=64).convert('RGB')
            return np.array(img)
        return clip.fl_image(effect)

    def apply_vignette(self, clip: mp.VideoClip, strength: float = 0.4) -> mp.VideoClip:
        """Applies a vignette (darkened edges) effect."""
        w, h = clip.size
        # Create a radial gradient mask
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        # Normalize the distance
        max_dist = np.sqrt(center_x**2 + center_y**2)
        radial_grad = dist_from_center / max_dist
        # Create the vignette mask, strength controls the darkness
        vignette_mask = 1 - (radial_grad**2) * strength
        
        def effect(frame):
            # Apply the mask to each color channel
            return (frame * vignette_mask[:, :, np.newaxis]).astype('uint8')
            
        return clip.fl_image(effect)

    def apply_fade_in_out(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Applies a 1-second fade-in and fade-out."""
        return clip.fx(vfx.fadein, 1).fx(vfx.fadeout, 1)


class AuthManager:
    """
    Manages Instagram authentication, including session handling and 2FA.
    """
    SESSION_FILE = "ig_session.json"

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.client = Client()
        self.login_status = "UNKNOWN"
        self.login_error_message = ""

    def login(self, verification_code: str = None, two_factor_code: str = None) -> tuple[bool, str]:
        """
        Handles the complete Instagram login flow.

        Args:
            verification_code (str, optional): The SMS or email verification code.
            two_factor_code (str, optional): The 2FA (TOTP) code.

        Returns:
            A tuple containing:
            - bool: True for successful login, False otherwise.
            - str: A status message ('SUCCESS', '2FA_REQUIRED', 'SMS_REQUIRED', 'FAILURE').
        """
        # If already logged in, no need to do it again.
        if self.client.user_id:
            return True, "SUCCESS"

        # Step 6.1 & 6.2: Check for and try to use a session file.
        if os.path.exists(self.SESSION_FILE):
            logging.info(f"Session file '{self.SESSION_FILE}' found. Attempting to log in.")
            try:
                self.client.load_settings(self.SESSION_FILE)
                self.client.login(self.username, self.password)
                self.client.get_timeline_feed() # Check if the session is valid
                logging.info("Login successful using session file.")
                self.login_status = "SUCCESS"
                return True, self.login_status
            except (LoginRequired, BadPassword, Exception) as e:
                logging.warning(f"Session file is invalid or expired, will perform a fresh login. Reason: {e}")
                # Delete the invalid session file
                os.remove(self.SESSION_FILE)
        
        # Step 6.3: Fresh login using username and password
        logging.info("No valid session found. Attempting a fresh login.")
        try:
            if two_factor_code:
                # Handle 2FA login
                logging.info("Attempting login with 2FA code.")
                self.client.login(self.username, self.password, verification_code=two_factor_code)
            elif verification_code:
                # Handle SMS/email challenge code
                logging.info("Attempting login with verification code.")
                self.client.challenge_code_login(verification_code)
            else:
                # Standard login
                self.client.login(self.username, self.password)

        except TwoFactorRequired:
            self.login_status = "2FA_REQUIRED"
            logging.info("2FA code is required.")
            return False, self.login_status
        
        except ChallengeRequired:
            # This exception means a verification code (SMS/email) is needed.
            # The client state is now waiting for the code. We need to inform the handler.
            logging.info("Challenge code (SMS/Email) is required.")
            self.login_status = "SMS_REQUIRED"
            return False, self.login_status
            
        except (BadPassword, LoginRequired) as e:
            self.login_status = "FAILURE"
            self.login_error_message = f"Login failed: Incorrect username or password. Details: {e}"
            logging.error(self.login_error_message)
            return False, self.login_status

        except Exception as e:
            self.login_status = "FAILURE"
            self.login_error_message = f"An unexpected error occurred during login: {e}"
            logging.error(self.login_error_message)
            return False, self.login_status

        # Step 6.8: Save session if login is successful
        logging.info("Login successful.")
        self.client.dump_settings(self.SESSION_FILE)
        logging.info(f"Session settings saved to '{self.SESSION_FILE}'.")
        self.login_status = "SUCCESS"
        return True, self.login_status


class ImageProcessor:
    """
    Handles the final processing of images to prepare them for Instagram.
    """
    TARGET_SIZE = 1080  # For a 1080x1080 square canvas

    @staticmethod
    def process(path: str, output_path: str) -> str:
        """
        Processes a single image to fit within a 1080x1080 square with a black background,
        maintaining its original aspect ratio.

        Args:
            path (str): The path to the input image.
            output_path (str): The path to save the processed image.

        Returns:
            The path to the processed image.
        """
        try:
            # 14.2: Create a black background of size 1080x1080
            background = Image.new('RGB', (ImageProcessor.TARGET_SIZE, ImageProcessor.TARGET_SIZE), 'black')
            
            with Image.open(path) as img:
                # 14.1 & 14.3: Get dimensions and calculate new size
                img.thumbnail((ImageProcessor.TARGET_SIZE, ImageProcessor.TARGET_SIZE), Image.Resampling.LANCZOS)
                
                # 14.4: Calculate position to paste the image in the center
                paste_x = (ImageProcessor.TARGET_SIZE - img.width) // 2
                paste_y = (ImageProcessor.TARGET_SIZE - img.height) // 2
                
                # Paste the resized image onto the black background
                background.paste(img, (paste_x, paste_y))

            # Save the final image
            background.save(output_path, format='WEBP', quality=100, lossless=True, method=6, optimize=True, subsampling=0)
            logging.info(f"Successfully processed image '{path}' and saved to '{output_path}'")
            return output_path
            
        except Exception as e:
            logging.error(f"Failed to process image at {path}: {e}")
            raise


class VideoProcessor:
    """
    Handles the final processing of videos to prepare them for Instagram.
    """

    LANDSCAPE_SIZE = (1280, 720)
    PORTRAIT_SIZE = (720, 1280)

    @staticmethod
    def process(path: str, output_path: str) -> str:
        """
        Processes a single video to fit within a 1280x720 or 720x1280 canvas
        with a black background, maintaining its original aspect ratio and quality.

        Args:
            path (str): The path to the input video.
            output_path (str): The path to save the processed video.

        Returns:
            The path to the processed video.
        """
        video_clip = None
        try:
            video_clip = mp.VideoFileClip(path)
            
            # 15.1: Check dimensions to determine orientation
            is_landscape = video_clip.w >= video_clip.h
            
            # 15.2: Set target canvas size
            if is_landscape:
                target_size = VideoProcessor.LANDSCAPE_SIZE
                logging.info(f"Processing '{path}' as landscape video.")
            else:
                target_size = VideoProcessor.PORTRAIT_SIZE
                logging.info(f"Processing '{path}' as portrait video.")

            # 15.3: Resize video to fit the target canvas while maintaining aspect ratio
            resized_clip = video_clip.resize(height=target_size[1]) if is_landscape else video_clip.resize(width=target_size[0])
            
            # 15.2 & 15.4: Create a black background and composite the video on top
            background_clip = mp.ColorClip(size=target_size, color=(0, 0, 0), duration=video_clip.duration)
            
            final_clip = mp.CompositeVideoClip([background_clip, resized_clip.set_position("center")])
            
            # Ensure the original audio is preserved
            final_clip.audio = resized_clip.audio

            # 15.3.1: Write with high-quality settings
            final_clip.write_videofile(
                output_path,
                codec='libx265',
                audio_codec='aac',
                preset='slow',
                ffmpeg_params=[
                    '-crf', '18',                                        
                    '-b:a', '192k',
                    '-tag:v','hvc1',
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p'
                ],
                threads=4
            )           
            
            logging.info(f"Successfully processed video '{path}' and saved to '{output_path}'")
            return output_path

        except Exception as e:
            logging.error(f"Failed to process video at {path}: {e}")
            raise
        finally:
            if video_clip:
                video_clip.close()
            if 'resized_clip' in locals() and resized_clip:
                resized_clip.close()
            if 'final_clip' in locals() and final_clip:
                final_clip.close()
            if 'background_clip' in locals() and background_clip:
                background_clip.close()


class GIFConverter:
    @staticmethod
    def convert(path: str) -> str:
        """
        Converts a GIF file to an MP4 video, preserving quality and dimensions.
        The output file will have a new name to avoid conflicts.
        """
        logging.info(f"Starting GIF to MP4 conversion for: {os.path.basename(path)}")
        clip = None
        try:
            clip = mp.VideoFileClip(path)
            
            # Create a new, unique filename for the output
            base_name = os.path.splitext(os.path.basename(path))[0]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            out_name = f"{base_name}_{timestamp}.mp4"
            out_path = os.path.join(os.path.dirname(path), out_name)

            # Write the video file with a standard codec and no audio
            clip.write_videofile(out_path, codec='libx264', preset='slow', ffmpeg_params=['-crf','18'], audio=False, logger='bar', threads=4)
            
            logging.info(f"Successfully converted GIF to MP4: {out_name}")
            return out_path
        except Exception as e:
            logging.error(f"Error converting GIF {path} to MP4: {e}")
            raise
        finally:
            # Ensure the clip is closed to release file locks
            if clip:
                clip.close()


class WatermarkEngine:
    """
    Handles the creation of both image and text watermark layers.
    """

    @staticmethod
    def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
        """Wraps text to fit a specified width."""
        
        # The textwrap library is a simpler and more robust way to handle this.
        # We need to estimate an average character width to tell textwrap how many chars to wrap at.
        avg_char_width = font.getbbox("x")[2]
        if avg_char_width == 0: # Handle potential empty or zero-width bbox
            avg_char_width = font.getbbox("a")[2] # Fallback
            if avg_char_width == 0:
                return text # Cannot wrap, return original text

        # Calculate wrap width in characters
        wrap_width = int(max_width / avg_char_width)
        
        # Use textwrap to handle wrapping
        lines = textwrap.wrap(text, width=wrap_width)
        
        # In case textwrap estimation is off, we do a final check
        # This is a fallback and might not be perfect, but more robust than full manual wrapping.
        final_lines = []
        for line in lines:
            while font.getbbox(line)[2] > max_width:
                # If a line is still too long, trim words from the end.
                # This can happen with very long words.
                line = line[:-1]
            final_lines.append(line)

        return "\n".join(final_lines)

    @staticmethod
    def _calculate_position(
        layer_size: Tuple[int, int],
        watermark_size: Tuple[int, int],
        position: str,
        margin: int = 0
    ) -> Tuple[int, int]:
        """Calculates the (x, y) coordinates for the watermark based on a position string and a margin."""
        layer_width, layer_height = layer_size
        wm_width, wm_height = watermark_size
        
        # Horizontal positioning
        if 'left' in position:
            x = 50
        elif 'center' in position:
            x = (layer_width - wm_width) // 2
        elif 'right' in position:
            x = layer_width - wm_width - 50
        else: # Default to center
            x = (layer_width - wm_width) // 2

        # Vertical positioning
        if 'top' in position:
            y = 50
        elif 'middle' in position:
            y = (layer_height - wm_height) // 2
        elif 'bottom' in position:
            y = layer_height - wm_height - 50
        else: # Default to middle
            y = (layer_height - wm_height) // 2
            
        return (x, y)

    @staticmethod
    def create_image_watermark_layer(
        media_dimensions: Tuple[int, int],
        watermark_path: str,
        position: str,
        scale_percent: int,
        opacity_percent: int,
        output_path: str
    ) -> None:
        """
        Creates a transparent layer with a scaled and positioned image watermark.
        """
        logging.info(f"Creating image watermark layer for media size {media_dimensions}")
        
        # Create a transparent background layer matching the media size
        transparent_layer = Image.new('RGBA', media_dimensions, (0, 0, 0, 0))
        
        with Image.open(watermark_path).convert("RGBA") as watermark_img:
            # Scale the watermark
            scale_ratio = scale_percent / 100.0
            new_size = (int(watermark_img.width * scale_ratio), int(watermark_img.height * scale_ratio))
            watermark_img = watermark_img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Adjust opacity
            alpha = watermark_img.split()[3]
            alpha = alpha.point(lambda p: p * (opacity_percent / 100.0))
            watermark_img.putalpha(alpha)
            
            # Calculate position and paste
            paste_position = WatermarkEngine._calculate_position(media_dimensions, watermark_img.size, position)
            transparent_layer.paste(watermark_img, paste_position, watermark_img)

        # Save the final layer
        transparent_layer.save(output_path, "PNG")
        logging.info(f"Image watermark layer saved to {output_path}")

    @staticmethod
    def create_text_watermark_layer(
        media_dimensions: Tuple[int, int],
        text: str,
        font_path: str,
        font_size: int,
        color: str,
        position: str,
        output_path: str
    ) -> None:
        """
        Creates a transparent layer with rendered, wrapped text with margins.
        """
        logging.info(f"Creating text watermark layer for media size {media_dimensions}")
        
        MARGIN = 30
        
        # Create a transparent background layer
        transparent_layer = Image.new('RGBA', media_dimensions, (0, 0, 0, 0))
        draw = ImageDraw.Draw(transparent_layer)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            logging.error(f"Font file not found at {font_path}. Using default font.")
            font = ImageFont.load_default()

        # --- Text Wrapping Logic ---
        max_text_width = media_dimensions[0] - (2 * MARGIN)
        wrapped_text = WatermarkEngine._wrap_text(text, font, max_text_width)
        
        # Get wrapped text block size
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position with margin and draw text
        text_position = WatermarkEngine._calculate_position(
            media_dimensions, (text_width, text_height), position, margin=MARGIN
        )
        
        # Color mapping
        color_map = {
            'white': (255, 255, 255), 'black': (0, 0, 0), 'red': (255, 0, 0),
            'blue': (0, 0, 255), 'yellow': (255, 255, 0), 'green': (0, 128, 0)
        }
        text_color = color_map.get(color.lower(), (255, 255, 255)) # Default to white
        
        draw.text(text_position, wrapped_text, font=font, fill=text_color, align="center")

        # Save the final layer
        transparent_layer.save(output_path, "PNG")
        logging.info(f"Text watermark layer saved to {output_path}")


def is_video_file(path: str) -> bool:
    return path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))

class MediaCombiner:
    @staticmethod
    def combine(base_path: str, output_path: str, s1_layer_path: str = None, s2_layer_path: str = None, s3_audio_path: str = None) -> str:
        """
        Overlays watermark layers onto a media file and handles audio for videos.
        """
        if is_video_file(base_path):
            return MediaCombiner._combine_video(base_path, output_path, s1_layer_path, s2_layer_path, s3_audio_path)
        else:
            return MediaCombiner._combine_image(base_path, output_path, s1_layer_path, s2_layer_path)

    @staticmethod
    def _combine_image(base_image_path: str, output_path: str, s1_layer_path: str, s2_layer_path: str) -> str:
        try:
            base_image = Image.open(base_image_path).convert("RGBA")
            if s1_layer_path and os.path.exists(s1_layer_path):
                with Image.open(s1_layer_path) as layer1:
                    base_image.paste(layer1, (0, 0), layer1)
            if s2_layer_path and os.path.exists(s2_layer_path):
                with Image.open(s2_layer_path) as layer2:
                    base_image.paste(layer2, (0, 0), layer2)
            final_image = base_image.convert("RGB")
            final_image.save(output_path, format='JPEG', quality=100, optimize=True, subsampling=0)
            return output_path
        except Exception as e:
            logging.error(f"Error combining image {base_image_path}: {e}")
            raise

    @staticmethod
    def _combine_video(base_video_path: str, output_path: str, s1_layer_path: str, s2_layer_path: str, s3_audio_path: str) -> str:
        video_clip = s1_clip = s2_clip = audio_clip = None
        try:
            video_clip = mp.VideoFileClip(base_video_path)
            
            # Prepare video layers
            clips_to_composite = [video_clip]
            if s1_layer_path and os.path.exists(s1_layer_path):
                s1_clip = mp.ImageClip(s1_layer_path).set_duration(video_clip.duration).set_position(("center", "center"))
                clips_to_composite.append(s1_clip)
            if s2_layer_path and os.path.exists(s2_layer_path):
                s2_clip = mp.ImageClip(s2_layer_path).set_duration(video_clip.duration).set_position(("center", "center"))
                clips_to_composite.append(s2_clip)
            
            final_video = mp.CompositeVideoClip(clips_to_composite)

            # Step 12.7: Handle audio replacement
            if s3_audio_path and os.path.exists(s3_audio_path):
                logging.info(f"Replacing audio for {os.path.basename(base_video_path)} with {os.path.basename(s3_audio_path)}")
                audio_clip = mp.AudioFileClip(s3_audio_path)
                # The audio is already trimmed, just set it
                final_video = final_video.set_audio(audio_clip)
            else:
                # Keep original audio if no new audio is provided
                final_video.audio = video_clip.audio
                logging.info(f"Keeping original audio for {os.path.basename(base_video_path)}")

            final_video.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac',
                preset='medium', 
                ffmpeg_params=['-crf', '18'], 
                threads=4
            )
            return output_path
        except Exception as e:
            logging.error(f"Error combining video {base_video_path}: {e}")
            raise
        finally:
            # Close all clips to free up resources
            if video_clip: video_clip.close()
            if s1_clip: s1_clip.close()
            if s2_clip: s2_clip.close()
            if audio_clip: audio_clip.close()
            if 'final_video' in locals() and final_video: final_video.close()


class InstagramUploader:
    def upload_photo(self, client: Client, path: str, caption: str):
        """Uploads a single photo to Instagram."""
        logging.info(f"Uploading photo from {path} with caption: '{caption[:30]}...'")
        try:
            client.photo_upload(path, caption=caption)
            logging.info("Photo upload successful.")
        except Exception as e:
            logging.error(f"Failed to upload photo {path}: {e}")
            raise

    def upload_video(self, client: Client, path: str, caption: str):
        """Uploads a single video to Instagram."""
        logging.info(f"Uploading video from {path} with caption: '{caption[:30]}...'")
        try:
            client.video_upload(path, caption=caption)
            logging.info("Video upload successful.")
        except Exception as e:
            logging.error(f"Failed to upload video {path}: {e}")
            raise

    def upload_album(self, client: Client, paths: List[str], caption: str):
        """Uploads an album of photos and videos to Instagram."""
        if not paths or len(paths) < 2:
            raise ValueError("An album must contain at least 2 media files.")
        
        logging.info(f"Uploading album with {len(paths)} items and caption: '{caption[:30]}...'")
        try:
            client.album_upload(paths, caption=caption)
            logging.info("Album upload successful.")
        except Exception as e:
            logging.error(f"Failed to upload album: {e}")
            raise


# Step 2: Check for required libraries
def check_and_install_dependencies():
    """
    Checks if all required Python libraries are installed and installs them if not.
    """
    REQUIRED_LIBRARIES = {
        'python-telegram-bot': 'telegram',
        'instagrapi': 'instagrapi',
        'Pillow': 'PIL',
        'python-dotenv': 'dotenv',
        'moviepy': 'moviepy.editor',
        'filetype': 'filetype',
        'nest-asyncio': 'nest-asyncio'
    }
    
    missing_libraries = []
    for package_name, import_name in REQUIRED_LIBRARIES.items():
        try:
            import_module(import_name)
            logging.info(f"'{package_name}' is already installed.")
        except ImportError:
            logging.warning(f"'{package_name}' is not installed.")
            missing_libraries.append(package_name)
    
    if missing_libraries:
        logging.info(f"Attempting to install missing libraries: {', '.join(missing_libraries)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_libraries])
            logging.info("All missing dependencies have been successfully installed.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install dependencies: {e}. Please install them manually and restart the bot.")
            sys.exit(1)

# Step 3: Check and prepare folders
def prepare_folders() -> Tuple[str, List[str], Optional[str]]:
    """
    Ensures that the necessary folders ('downloads', 'fonts') exist and are prepared.

    Returns:
        A tuple containing:
        - The absolute path to the downloads folder.
        - A list of paths to available .ttf font files.
        - A warning message if no fonts are found, otherwise None.
    """
    # 3.1: Downloads folder
    downloads_path = os.path.join(os.getcwd(), 'downloads')
    if os.path.exists(downloads_path):
        logging.info("Downloads folder exists. Clearing its contents.")
        for f in os.listdir(downloads_path):
            try:
                os.remove(os.path.join(downloads_path, f))
            except Exception as e:
                logging.error(f"Could not remove file {f} from downloads: {e}")
    else:
        logging.info("Downloads folder not found. Creating it.")
        os.makedirs(downloads_path)

    # 3.2: Fonts folder
    fonts_path = os.path.join(os.getcwd(), 'fonts')
    if not os.path.exists(fonts_path):
        logging.info("Fonts folder not found. Creating it.")
        os.makedirs(fonts_path)
    
    # 3.3: Check for .ttf files
    font_files = [os.path.join(fonts_path, f) for f in os.listdir(fonts_path) if f.lower().endswith('.ttf')]
    font_warning = None
    if not font_files:
        font_warning = "Warning: The 'fonts' directory is empty or contains no .ttf files. Text watermarking will not be available."
        logging.warning(font_warning)
    else:
        logging.info(f"Found {len(font_files)} font(s): {', '.join([os.path.basename(f) for f in font_files])}")
        
    return downloads_path, font_files, font_warning

# Step 4: Check .env file
def load_environment_variables() -> Tuple[str, str, str]:
    """
    Loads required variables from a .env file and ensures they are present.

    Returns:
        A tuple containing the Telegram token, Instagram username, and Instagram password.
    """
    load_dotenv()
    
    telegram_token = os.getenv('TELEGRAM_TOKEN')
    instagram_user = os.getenv('INSTAGRAM_USER')
    instagram_pass = os.getenv('INSTAGRAM_PASS')

    missing_vars = []
    if not telegram_token:
        missing_vars.append('TELEGRAM_TOKEN')
    if not instagram_user:
        missing_vars.append('INSTAGRAM_USER')
    if not instagram_pass:
        missing_vars.append('INSTAGRAM_PASS')

    if missing_vars:
        error_message = f"Error: The .env file is missing or invalid. The following variables are required: {', '.join(missing_vars)}"
        logging.error(error_message)
        # This error is critical and should be communicated to the user via Telegram if possible,
        # but the bot can't start without the token, so we exit.
        sys.exit(error_message)
        
    return telegram_token, instagram_user, instagram_pass

def setup_logging():
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("bot_activity.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Reduce the noise from the HTTP library used by python-telegram-bot
    logging.getLogger("httpx").setLevel(logging.WARNING)

def initialize_app() -> dict:
    """
    Runs all setup steps and returns a configuration dictionary for the bot.
    """
    setup_logging()
    logging.info("--- Starting Bot Setup ---")
    
    check_and_install_dependencies()
    
    telegram_token, instagram_user, instagram_pass = load_environment_variables()
    downloads_path, font_files, font_warning = prepare_folders()
    
    try:
        nest_asyncio.apply()
        logging.info("nest_asyncio has been applied.")
    except ImportError:
        pass
        
    logging.info("--- Bot Setup Complete ---")
    
    return {
        "telegram_token": telegram_token,
        "instagram_user": instagram_user,
        "instagram_pass": instagram_pass,
        "downloads_path": downloads_path,
        "font_files": font_files,
        "font_warning": font_warning
    }


media_counter = 1
MAX_AUTH_ATTEMPTS = 3

# --- Helper Functions ---
def get_media_dimensions(path: str) -> Optional[tuple]:
    try:
        if is_video_file(path):
            with mp.VideoFileClip(path) as clip: return clip.size
        else:
            with Image.open(path) as img: return img.size
    except Exception as e:
        logging.error(f"Could not get dimensions for {path}: {e}"); return None

def get_video_duration(path: str) -> Optional[float]:
    try:
        with mp.VideoFileClip(path) as clip:
            return clip.duration
    except Exception as e:
        logging.error(f"Could not get duration for video {path}: {e}")
        return None

# --- Authentication Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logging.info("'/start' command received. Initiating authentication.")
    context.user_data['auth_attempts'] = 0
    ig_manager: AuthManager = context.application.bot_data['ig_manager']
    success, status = await asyncio.to_thread(ig_manager.login)
    if success:
        await update.message.reply_text("✅ Connection to Telegram and Instagram is successful.")
        return await send_welcome_message(update, context)
    if status == "2FA_REQUIRED":
        await update.message.reply_text("🔐 Please enter your 2FA code (from your authenticator app).")
        return States.AUTH_2FA
    elif status == "SMS_REQUIRED":
        await update.message.reply_text("📱 Please enter the SMS code sent to your phone.")
        return States.AUTH_SMS
    else:
        await update.message.reply_text(f"❌ Instagram login failed: {ig_manager.login_error_message}")
        return ConversationHandler.END

async def handle_2fa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == '❌ Cancel': return await cancel(update, context)
    context.user_data['auth_attempts'] += 1
    if context.user_data['auth_attempts'] > MAX_AUTH_ATTEMPTS:
        await update.message.reply_text("❌ Too many incorrect attempts. Halting operation.")
        return ConversationHandler.END
    code = update.message.text.strip()
    ig_manager: AuthManager = context.application.bot_data['ig_manager']
    success, status = await asyncio.to_thread(ig_manager.login, two_factor_code=code)
    if success:
        await update.message.reply_text("✅ Instagram connection successful!")
        return await send_welcome_message(update, context)
    else:
        remaining_attempts = MAX_AUTH_ATTEMPTS - context.user_data['auth_attempts']
        await update.message.reply_text(f"❌ Incorrect 2FA code. Please try again. ({remaining_attempts} attempts remaining)")
        return States.AUTH_2FA

async def handle_sms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == '❌ Cancel': return await cancel(update, context)
    context.user_data['auth_attempts'] += 1
    if context.user_data['auth_attempts'] > MAX_AUTH_ATTEMPTS:
        await update.message.reply_text("❌ Too many incorrect attempts. Halting operation.")
        return ConversationHandler.END
    code = update.message.text.strip()
    ig_manager: AuthManager = context.application.bot_data['ig_manager']
    success, status = await asyncio.to_thread(ig_manager.login, verification_code=code)
    if success:
        await update.message.reply_text("✅ Instagram connection successful!")
        return await send_welcome_message(update, context)
    else:
        remaining_attempts = MAX_AUTH_ATTEMPTS - context.user_data['auth_attempts']
        await update.message.reply_text(f"❌ Incorrect SMS code. Please try again. ({remaining_attempts} attempts remaining)")
        return States.AUTH_SMS

async def send_welcome_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # --- Directory Cleanup Logic ---
    # Per user request, clear the contents of the downloads folder at the start of any new workflow.
    downloads_path = context.application.bot_data['downloads_path']
    try:
        if not os.path.exists(downloads_path):
            os.makedirs(downloads_path)
            logging.info(f"Downloads directory created at: {downloads_path}")
        else:
            logging.info(f"Clearing contents of downloads directory: {downloads_path}")
            for filename in os.listdir(downloads_path):
                file_path = os.path.join(downloads_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            logging.info("Downloads directory contents cleared.")
    except Exception as e:
        logging.error(f"Could not clear downloads directory {downloads_path}: {e}")
        await update.message.reply_text("⚠️ Warning: Could not clean up temporary file directory. Please check bot logs.")

    await update.message.reply_text("Welcome! You can send 'Cancel' at any point to stop the current operation.")
    keyboard = [['📤 Album', '📎 Single'], ['❌ Cancel']]
    await update.message.reply_text('🤖 Please choose an upload mode:', reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True))
    context.user_data.clear()
    return States.MEDIA_TYPE

# --- Media Handling and Validation ---
async def handle_media_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    mode = 'album' if 'Album' in text else 'single'
    context.user_data['mode'] = mode
    msg = "Please send up to 10 photos or videos. Press 'Done' when you have sent all your files." if mode == 'album' else "Please send one photo or video."
    keyboard = [['🏁 Done', '❌ Cancel']] if mode == 'album' else [['❌ Cancel']]
    context.user_data['files'] = []
    await update.message.reply_text(msg, reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))
    return States.RECEIVE_MEDIA

async def download_media(update: Update, context: ContextTypes.DEFAULT_TYPE, downloads_path: str) -> Optional[str]:
    global media_counter
    msg = update.message
    file_id = None
    ext = '.jpg' # Default
    if msg.photo:
        file_id = msg.photo[-1].file_id
    elif msg.video:
        file_id = msg.video.file_id
        ext = '.mp4'
    elif msg.animation:
        file_id = msg.animation.file_id
        ext = '.gif'
    
    if not file_id:
        await msg.reply_text('⚠️ Could not identify file to download!'); return None

    file = await context.bot.get_file(file_id)
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{media_counter:03d}{ext}"
    media_counter += 1
    path = os.path.join(downloads_path, name)
    await file.download_to_drive(path)
    logging.info(f'Downloaded: {path}')
    return path

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    mode = context.user_data.get('mode', 'single')
    files = context.user_data.setdefault('files', [])
    if mode == 'album' and len(files) >= 10:
        await update.message.reply_text("You have already sent 10 files. Please press 'Done'.")
        return States.RECEIVE_MEDIA
    path = await download_media(update, context, context.application.bot_data['downloads_path'])
    if not path: return States.RECEIVE_MEDIA
    files.append(path)
    if mode == 'single':
        return await process_media(update, context)
    else:
        await update.message.reply_text(f"✅ Received file {len(files)} of 10.")
        return States.RECEIVE_MEDIA

async def process_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    files = context.user_data.get('files', [])
    mode = context.user_data.get('mode')
    if mode == 'album' and len(files) < 2:
        await update.message.reply_text("❌ Album uploads require at least 2 files. Your operation has been cancelled.", reply_markup=ReplyKeyboardRemove())
        return await start(update, context)
    await update.message.reply_text(f"Received {len(files)} file(s). Now starting validation...", reply_markup=ReplyKeyboardRemove())
    
    validated_files = []
    original_files = list(files) # Create a copy to iterate over
    conversion_occurred = False # Flag to check for GIF conversions

    for i, file_path in enumerate(original_files):
        try:
            file_type = FileValidator.validate(file_path)
            if file_type == 'gif':
                new_path = await asyncio.to_thread(GIFConverter.convert, file_path)
                original_files[i] = new_path
                file_path = new_path
                file_type = 'video'
                conversion_occurred = True # Set flag
            if file_type == 'video':
                duration = get_video_duration(file_path)
                if duration is None: raise ValueError(f"Could not read video duration for {os.path.basename(file_path)}.")
                if duration > 60:
                    await update.message.reply_text(f"❌ Video '{os.path.basename(file_path)}' is longer than 60 seconds ({duration:.1f}s) and cannot be processed.")
                    return await start(update, context)
            validated_files.append(file_path)
        except ValueError as e:
            await update.message.reply_text(f"❌ File '{os.path.basename(file_path)}' is not a supported type. Error: {e}")
            return await start(update, context)
            
    if not validated_files:
        await update.message.reply_text('No valid files to process.')
        return await start(update, context)
        
    context.user_data['processed'] = validated_files
    await update.message.reply_text('✅ File validation complete.')

    if conversion_occurred:
        await update.message.reply_text('Your GIF file(s) have been converted to video. Here is the preview:')
        return await send_previews(update, validated_files)
    else:
        # No conversion, so no need for an initial preview
        await update.message.reply_text('Do you want to continue with editing?', reply_markup=ReplyKeyboardMarkup([['✅ Yes, continue', '❌ No, Upload As Is'], ['❌ Cancel']], resize_keyboard=True))
        return States.CONFIRM

async def send_previews(update: Update, files: List[str]) -> int:
    media_group = [InputMediaPhoto(media=open(f, 'rb')) if not is_video_file(f) else InputMediaVideo(media=open(f, 'rb')) for f in files]
    await update.message.reply_media_group(media=media_group)
    await update.message.reply_text('Do you want to continue with editing?', reply_markup=ReplyKeyboardMarkup([['✅ Yes, continue', '❌ No, Upload As Is'], ['❌ Cancel']], resize_keyboard=True))
    return States.CONFIRM

async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if 'Yes' in update.message.text:
        await update.message.reply_text('Do you want to add an image watermark?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
        return States.ASK_IMAGE_WATERMARK
    else: # User chose 'No, Upload As Is'
        context.user_data['combined_files'] = context.user_data['processed']
        return await start_final_processing(update, context)

# --- Watermark Handlers ---
async def ask_image_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == 'Yes':
        await update.message.reply_text('Please send the watermark image file.', reply_markup=ReplyKeyboardRemove())
        return States.RECEIVE_IMAGE_WATERMARK
    else:
        return await ask_text_watermark(update, context)

async def receive_image_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message.photo:
        await update.message.reply_text('That is not an image. Please send an image file.')
        return States.RECEIVE_IMAGE_WATERMARK
    
    watermark_file = await update.message.photo[-1].get_file()
    watermark_path = os.path.join(context.application.bot_data['downloads_path'], 'watermark_img.png')
    await watermark_file.download_to_drive(watermark_path)
    
    with Image.open(watermark_path) as img:
        w, h = img.size
        if not (120 <= max(w, h) <= 480):
            await update.message.reply_text('Watermark size invalid (must be 120-480px). Please try again.')
            return States.RECEIVE_IMAGE_WATERMARK
            
    context.user_data['image_watermark_path'] = watermark_path
    kb = [['top-left', 'top-center', 'top-right'], ['middle-left', 'middle-center', 'middle-right'], ['bottom-left', 'bottom-center', 'bottom-right'], ['❌ Cancel']]
    await update.message.reply_text('Choose watermark position:', reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
    return States.CHOOSE_IMG_WATERMARK_POSITION

async def handle_img_position(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['img_watermark_position'] = update.message.text.lower()
    keyboard = [['50', '60', '70'], ['80', '90', '100'], ['❌ Cancel']]
    await update.message.reply_text('Choose scale (50-100%):', reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True))
    return States.CHOOSE_IMG_WATERMARK_SCALE

async def handle_img_scale(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['img_watermark_scale'] = int(update.message.text)
    keyboard = [['100', '90', '80'], ['70', '60', '50'], ['❌ Cancel']]
    await update.message.reply_text('Choose opacity (50-100%):', reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True))
    return States.CHOOSE_IMG_WATERMARK_OPACITY

async def generate_and_preview_image_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['img_watermark_opacity'] = int(update.message.text)
    await update.message.reply_text('⏳ Generating preview...', reply_markup=ReplyKeyboardRemove())
    
    media_dims = get_media_dimensions(context.user_data['processed'][0])
    if not media_dims:
        await update.message.reply_text('Error: Could not get media dimensions.'); return await cancel(update, context)
        
    output_path = os.path.join(context.application.bot_data['downloads_path'], 'S1_preview.png')
    try:
        await asyncio.to_thread(
            WatermarkEngine.create_image_watermark_layer,
            media_dimensions=media_dims,
            watermark_path=context.user_data['image_watermark_path'],
            position=context.user_data['img_watermark_position'],
            scale_percent=context.user_data['img_watermark_scale'],
            opacity_percent=context.user_data['img_watermark_opacity'],
            output_path=output_path
        )
        await update.message.reply_photo(photo=open(output_path, 'rb'), caption="Is this watermark okay?")
        await update.message.reply_text('Confirm this watermark?', reply_markup=ReplyKeyboardMarkup([['✅ Yes, Confirm', '❌ No, Retry'], ['❌ Cancel']], one_time_keyboard=True))
        return States.CONFIRM_IMG_WATERMARK
    except Exception as e:
        await update.message.reply_text(f"Error creating watermark preview: {e}"); return await cancel(update, context)

async def handle_img_watermark_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if 'No' in update.message.text:
        await update.message.reply_text('Do you want to add an image watermark?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
        return States.ASK_IMAGE_WATERMARK
        
    await update.message.reply_text("Applying image watermark to all media...", reply_markup=ReplyKeyboardRemove())
    s1_layers = []
    downloads_path = context.application.bot_data['downloads_path']
    for i, media_path in enumerate(context.user_data['processed']):
        media_dims = get_media_dimensions(media_path)
        if not media_dims: continue
        output_path = os.path.join(downloads_path, f'S1_{i+1}.png')
        try:
            await asyncio.to_thread(
                WatermarkEngine.create_image_watermark_layer,
                media_dimensions=media_dims,
                watermark_path=context.user_data['image_watermark_path'],
                position=context.user_data['img_watermark_position'],
                scale_percent=context.user_data['img_watermark_scale'],
                opacity_percent=context.user_data['img_watermark_opacity'],
                output_path=output_path
            )
            s1_layers.append(output_path)
        except Exception as e:
            logging.error(f"Failed to create image watermark for {media_path}: {e}")
    context.user_data['S1_layers'] = s1_layers
    await update.message.reply_text('Image watermark layers created.')
    return await ask_text_watermark(update, context)

async def ask_text_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text('Do you want to add a text watermark?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
    return States.ASK_TEXT_WATERMARK

async def handle_ask_text_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if 'Yes' in update.message.text:
        await update.message.reply_text('Please enter the text for the watermark.', reply_markup=ReplyKeyboardRemove())
        return States.RECEIVE_TEXT
    else:
        return await _check_and_ask_music(update, context)

async def receive_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == '❌ Cancel': return await cancel(update, context)
    context.user_data['text_watermark_text'] = update.message.text
    font_names = [os.path.basename(f) for f in context.application.bot_data['font_files']]
    if not font_names:
        if context.application.bot_data['font_warning']:
            await update.message.reply_text(context.application.bot_data['font_warning'])
        return await _check_and_ask_music(update, context)
    keyboard = [[name] for name in font_names]
    keyboard.append(['❌ Cancel'])
    await update.message.reply_text('Choose a font:', reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True))
    return States.CHOOSE_FONT

async def handle_font(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == '❌ Cancel': return await cancel(update, context)
    context.user_data['text_watermark_font'] = update.message.text
    keyboard = [['10', '15', '20'], ['25', '30', '35'], ['40', '45', '50'], ['❌ Cancel']]
    await update.message.reply_text('Choose font size (10-50):', reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True))
    return States.CHOOSE_FONT_SIZE

async def handle_font_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['text_watermark_size'] = int(update.message.text)
    colors = [['White', 'Black', 'Red'], ['Blue', 'Yellow', 'Green'], ['❌ Cancel']]
    await update.message.reply_text('Choose a color:', reply_markup=ReplyKeyboardMarkup(colors, one_time_keyboard=True))
    return States.CHOOSE_COLOR

async def handle_color(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['text_watermark_color'] = update.message.text
    positions = [['top–center'], ['middle–center'], ['bottom–center'], ['❌ Cancel']]
    await update.message.reply_text('Choose text position:', reply_markup=ReplyKeyboardMarkup(positions, one_time_keyboard=True))
    return States.CHOOSE_TEXT_POSITION

async def generate_and_preview_text_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['text_watermark_position'] = update.message.text.lower()
    await update.message.reply_text('⏳ Generating preview...', reply_markup=ReplyKeyboardRemove())

    media_dims = get_media_dimensions(context.user_data['processed'][0])
    if not media_dims:
        await update.message.reply_text('Error: Could not get media dimensions.'); return await cancel(update, context)
        
    font_name = context.user_data['text_watermark_font']
    font_path = next((f for f in context.application.bot_data['font_files'] if os.path.basename(f) == font_name), None)
    if not font_path:
        await update.message.reply_text(f"Error: Font '{font_name}' not found.")
        return await _check_and_ask_music(update, context)
        
    output_path = os.path.join(context.application.bot_data['downloads_path'], 'S2_preview.png')
    try:
        await asyncio.to_thread(
            WatermarkEngine.create_text_watermark_layer,
            media_dimensions=media_dims,
            text=context.user_data['text_watermark_text'],
            font_path=font_path,
            font_size=context.user_data['text_watermark_size'],
            color=context.user_data['text_watermark_color'],
            position=context.user_data['text_watermark_position'],
            output_path=output_path
        )
        await update.message.reply_photo(photo=open(output_path, 'rb'), caption="Is this text watermark okay?")
        await update.message.reply_text('Confirm this text watermark?', reply_markup=ReplyKeyboardMarkup([['✅ Yes, Confirm', '❌ No, Retry'], ['❌ Cancel']], one_time_keyboard=True))
        return States.CONFIRM_TEXT_WATERMARK
    except Exception as e:
        await update.message.reply_text(f"Error creating watermark preview: {e}"); return await cancel(update, context)

async def handle_text_watermark_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if 'No' in update.message.text:
        await update.message.reply_text('Do you want to add a text watermark?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
        return States.ASK_TEXT_WATERMARK
        
    await update.message.reply_text("Applying text watermark to all media...", reply_markup=ReplyKeyboardRemove())
    s2_layers = []
    downloads_path = context.application.bot_data['downloads_path']
    font_name = context.user_data['text_watermark_font']
    font_path = next((f for f in context.application.bot_data['font_files'] if os.path.basename(f) == font_name), None)
    
    for i, media_path in enumerate(context.user_data['processed']):
        media_dims = get_media_dimensions(media_path)
        if not media_dims: continue
        output_path = os.path.join(downloads_path, f'S2_{i+1}.png')
        try:
            await asyncio.to_thread(
                WatermarkEngine.create_text_watermark_layer,
                media_dimensions=media_dims,
                text=context.user_data['text_watermark_text'],
                font_path=font_path,
                font_size=context.user_data['text_watermark_size'],
                color=context.user_data['text_watermark_color'],
                position=context.user_data['text_watermark_position'],
                output_path=output_path
            )
            s2_layers.append(output_path)
        except Exception as e:
            logging.error(f"Failed to create text watermark for {media_path}: {e}")
    context.user_data['S2_layers'] = s2_layers
    await update.message.reply_text('Text watermark layers created.')
    return await _check_and_ask_music(update, context)

# --- Music Handlers (Step 12) ---
async def _check_and_ask_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Checks if any videos were sent and asks the user about adding music."""
    has_video = any(is_video_file(p) for p in context.user_data.get('processed', []))
    if not has_video:
        logging.info("No videos in batch, skipping music step.")
        # In the future, this will go to step 13 (combine_user_changes)
        await update.message.reply_text("No videos found, skipping music step.")
        return await combine_changes(update, context)

    await update.message.reply_text('Do you want to add music to the video(s)?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
    return States.ASK_ADD_MUSIC

async def ask_add_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles user's decision to add music or not."""
    if 'Yes' in update.message.text:
        await update.message.reply_text('Please send the music file (as an audio file).', reply_markup=ReplyKeyboardRemove())
        return States.RECEIVE_MUSIC
    else:
        # Skip to the next major step
        return await combine_changes(update, context)

async def receive_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives the audio file from the user."""
    if not update.message.audio:
        await update.message.reply_text('That is not a valid audio file. Please try again.')
        return States.RECEIVE_MUSIC
    
    audio_file = await update.message.audio.get_file()
    audio_path = os.path.join(context.application.bot_data['downloads_path'], 'music.mp3')
    await audio_file.download_to_drive(audio_path)
    context.user_data['music_path'] = audio_path
    
    await update.message.reply_text('Please enter the start time for the music in MM:SS format (e.g., 01:23).')
    return States.RECEIVE_MUSIC_START_TIME

async def receive_music_start_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives the music start time and generates a preview based on the longest video."""
    if update.message.text == '❌ Cancel': return await cancel(update, context)
    start_time_str = update.message.text
    context.user_data['music_start_time'] = start_time_str
    
    video_paths = [p for p in context.user_data.get('processed', []) if is_video_file(p)]
    if not video_paths:
        await update.message.reply_text("No videos found to add music to. Skipping music step.")
        return await combine_changes(update, context)

    # Use the duration of the longest video for the preview trim
    durations = [get_video_duration(p) for p in video_paths if get_video_duration(p) is not None]
    preview_duration = max(durations) if durations else 60.0
    
    await update.message.reply_text(
        "⏳ Trimming audio for preview based on your longest video. "
        "The final audio will be matched to each video's individual length.",
        reply_markup=ReplyKeyboardRemove()
    )
    output_path = os.path.join(context.application.bot_data['downloads_path'], 'S3_preview.mp3')
    
    try:
        await asyncio.to_thread(
            MusicAdder.trim_audio,
            audio_path=context.user_data['music_path'],
            video_duration=preview_duration,
            start_time_str=start_time_str,
            output_path=output_path
        )
        await update.message.reply_audio(audio=open(output_path, 'rb'), caption="Here is a preview of the trimmed audio.")
        await update.message.reply_text('Is this correct?', reply_markup=ReplyKeyboardMarkup([['✅ Yes, Confirm', '❌ No, Retry'], ['❌ Cancel']], one_time_keyboard=True))
        return States.CONFIRM_MUSIC
    except ValueError as e:
        await update.message.reply_text(f"❌ Error: {e}. Please enter a valid start time.")
        return States.RECEIVE_MUSIC_START_TIME
    except Exception as e:
        await update.message.reply_text(f"An unexpected error occurred while processing the audio: {e}")
        return await cancel(update, context)

async def handle_music_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles user confirmation of the trimmed audio."""
    if 'No' in update.message.text:
        # If user retries, we clean up the preview file to avoid confusion
        preview_path = os.path.join(context.application.bot_data['downloads_path'], 'S3_preview.mp3')
        if os.path.exists(preview_path):
            os.remove(preview_path)
        await update.message.reply_text('Do you want to add music to the video(s)?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
        return States.ASK_ADD_MUSIC

    # Don't create a single final audio file here.
    # Just confirm that music should be added in the next step.
    context.user_data['music_confirmed'] = True
    
    await update.message.reply_text('✅ Music confirmed. It will be added to each video individually.')
    return await combine_changes(update, context)

# --- Final Combination and Upload (Step 13) ---
async def combine_changes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Combines all selected edits (watermarks, audio) onto the base media files."""
    await update.message.reply_text(
        "Applying selected edits (watermarks/audio)...", 
        reply_markup=ReplyKeyboardRemove()
    )
    
    s1_layers = context.user_data.get('S1_layers', [])
    s2_layers = context.user_data.get('S2_layers', [])
    music_confirmed = context.user_data.get('music_confirmed', False)
    base_files = context.user_data.get('processed', [])
    
    # If no edits were made, just copy the files and proceed
    if not any([s1_layers, s2_layers, music_confirmed]):
        context.user_data['combined_files'] = base_files
        await update.message.reply_text("No edits were selected. Proceeding to final processing.")
        return await start_final_processing(update, context)

    combiner = MediaCombiner()
    combined_files = []
    downloads_path = context.application.bot_data['downloads_path']

    for i, file_path in enumerate(base_files):
        s1 = s1_layers[i] if i < len(s1_layers) else None
        s2 = s2_layers[i] if i < len(s2_layers) else None
        audio_for_this_video = None

        # --- Per-video audio trimming logic ---
        if music_confirmed and is_video_file(file_path):
            video_duration = get_video_duration(file_path)
            if video_duration:
                trimmed_audio_path = os.path.join(downloads_path, f"S3_{i+1}.mp3")
                try:
                    # This is now a synchronous call inside an asyncio.to_thread context
                    MusicAdder.trim_audio(
                        audio_path=context.user_data['music_path'],
                        video_duration=video_duration,
                        start_time_str=context.user_data['music_start_time'],
                        output_path=trimmed_audio_path
                    )
                    audio_for_this_video = trimmed_audio_path
                except Exception as e:
                    logging.error(f"Failed to trim audio for {file_path}: {e}")
                    await update.message.reply_text(f"⚠️ Could not apply audio to {os.path.basename(file_path)} due to an error.")
        
        output_filename = f"combined_{i}_{os.path.basename(file_path)}"
        output_path = os.path.join(downloads_path, output_filename)
        
        try:
            path = await asyncio.to_thread(
                combiner.combine,
                base_path=file_path,
                output_path=output_path,
                s1_layer_path=s1,
                s2_layer_path=s2,
                s3_audio_path=audio_for_this_video # Pass the unique audio path
            )
            combined_files.append(path)
        except Exception as e:
            logging.error(f"Failed to combine media {file_path}: {e}")
            await update.message.reply_text(f"❌ An error occurred while applying edits to {os.path.basename(file_path)}.")
            return await cancel(update, context)

    context.user_data['combined_files'] = combined_files
    await update.message.reply_text('Edits applied. Here is a preview of the result:')
    
    media_group = [InputMediaPhoto(media=open(f, 'rb')) if not is_video_file(f) else InputMediaVideo(media=open(f, 'rb')) for f in combined_files]
    await update.message.reply_media_group(media=media_group)
    
    await update.message.reply_text(
        'Are these edits correct?',
        reply_markup=ReplyKeyboardMarkup([['✅ Yes, continue', '❌ No, restart edits'], ['❌ Cancel']], one_time_keyboard=True)
    )
    return States.CONFIRM_COMBINED_MEDIA

async def handle_combined_media_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles user confirmation of the combined media."""
    if 'No' in update.message.text:
        # Restart the editing process from the beginning
        await update.message.reply_text("Restarting editing process...")
        await update.message.reply_text('Do you want to add an image watermark?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
        return States.ASK_IMAGE_WATERMARK
    
    # Proceed to the final processing step
    return await start_final_processing(update, context)

async def start_final_processing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Processes the combined media to their final dimensions and quality for Instagram.
    """
    await update.message.reply_text(
        "Starting final processing (resizing and padding)...", 
        reply_markup=ReplyKeyboardRemove()
    )
    
    final_files = []
    downloads_path = context.application.bot_data['downloads_path']
    
    for i, file_path in enumerate(context.user_data['combined_files']):
        output_filename = f"final_{i}_{os.path.basename(file_path)}"
        output_path = os.path.join(downloads_path, output_filename)
        
        try:
            if is_video_file(file_path):
                path = await asyncio.to_thread(VideoProcessor.process, path=file_path, output_path=output_path)
            else:
                path = await asyncio.to_thread(ImageProcessor.process, path=file_path, output_path=output_path)
            final_files.append(path)
        except Exception as e:
            logging.error(f"Failed during final processing for {file_path}: {e}")
            await update.message.reply_text(f"❌ An error occurred during final processing for {os.path.basename(file_path)}.")
            return await cancel(update, context)

    context.user_data['final_files'] = final_files
    await update.message.reply_text('This is the final result. Please confirm.')
    
    media_group = [InputMediaPhoto(media=open(f, 'rb')) if not is_video_file(f) else InputMediaVideo(media=open(f, 'rb')) for f in final_files]
    await update.message.reply_media_group(media=media_group)
    
    keyboard = [['✅ Yes, looks good', '❌ No, restart edits']]
    # Only offer video effects if there is at least one video
    if any(is_video_file(f) for f in final_files):
        keyboard[0].insert(1, 'Add Video Effects')

    keyboard.append(['❌ Cancel'])
    await update.message.reply_text(
        'Is this result okay?',
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    )
    return States.CONFIRM_FINAL_MEDIA

async def handle_final_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles user confirmation of the final processed media."""
    text = update.message.text
    if 'restart' in text:
        await update.message.reply_text("Restarting editing process...")
        await update.message.reply_text('Do you want to add an image watermark?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No']], one_time_keyboard=True))
        return States.ASK_IMAGE_WATERMARK
    elif 'Effects' in text:
        return await ask_video_effects(update, context)
    else: # 'looks good'
        await update.message.reply_text('Please enter the final caption for your post.', reply_markup=ReplyKeyboardRemove())
        return States.CAPTION

# --- Video Effects Handlers (Step 16) ---
async def ask_video_effects(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the user to select video effects."""
    context.user_data['selected_effects'] = []
    effects_list = [
        'Black & White', 'Color Saturation', 'Contrast / Brightness',
        'Chromatic Aberration', 'Pixelated Effect',
        'Invert Colors', 'Speed Control', 'Rotate',
        'VHS Look', 'Film Grain', 'Glitch',
        'Rolling Shutter', 'Neon Glow',
        'Cartoon / Painterly', 'Vignette', 'Fade In/Out'
    ]
    # Create a 3-column keyboard layout
    keyboard = [effects_list[i:i + 3] for i in range(0, len(effects_list), 3)]
    keyboard.append(['✅ Done Selecting', '❌ Cancel'])
    await update.message.reply_text(
        "Select up to 3 video effects. You can click an effect again to deselect it. Press 'Done Selecting' when finished.",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )
    return States.CHOOSE_EFFECTS

async def choose_effects(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles user's selection of video effects."""
    choice = update.message.text
    if choice == '❌ Cancel': return await cancel(update, context)
    selected = context.user_data.get('selected_effects', [])

    if 'Done' in choice:
        if not selected:
            await update.message.reply_text("No effects selected. Please enter the final caption.", reply_markup=ReplyKeyboardRemove())
            return States.CAPTION
        else:
            await update.message.reply_text(f"Applying effects: {', '.join(selected)}. Please wait...", reply_markup=ReplyKeyboardRemove())
            return await process_and_confirm_effects(update, context)

    if choice not in selected and len(selected) < 3:
        selected.append(choice)
        await update.message.reply_text(f"Added '{choice}'. Current effects: {', '.join(selected)}.")
    elif choice in selected:
        selected.remove(choice)
        await update.message.reply_text(f"Removed '{choice}'. Current effects: {', '.join(selected)}.")
    else:
        await update.message.reply_text("You can only select up to 3 effects.")

    return States.CHOOSE_EFFECTS

async def process_and_confirm_effects(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Applies the selected effects and sends a preview for confirmation."""
    engine = EffectsEngine()
    effects_applied_files = []
    
    for i, file_path in enumerate(context.user_data['final_files']):
        if is_video_file(file_path):
            output_path = os.path.join(context.application.bot_data['downloads_path'], f"effects_{i}_{os.path.basename(file_path)}")
            try:
                path = await asyncio.to_thread(
                    engine.apply_effects_in_sequence,
                    video_path=file_path,
                    effects=context.user_data['selected_effects'],
                    output_path=output_path
                )
                effects_applied_files.append(path)
            except Exception as e:
                logging.error(f"Error applying effects to {file_path}: {e}")
                await update.message.reply_text(f"❌ An error occurred while applying effects to {os.path.basename(file_path)}.")
                effects_applied_files.append(file_path) # Append original if effect fails
        else:
            effects_applied_files.append(file_path) # Keep non-video files as they are

    context.user_data['final_files_with_effects'] = effects_applied_files
    await update.message.reply_text('Preview of video(s) with effects:')
    
    media_group = [InputMediaVideo(media=open(f, 'rb')) for f in effects_applied_files if is_video_file(f)]
    if media_group:
        await update.message.reply_media_group(media=media_group)
    
    await update.message.reply_text(
        'Confirm final result with effects?',
        reply_markup=ReplyKeyboardMarkup([['✅ Yes, upload', '❌ No, restart effects'], ['❌ Cancel']], one_time_keyboard=True)
    )
    return States.CONFIRM_EFFECTS

async def handle_effects_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the final confirmation after applying effects."""
    if 'Yes' in update.message.text:
        context.user_data['final_files'] = context.user_data['final_files_with_effects']
        await update.message.reply_text('Effects confirmed. Please enter the final caption.', reply_markup=ReplyKeyboardRemove())
        return States.CAPTION
    else: # 'No, restart effects'
        await update.message.reply_text("Restarting effect selection...")
        return await ask_video_effects(update, context)

async def handle_caption_and_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives the caption and uploads the final media to Instagram."""
    if update.message.text == '❌ Cancel': return await cancel(update, context)
    caption = update.message.text
    await update.message.reply_text("🚀 Uploading to Instagram...", reply_markup=ReplyKeyboardRemove())

    try:
        files_to_upload = context.user_data.get('final_files', [])
        if not files_to_upload:
            await update.message.reply_text("❌ Error: No final files were found to upload.")
            return await cancel(update, context)

        mode = context.user_data.get('mode')
        ig_uploader = context.application.bot_data['ig_uploader']
        ig_client = context.application.bot_data['ig_manager'].client

        if mode == 'album':
            await asyncio.to_thread(ig_uploader.upload_album, client=ig_client, paths=files_to_upload, caption=caption)
        else:
            file_path = files_to_upload[0]
            if is_video_file(file_path):
                await asyncio.to_thread(ig_uploader.upload_video, client=ig_client, path=file_path, caption=caption)
            else:
                await asyncio.to_thread(ig_uploader.upload_photo, client=ig_client, path=file_path, caption=caption)
        
        await update.message.reply_text('✅ Upload successful!')

    except Exception as e:
        logging.exception("Upload to Instagram failed.")
        await update.message.reply_text(f'❌ An error occurred during upload: {e}')
    
    # Restart the conversation for a new upload
    await update.message.reply_text("Let's start a new project!")
    return await send_welcome_message(update, context)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the current operation and returns to the main menu."""
    await update.message.reply_text('♻️ Operation cancelled. Returning to the main menu.', reply_markup=ReplyKeyboardRemove())
    # Instead of ending, we restart the conversation from the beginning
    return await send_welcome_message(update, context)

def get_conversation_handler() -> ConversationHandler:
    """Builds the main conversation handler."""
    return ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            # Authentication
            States.AUTH_2FA: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_2fa)],
            States.AUTH_SMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_sms)],
            # Media Handling
            States.MEDIA_TYPE: [MessageHandler(filters.Regex('^📤 Album$|^📎 Single$') & ~filters.COMMAND, handle_media_type)],
            States.RECEIVE_MEDIA: [
                MessageHandler(filters.PHOTO | filters.VIDEO | filters.ANIMATION, handle_media),
                MessageHandler(filters.TEXT & filters.Regex(r'^🏁 Done$'), process_media)
            ],
            States.CONFIRM: [MessageHandler(filters.Regex('^✅ Yes, continue$|^❌ No, Upload As Is$') & ~filters.COMMAND, handle_confirmation)],
            # Image Watermark
            States.ASK_IMAGE_WATERMARK: [MessageHandler(filters.Regex('^Yes$|^No$'), ask_image_watermark)],
            States.RECEIVE_IMAGE_WATERMARK: [MessageHandler(filters.PHOTO, receive_image_watermark)],
            States.CHOOSE_IMG_WATERMARK_POSITION: [MessageHandler(filters.Regex('^(top|middle|bottom)-(left|center|right)$'), handle_img_position)],
            States.CHOOSE_IMG_WATERMARK_SCALE: [MessageHandler(filters.Regex('^50$|^60$|^70$|^80$|^90$|^100$'), handle_img_scale)],
            States.CHOOSE_IMG_WATERMARK_OPACITY: [MessageHandler(filters.Regex('^100$|^90$|^80$|^70$|^60$|^50$'), generate_and_preview_image_watermark)],
            States.CONFIRM_IMG_WATERMARK: [MessageHandler(filters.Regex('^✅ Yes, Confirm$|^❌ No, Retry$'), handle_img_watermark_confirmation)],
            # Text Watermark
            States.ASK_TEXT_WATERMARK: [MessageHandler(filters.Regex('^Yes$|^No$'), handle_ask_text_watermark)],
            States.RECEIVE_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_text)],
            States.CHOOSE_FONT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_font)],
            States.CHOOSE_FONT_SIZE: [MessageHandler(filters.Regex('^10$|^15$|^20$|^25$|^30$|^35$|^40$|^45$|^50$'), handle_font_size)],
            States.CHOOSE_COLOR: [MessageHandler(filters.Regex('^White$|^Black$|^Red$|^Blue$|^Yellow$|^Green$'), handle_color)],
            States.CHOOSE_TEXT_POSITION: [MessageHandler(filters.Regex('^top–center$|^middle–center$|^bottom–center$'), generate_and_preview_text_watermark)],
            States.CONFIRM_TEXT_WATERMARK: [MessageHandler(filters.Regex('^✅ Yes, Confirm$|^❌ No, Retry$'), handle_text_watermark_confirmation)],
            # Music
            States.ASK_ADD_MUSIC: [MessageHandler(filters.Regex('^Yes$|^No$'), ask_add_music)],
            States.RECEIVE_MUSIC: [MessageHandler(filters.AUDIO, receive_music)],
            States.RECEIVE_MUSIC_START_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_music_start_time)],
            States.CONFIRM_MUSIC: [MessageHandler(filters.Regex('^✅ Yes, Confirm$|^❌ No, Retry$'), handle_music_confirmation)],
            # Combination & Final Processing
            States.CONFIRM_COMBINED_MEDIA: [MessageHandler(filters.Regex('^✅ Yes, continue$|^❌ No, restart edits$'), handle_combined_media_confirmation)],
            States.CONFIRM_FINAL_MEDIA: [MessageHandler(filters.Regex('^✅ Yes, looks good$|^❌ No, restart edits$|^Add Video Effects$'), handle_final_confirmation)],
            # Video Effects
            States.ASK_VIDEO_EFFECTS: [MessageHandler(filters.TEXT, ask_video_effects)],
            States.CHOOSE_EFFECTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_effects)],
            States.CONFIRM_EFFECTS: [MessageHandler(filters.Regex('^✅ Yes, upload$|^❌ No, restart effects$'), handle_effects_confirmation)],
            # Caption and Upload
            States.CAPTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_caption_and_upload)],
        },
        fallbacks=[CommandHandler('cancel', cancel), MessageHandler(filters.Regex('^❌ Cancel$'), cancel)],
        conversation_timeout=1440, # 24 minutes
        allow_reentry=True
    )


def main():
    """Main function to configure and run the bot."""
    # Set environment variables for specific runtime conditions
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime'
    os.environ['ALSA_CONFIG_PATH'] = '/dev/null'

    # Run the setup process and get the configuration
    config = initialize_app()

    # Create instances of the managers
    ig_manager = AuthManager(
        username=config["instagram_user"],
        password=config["instagram_pass"]
    )
    ig_uploader = InstagramUploader()

    # Create the Telegram Application using the token from setup
    builder = Application.builder().token(config["telegram_token"])
    
    # Set a high connection pool size to avoid issues with multiple media uploads
    builder.get_updates_http_version("1.1")
    builder.http_version("1.1")
    
    # Increase timeouts to handle large files and slow connections, per user request
    builder.read_timeout(300)
    builder.write_timeout(300)

    app = builder.build()

    # --- Share instances and config with the application context ---
    # This makes them accessible in all handlers via context.application.bot_data
    app.bot_data['ig_manager'] = ig_manager
    app.bot_data['ig_uploader'] = ig_uploader
    app.bot_data['downloads_path'] = config["downloads_path"]
    app.bot_data['font_files'] = config["font_files"]
    app.bot_data['font_warning'] = config["font_warning"]
    
    # Add the conversation handler to the application
    conv_handler = get_conversation_handler()
    app.add_handler(conv_handler)

    # Start the bot
    logging.info("Bot is starting to poll for updates...")
    app.run_polling()

if __name__ == '__main__':
    main()
