import moviepy.editor as mp
from moviepy.video.fx import all as vfx
import numpy as np
import random
import re
import os
from scipy.ndimage import sobel
from scipy.interpolate import RegularGridInterpolator
from PIL import Image, ImageFilter
import uuid

class EffectsEngine:
    @staticmethod
    def parse_cube_file(file_path):
        """Parses a .cube LUT file and returns the table data and size."""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        lut_size = 0
        lut_data = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('LUT_3D_SIZE'):
                lut_size = int(line.split()[-1])
            elif re.match(r'^[0-9eE.+-]+\s+[0-9eE.+-]+\s+[0-9eE.+-]+', line):
                lut_data.append([float(c) for c in line.split()])

        if lut_size == 0 or not lut_data:
            raise ValueError("Invalid or unsupported .cube file format.")

        return np.array(lut_data).reshape((lut_size, lut_size, lut_size, 3)), lut_size

    def apply_lut(self, clip, cube_file_path):
        """Applies a 3D LUT to a video clip."""
        lut_table, lut_size = self.parse_cube_file(cube_file_path)
        
        # Create the grid points for the interpolator
        grid_points = np.linspace(0, 1, lut_size)
        
        # Create the interpolator
        interpolator = RegularGridInterpolator((grid_points, grid_points, grid_points), lut_table)

        def apply_lut_to_frame(frame):
            # Normalize frame to 0-1 range
            original_shape = frame.shape
            normalized_frame = frame.astype(np.float32) / 255.0
            
            # Reshape for interpolation
            pixels = normalized_frame.reshape(-1, 3)
            
            # Interpolate
            new_pixels = interpolator(pixels)
            
            # Denormalize and reshape back
            new_frame = (np.clip(new_pixels, 0, 1) * 255).astype(np.uint8)
            
            return new_frame.reshape(original_shape)

        return clip.fl_image(apply_lut_to_frame)
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

    def apply_ken_burns(self, clip: mp.VideoClip, zoom_factor: float = 1.2) -> mp.VideoClip:
        """Applies a Ken Burns (zoom-in) effect."""
        duration = clip.duration
        w, h = clip.size

        def effect(get_frame, t):
            frame = get_frame(t)
            
            # Calculate current zoom level
            current_zoom = 1.0 + (zoom_factor - 1.0) * (t / duration)
            
            pil_img = Image.fromarray(frame)
            
            # Resize (zoom in)
            zoomed_w = int(w * current_zoom)
            zoomed_h = int(h * current_zoom)
            zoomed_img = pil_img.resize((zoomed_w, zoomed_h), Image.LANCZOS)
            
            # Crop the center
            crop_x = (zoomed_w - w) // 2
            crop_y = (zoomed_h - h) // 2
            
            cropped_img = zoomed_img.crop((crop_x, crop_y, crop_x + w, crop_y + h))
            
            return np.array(cropped_img)

        # Use fl to apply the time-dependent effect
        return clip.fl(effect, apply_to=['video'])

    def __init__(self):
        """
        Initializes the EffectsEngine and the mapping of effect names to methods.
        """
        self.effects_map = {
            # Parameterized Effects
            'Cinematic Color (LUT)': self.apply_lut,

            # Simple Effects
            'Ken Burns': self.apply_ken_burns,
            'Black & White': self.apply_black_and_white,
            'Fade In/Out': self.apply_fade_in_out,
            'Pixelated Effect': self.apply_pixelated,
            'Glitch': self.apply_glitch,
            'Neon Glow': self.apply_neon_glow,
            'VHS Look': self.apply_vhs_look,
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

    def apply_effects_in_sequence(self, video_path: str, effects: list, output_path: str, quality: str = 'final') -> str:
        """
        Applies a list of effects to a video in the specified order.
        Effects can be strings (for simple effects) or tuples (for parameterized effects).
        """
        clip = mp.VideoFileClip(video_path)
        
        for effect in effects:
            if isinstance(effect, tuple):
                effect_name, *params = effect
                if effect_name in self.effects_map:
                    clip = self.effects_map[effect_name](clip, *params)
                else:
                    print(f"Warning: Parameterized effect '{effect_name}' not found.")
            elif isinstance(effect, str):
                if effect in self.effects_map:
                    clip = self.effects_map[effect](clip)
                else:
                    print(f"Warning: Effect '{effect}' not found.")
            else:
                print(f"Warning: Invalid effect format: {effect}")

        if quality == 'draft':
            ffmpeg_params = ['-preset', 'ultrafast', '-crf', '28']
        else: # final
            ffmpeg_params = ['-preset', 'slow', '-crf', '18']

        clip.write_videofile(output_path, codec='libx264', audio_codec='aac', ffmpeg_params=ffmpeg_params)
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

    def apply_contrast_brightness(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Adjusts contrast and brightness based on a level."""
        contrast_map = {
            'low': -0.3,
            'medium': 0.5,
            'high': 1.0
        }
        contrast_value = contrast_map.get(level, 0.5) # Default to medium
        return clip.fx(vfx.lum_contrast, contrast=contrast_value)
        
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
