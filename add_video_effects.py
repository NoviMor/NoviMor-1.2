import moviepy.editor as mp
from moviepy.video.fx import all as vfx
import numpy as np
import random
import re
import os
from scipy.ndimage import sobel, zoom
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

    def apply_lut(self, clip, cube_file_path, level: str = 'high'):
        """Applies a 3D LUT to a video clip with a specified blend level."""
        level_map = {'low': 0.4, 'medium': 0.7, 'high': 1.0}
        blend = level_map.get(level, 1.0)

        if blend == 0:
            return clip
        
        lut_table, lut_size = self.parse_cube_file(cube_file_path)
        grid_points = np.linspace(0, 1, lut_size)
        interpolator = RegularGridInterpolator((grid_points, grid_points, grid_points), lut_table)

        def apply_lut_to_frame(frame):
            original_shape = frame.shape
            normalized_frame = frame.astype(np.float32) / 255.0
            pixels = normalized_frame.reshape(-1, 3)
            new_pixels = interpolator(pixels)
            new_frame = (np.clip(new_pixels, 0, 1) * 255).astype(np.uint8)
            return new_frame.reshape(original_shape)

        lut_clip = clip.fl_image(apply_lut_to_frame)

        if blend == 1.0:
            return lut_clip
        else:
            # Composite the original clip with the LUT clip on top with opacity
            final_clip = mp.CompositeVideoClip([clip, lut_clip.set_opacity(blend)])
            return final_clip
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

    def apply_ken_burns(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Applies a Ken Burns (zoom-in) effect using numpy/scipy for performance."""
        level_map = {'low': 1.0, 'medium': 1.25, 'high': 1.5}
        zoom_factor = level_map.get(level, 1.25)

        if zoom_factor == 1.0:
            return clip

        duration = clip.duration
        w, h = clip.size

        def effect(get_frame, t):
            frame = get_frame(t)
            
            current_zoom = 1.0 + (zoom_factor - 1.0) * (t / duration)
            
            # Zoom the frame using scipy
            zoomed_frame = zoom(frame, (current_zoom, current_zoom, 1), order=1)
            
            # Crop the center
            zh, zw, _ = zoomed_frame.shape
            crop_x_start = (zw - w) // 2
            crop_y_start = (zh - h) // 2
            
            return zoomed_frame[crop_y_start:crop_y_start+h, crop_x_start:crop_x_start+w, :]

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

    def apply_color_saturation(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Applies color saturation effect based on a level."""
        level_map = {'low': 1.3, 'medium': 1.7, 'high': 2.2}
        factor = level_map.get(level, 1.7)
        return clip.fx(vfx.colorx, factor)

    def apply_contrast_brightness(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Adjusts contrast and brightness based on a level."""
        contrast_map = {
            'low': 0.2,
            'medium': 0.6,
            'high': 1.0
        }
        contrast_value = contrast_map.get(level, 0.6)
        return clip.fx(vfx.lum_contrast, contrast=contrast_value)
        
    def apply_chromatic_aberration(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Applies a chromatic aberration (RGB split) effect based on a level."""
        level_map = {'low': 3, 'medium': 6, 'high': 10}
        shift = level_map.get(level, 6)
        def effect(frame):
            r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
            r_shifted = np.roll(r, -shift, axis=1)
            b_shifted = np.roll(b, shift, axis=1)
            return np.stack([r_shifted, g, b_shifted], axis=-1).astype('uint8')
        return clip.fl_image(effect).set_duration(clip.duration)

    def apply_pixelated(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Applies a pixelated effect by resizing down and then up."""
        level_map = {'low': 6, 'medium': 12, 'high': 20}
        pixel_size = level_map.get(level, 12)
        return clip.fx(vfx.resize, 1/pixel_size).fx(vfx.resize, pixel_size)

    def apply_invert_colors(self, clip: mp.VideoClip) -> mp.VideoClip:
        """Inverts the colors of the video."""
        return clip.fx(vfx.invert_colors)

    def apply_speed_control(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Changes the speed of the video based on a level."""
        level_map = {'low': 1.25, 'medium': 1.5, 'high': 2.0}
        factor = level_map.get(level, 1.5)
        return clip.fx(vfx.speedx, factor)

    def apply_rotate(self, clip: mp.VideoClip, level: str = 'high') -> mp.VideoClip:
        """Rotates the video a specified number of degrees."""
        # Moviepy rotates counter-clockwise, so we use negative for clockwise
        level_map = {'low': -15, 'medium': -45, 'high': -90}
        angle = level_map.get(level, -90)
        return clip.fx(vfx.rotate, angle)

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

    def apply_film_grain(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Adds film grain noise to each frame based on a level."""
        level_map = {'low': 0.1, 'medium': 0.175, 'high': 0.25}
        strength = level_map.get(level, 0.175)
        def effect(frame):
            noise = np.random.randint(-25, 25, frame.shape) * strength
            return np.clip(frame + noise, 0, 255).astype('uint8')
        return clip.fl_image(effect).set_duration(clip.duration)

    def apply_glitch(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Applies an approximate digital glitch effect."""
        level_map = {'low': 0.1, 'medium': 0.2, 'high': 0.3} # 10%, 20%, 30% chance
        probability = level_map.get(level, 0.2)
        def effect(get_frame, t):
            frame = get_frame(t).copy()
            if random.random() < probability:
                h, w, _ = frame.shape
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

    def apply_rolling_shutter(self, clip: mp.VideoClip, level: str = 'medium', freq: float = 5) -> mp.VideoClip:
        """Applies a rolling shutter wobble effect."""
        level_map = {'low': 5, 'medium': 12, 'high': 20}
        intensity = level_map.get(level, 12)
        def effect(get_frame, t):
            frame = get_frame(t)
            h, w, _ = frame.shape
            shift = (intensity * np.sin(2 * np.pi * (freq * t + (np.arange(h) / h)))).astype(int)
            cols = np.arange(w)
            # Repeat for each row and add the shift
            shifted_cols = cols[np.newaxis, :] + shift[:, np.newaxis]
            # Clip the indices to be within the frame width
            shifted_cols = np.clip(shifted_cols, 0, w - 1)
            # Use advanced indexing to create the wobbled frame
            return frame[np.arange(h)[:, np.newaxis], shifted_cols]
        return clip.fl(effect)

    def apply_neon_glow(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Applies an approximate neon edge effect."""
        level_map = {'low': 80, 'medium': 50, 'high': 30} # Lower threshold = more glow
        threshold = level_map.get(level, 50)
        def effect(frame):
            gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
            sx = sobel(gray, axis=0, mode='constant')
            sy = sobel(gray, axis=1, mode='constant')
            edges = np.hypot(sx, sy)
            edges = (edges / np.max(edges) * 255)
            neon_color = np.array([0, 255, 255])
            neon_frame = np.zeros_like(frame)
            neon_frame[edges > threshold] = neon_color
            return neon_frame
        return clip.fl_image(effect).set_duration(clip.duration)

    def apply_cartoon_painterly(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Applies a simplified cartoon/painterly effect using median filter and posterization."""
        level_map = {'low': 3, 'medium': 5, 'high': 7} # Median filter size
        filter_size = level_map.get(level, 5)
        def effect(frame):
            img = Image.fromarray(frame)
            img = img.filter(ImageFilter.MedianFilter(size=filter_size))
            img = img.quantize(colors=64).convert('RGB')
            return np.array(img)
        return clip.fl_image(effect).set_duration(clip.duration)

    def apply_vignette(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Applies a vignette (darkened edges) effect."""
        level_map = {'low': 0.5, 'medium': 0.75, 'high': 1.0}
        strength = level_map.get(level, 0.75)
        
        w, h = clip.size
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        radial_grad = dist_from_center / max_dist
        vignette_mask = 1 - (radial_grad**2) * strength
        
        def effect(frame):
            return (frame * vignette_mask[:, :, np.newaxis]).astype('uint8')
            
        return clip.fl_image(effect).set_duration(clip.duration)

    def apply_fade_in_out(self, clip: mp.VideoClip, level: str = 'medium') -> mp.VideoClip:
        """Applies a fade-in and fade-out with variable duration."""
        level_map = {'low': 1.0, 'medium': 1.5, 'high': 2.0} # Duration in seconds
        duration = level_map.get(level, 1.5)
        return clip.fx(vfx.fadein, duration).fx(vfx.fadeout, duration)
