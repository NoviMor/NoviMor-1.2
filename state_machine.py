from enum import IntEnum, auto

class States(IntEnum):
    START = auto() # Represents the entry point after /start
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
    COMBINE_CHANGES = auto()
    CONFIRM_COMBINED_MEDIA = auto()

    # Final Processing (Steps 14 & 15)
    START_FINAL_PROCESSING = auto()
    CONFIRM_FINAL_MEDIA = auto()

    # Video Effects (Step 16)
    ASK_VIDEO_EFFECTS = auto()
    CHOOSE_EFFECTS = auto()
    CONFIRM_EFFECTS = auto()

    # LUT Sub-conversation
    ASK_LUT_TYPE = auto()
    CHOOSE_BUILTIN_LUT = auto()
    RECEIVE_LUT_FILE = auto()

    # Parameterization Sub-conversations
    ASK_CONTRAST_LEVEL = auto()
    ASK_SATURATION_LEVEL = auto()
    ASK_ABERRATION_LEVEL = auto()
    ASK_PIXELATE_LEVEL = auto()
    ASK_SPEED_LEVEL = auto()
    ASK_ROTATE_OPTION = auto()
    ASK_GRAIN_LEVEL = auto()
    ASK_KENBURNS_LEVEL = auto()
    ASK_LUT_BLEND_LEVEL = auto()
    ASK_GLITCH_LEVEL = auto()
    ASK_SHUTTER_LEVEL = auto()
    ASK_NEON_LEVEL = auto()
    ASK_CARTOON_LEVEL = auto()
    ASK_VIGNETTE_LEVEL = auto()
    ASK_FADE_DURATION = auto()

    # Render Quality
    ASK_RENDER_QUALITY = auto()
    
    # Finalize
    CAPTION = auto()
