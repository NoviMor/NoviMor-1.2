import logging
import os
import shutil
from typing import Optional

from PIL import Image
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ContextTypes

# Assuming state_machine.py contains the States enum
from state_machine import States


# --- Helper Functions ---
def get_media_dimensions(path: str) -> Optional[tuple]:
    """Gets the dimensions (width, height) of an image or video."""
    try:
        if is_video_file(path):
            import moviepy.editor as mp
            with mp.VideoFileClip(path) as clip:
                return clip.size
        else:
            with Image.open(path) as img:
                return img.size
    except Exception as e:
        logging.error(f"Could not get dimensions for {path}: {e}")
        return None


def get_video_duration(path: str) -> Optional[float]:
    """Gets the duration of a video in seconds."""
    try:
        import moviepy.editor as mp
        with mp.VideoFileClip(path) as clip:
            return clip.duration
    except Exception as e:
        logging.error(f"Could not get duration for video {path}: {e}")
        return None


def is_video_file(path: str) -> bool:
    """Checks if a file path points to a video based on its extension."""
    return path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))


async def send_welcome_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Clears downloads folder, sends a welcome message, and asks for upload mode."""
    # --- Directory Cleanup Logic ---
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
    await update.message.reply_text(
        '🤖 Please choose an upload mode:',
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    )
    # Clear user_data for the new workflow
    context.user_data.clear()
    return States.MEDIA_TYPE


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the current operation and returns to the main menu."""
    await update.message.reply_text(
        '♻️ Operation cancelled. Returning to the main menu.',
        reply_markup=ReplyKeyboardRemove()
    )
    # Restart the conversation from the beginning
    return await send_welcome_message(update, context)
