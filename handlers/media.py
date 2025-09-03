import asyncio
import logging
import os
from datetime import datetime
from typing import List, Optional

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InputMediaPhoto, InputMediaVideo
from telegram.ext import ContextTypes

from media_processor import GIFConverter
from state_machine import States
from utils import FileValidator
from handlers.common import get_video_duration, is_video_file
from handlers import upload
# We will need to import the start function for error cases
# from handlers.auth import start 
# For now, this will cause a circular import. This will be resolved in the final step.

media_counter = 1

async def handle_media_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's choice of upload mode (single or album)."""
    text = update.message.text
    mode = 'album' if 'Album' in text else 'single'
    context.user_data['mode'] = mode
    msg = "Please send up to 10 photos or videos. Press 'Done' when you have sent all your files." if mode == 'album' else "Please send one photo or video."
    keyboard = [['🏁 Done', '❌ Cancel']] if mode == 'album' else [['❌ Cancel']]
    context.user_data['files'] = []
    await update.message.reply_text(msg, reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))
    return States.RECEIVE_MEDIA


async def download_media(update: Update, context: ContextTypes.DEFAULT_TYPE, downloads_path: str) -> Optional[str]:
    """Downloads a media file from a Telegram message."""
    global media_counter
    msg = update.message
    file_id = None
    ext = '.jpg'  # Default extension

    if msg.photo:
        file_id = msg.photo[-1].file_id
    elif msg.video:
        file_id = msg.video.file_id
        ext = '.mp4'
    elif msg.animation:
        file_id = msg.animation.file_id
        ext = '.gif'

    if not file_id:
        await msg.reply_text('⚠️ Could not identify file to download!')
        return None

    file = await context.bot.get_file(file_id)
    # Use a unique name for the downloaded file
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{media_counter:03d}{ext}"
    media_counter += 1
    path = os.path.join(downloads_path, name)
    await file.download_to_drive(path)
    logging.info(f'Downloaded: {path}')
    return path


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives a media file from the user and downloads it."""
    mode = context.user_data.get('mode', 'single')
    files = context.user_data.setdefault('files', [])

    if mode == 'album' and len(files) >= 10:
        await update.message.reply_text("You have already sent 10 files. Please press 'Done'.")
        return States.RECEIVE_MEDIA

    path = await download_media(update, context, context.application.bot_data['downloads_path'])
    if not path:
        return States.RECEIVE_MEDIA

    files.append(path)

    if mode == 'single':
        # If in single mode, proceed immediately to processing
        return await process_media(update, context)
    else:
        await update.message.reply_text(f"✅ Received file {len(files)} of 10.")
        return States.RECEIVE_MEDIA


async def process_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Validates the received media files, converting GIFs and checking video durations."""
    files = context.user_data.get('files', [])
    mode = context.user_data.get('mode')

    if mode == 'album' and len(files) < 2:
        await update.message.reply_text("❌ Album uploads require at least 2 files. Your operation has been cancelled.", reply_markup=ReplyKeyboardRemove())
        # This will need to be changed to return a state that restarts the flow.
        # For now, we can't call start() directly.
        return States.START # We'll need a state to represent the beginning

    await update.message.reply_text(f"Received {len(files)} file(s). Now starting validation...", reply_markup=ReplyKeyboardRemove())

    validated_files = []
    original_files = list(files)  # Create a copy to iterate over
    conversion_occurred = False  # Flag to check for GIF conversions

    for i, file_path in enumerate(original_files):
        try:
            file_type = FileValidator.validate(file_path)
            if file_type == 'gif':
                new_path = await asyncio.to_thread(GIFConverter.convert, file_path)
                original_files[i] = new_path
                file_path = new_path
                file_type = 'video'
                conversion_occurred = True
            
            if file_type == 'video':
                duration = get_video_duration(file_path)
                if duration is None:
                    raise ValueError(f"Could not read video duration for {os.path.basename(file_path)}.")
                if duration > 60:
                    await update.message.reply_text(f"❌ Video '{os.path.basename(file_path)}' is longer than 60 seconds ({duration:.1f}s) and cannot be processed.")
                    return States.START # Restart

            validated_files.append(file_path)
        except ValueError as e:
            await update.message.reply_text(f"❌ File '{os.path.basename(file_path)}' is not a supported type. Error: {e}")
            return States.START # Restart

    if not validated_files:
        await update.message.reply_text('No valid files to process.')
        return States.START # Restart

    context.user_data['processed'] = validated_files
    await update.message.reply_text('✅ File validation complete.')

    if conversion_occurred:
        await update.message.reply_text('Your GIF file(s) have been converted to video. Here is the preview:')
        return await send_previews(update, context, validated_files)
    else:
        await update.message.reply_text('Do you want to continue with editing?', reply_markup=ReplyKeyboardMarkup([['✅ Yes, continue', '❌ No, Upload As Is'], ['❌ Cancel']], resize_keyboard=True))
        return States.CONFIRM


async def send_previews(update: Update, context: ContextTypes.DEFAULT_TYPE, files: List[str]) -> int:
    """Sends a media group preview of the processed files."""
    media_group = []
    for f in files:
        if is_video_file(f):
            media_group.append(InputMediaVideo(media=open(f, 'rb')))
        else:
            media_group.append(InputMediaPhoto(media=open(f, 'rb')))
            
    await update.message.reply_media_group(media=media_group)
    await update.message.reply_text('Do you want to continue with editing?', reply_markup=ReplyKeyboardMarkup([['✅ Yes, continue', '❌ No, Upload As Is'], ['❌ Cancel']], resize_keyboard=True))
    return States.CONFIRM


async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the user if they want to proceed with editing or upload as is."""
    if 'Yes' in update.message.text:
        # Proceed to the first step of editing: image watermark
        await update.message.reply_text('Do you want to add an image watermark?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
        return States.ASK_IMAGE_WATERMARK
    else:  # User chose 'No, Upload As Is'
        # Skip all editing and go straight to the final processing step
        context.user_data['combined_files'] = context.user_data['processed']
        # Directly call the next handler instead of returning a state
        return await upload.start_final_processing(update, context)
