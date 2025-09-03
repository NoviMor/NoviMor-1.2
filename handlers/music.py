import asyncio
import logging
import os

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ContextTypes

from add_music_to_video import MusicAdder
from state_machine import States
from handlers.common import get_video_duration, is_video_file, cancel
from handlers import upload

async def ask_add_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles user's decision to add music or not."""
    if 'Yes' in update.message.text:
        await update.message.reply_text('Please send the music file (as an audio file).', reply_markup=ReplyKeyboardRemove())
        return States.RECEIVE_MUSIC
    else:
        # User chose not to add music, so skip to the combine step
        return await upload.combine_changes(update, context)


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
    """Receives the music start time and generates a preview."""
    if update.message.text == '❌ Cancel':
        return await cancel(update, context)
        
    start_time_str = update.message.text
    context.user_data['music_start_time'] = start_time_str

    video_paths = [p for p in context.user_data.get('processed', []) if is_video_file(p)]
    if not video_paths:
        await update.message.reply_text("No videos found to add music to. Skipping music step.")
        return await upload.combine_changes(update, context)

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
        preview_path = os.path.join(context.application.bot_data['downloads_path'], 'S3_preview.mp3')
        if os.path.exists(preview_path):
            os.remove(preview_path)
        # Go back to the start of the music conversation
        await update.message.reply_text('Do you want to add music to the video(s)?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
        return States.ASK_ADD_MUSIC

    # Confirm that music should be added in the next step.
    context.user_data['music_confirmed'] = True
    await update.message.reply_text('✅ Music confirmed. It will be added to each video individually.')
    
    # Proceed to the combine step
    return await upload.combine_changes(update, context)
