import asyncio
import logging
import os

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InputMediaPhoto, InputMediaVideo
from telegram.ext import ContextTypes

from add_music_to_video import MusicAdder
from combine_user_changes import MediaCombiner
from image_processor import ImageProcessor
from state_machine import States
from video_processor import VideoProcessor
from handlers.common import get_video_duration, is_video_file, cancel, send_welcome_message
from handlers import effects

# --- Final Combination and Upload Handlers ---

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

        if music_confirmed and is_video_file(file_path):
            video_duration = get_video_duration(file_path)
            if video_duration:
                trimmed_audio_path = os.path.join(downloads_path, f"S3_{i+1}.mp3")
                try:
                    MusicAdder.trim_audio(
                        audio_path=context.user_data['music_path'],
                        video_duration=video_duration,
                        start_time_str=context.user_data['music_start_time'],
                        output_path=trimmed_audio_path
                    )
                    audio_for_this_video = trimmed_audio_path
                except Exception as e:
                    logging.error(f"Failed to trim audio for {file_path}: {e}")

        output_filename = f"combined_{i}_{os.path.basename(file_path)}"
        output_path = os.path.join(downloads_path, output_filename)

        try:
            path = await asyncio.to_thread(
                combiner.combine,
                base_path=file_path,
                output_path=output_path,
                s1_layer_path=s1,
                s2_layer_path=s2,
                s3_audio_path=audio_for_this_video
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
        await update.message.reply_text("Restarting editing process...")
        await update.message.reply_text('Do you want to add an image watermark?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
        return States.ASK_IMAGE_WATERMARK

    return await start_final_processing(update, context)


async def start_final_processing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Processes the combined media to their final dimensions and quality for Instagram."""
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
        # Directly call the effects handler to start the sub-conversation
        return await effects.ask_video_effects(update, context)
    else:  # 'looks good'
        await update.message.reply_text('Please enter the final caption for your post.', reply_markup=ReplyKeyboardRemove())
        return States.CAPTION


async def handle_caption_and_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives the caption and uploads the final media to Instagram."""
    if update.message.text == '❌ Cancel':
        return await cancel(update, context)
        
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
