import asyncio
import logging
import os
from PIL import Image

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ContextTypes

from state_machine import States
from watermark_engine import WatermarkEngine
from handlers.common import get_media_dimensions, is_video_file, cancel
from handlers import upload

# --- Image Watermark Handlers ---

async def ask_image_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the user if they want to add an image watermark."""
    if update.message.text == 'Yes':
        await update.message.reply_text('Please send the watermark image file.', reply_markup=ReplyKeyboardRemove())
        return States.RECEIVE_IMAGE_WATERMARK
    else:
        # If no, proceed to ask about text watermark
        return await ask_text_watermark(update, context)


async def receive_image_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives and validates the watermark image from the user."""
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
    kb = [['top-left', 'top-center', 'top-right'],
          ['middle-left', 'middle-center', 'middle-right'],
          ['bottom-left', 'bottom-center', 'bottom-right'],
          ['❌ Cancel']]
    await update.message.reply_text('Choose watermark position:', reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True))
    return States.CHOOSE_IMG_WATERMARK_POSITION


async def handle_img_position(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's choice for image watermark position."""
    context.user_data['img_watermark_position'] = update.message.text.lower()
    keyboard = [['50', '60', '70'], ['80', '90', '100'], ['❌ Cancel']]
    await update.message.reply_text('Choose scale (50-100%):', reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True))
    return States.CHOOSE_IMG_WATERMARK_SCALE


async def handle_img_scale(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's choice for image watermark scale."""
    context.user_data['img_watermark_scale'] = int(update.message.text)
    keyboard = [['100', '90', '80'], ['70', '60', '50'], ['❌ Cancel']]
    await update.message.reply_text('Choose opacity (50-100%):', reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True))
    return States.CHOOSE_IMG_WATERMARK_OPACITY


async def generate_and_preview_image_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generates a preview of the image watermark and asks for confirmation."""
    context.user_data['img_watermark_opacity'] = int(update.message.text)
    await update.message.reply_text('⏳ Generating preview...', reply_markup=ReplyKeyboardRemove())

    media_dims = get_media_dimensions(context.user_data['processed'][0])
    if not media_dims:
        await update.message.reply_text('Error: Could not get media dimensions.')
        return await cancel(update, context)

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
        await update.message.reply_text(f"Error creating watermark preview: {e}")
        return await cancel(update, context)


async def handle_img_watermark_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Applies the image watermark to all media files if confirmed by the user."""
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
    # Proceed to ask about text watermark
    return await ask_text_watermark(update, context)


# --- Text Watermark Handlers ---

async def ask_text_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the user if they want to add a text watermark."""
    await update.message.reply_text('Do you want to add a text watermark?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
    return States.ASK_TEXT_WATERMARK


async def handle_ask_text_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's decision to add a text watermark or not."""
    if 'Yes' in update.message.text:
        await update.message.reply_text('Please enter the text for the watermark.', reply_markup=ReplyKeyboardRemove())
        return States.RECEIVE_TEXT
    else:
        # Finished with watermarks, ask the next question and transition
        has_video = any(is_video_file(p) for p in context.user_data.get('processed', []))
        if not has_video:
            logging.info("No videos in batch, skipping music step.")
            await update.message.reply_text("No videos found, skipping music step.")
            return await upload.combine_changes(update, context)
        
        await update.message.reply_text('Do you want to add music to the video(s)?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
        return States.ASK_ADD_MUSIC


async def receive_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives the text for the watermark."""
    if update.message.text == '❌ Cancel': return await cancel(update, context)
    context.user_data['text_watermark_text'] = update.message.text
    
    font_names = [os.path.basename(f) for f in context.application.bot_data.get('font_files', [])]
    if not font_names:
        if context.application.bot_data.get('font_warning'):
            await update.message.reply_text(context.application.bot_data['font_warning'])
        
        # No fonts, so can't add text watermark. Skip to next step.
        has_video = any(is_video_file(p) for p in context.user_data.get('processed', []))
        if not has_video:
            logging.info("No videos in batch, skipping music step.")
            await update.message.reply_text("No videos found, skipping music step.")
            return await upload.combine_changes(update, context)
            
        await update.message.reply_text('Do you want to add music to the video(s)?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
        return States.ASK_ADD_MUSIC
        
    keyboard = [[name] for name in font_names]
    keyboard.append(['❌ Cancel'])
    await update.message.reply_text('Choose a font:', reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True))
    return States.CHOOSE_FONT


async def handle_font(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's font choice."""
    if update.message.text == '❌ Cancel': return await cancel(update, context)
    context.user_data['text_watermark_font'] = update.message.text
    keyboard = [['10', '15', '20'], ['25', '30', '35'], ['40', '45', '50'], ['❌ Cancel']]
    await update.message.reply_text('Choose font size (10-50):', reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True))
    return States.CHOOSE_FONT_SIZE


async def handle_font_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's font size choice."""
    context.user_data['text_watermark_size'] = int(update.message.text)
    colors = [['White', 'Black', 'Red'], ['Blue', 'Yellow', 'Green'], ['❌ Cancel']]
    await update.message.reply_text('Choose a color:', reply_markup=ReplyKeyboardMarkup(colors, one_time_keyboard=True))
    return States.CHOOSE_COLOR


async def handle_color(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's color choice."""
    context.user_data['text_watermark_color'] = update.message.text
    positions = [['top–center'], ['middle–center'], ['bottom–center'], ['❌ Cancel']]
    await update.message.reply_text('Choose text position:', reply_markup=ReplyKeyboardMarkup(positions, one_time_keyboard=True))
    return States.CHOOSE_TEXT_POSITION


async def generate_and_preview_text_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generates a preview of the text watermark and asks for confirmation."""
    context.user_data['text_watermark_position'] = update.message.text.lower()
    await update.message.reply_text('⏳ Generating preview...', reply_markup=ReplyKeyboardRemove())

    media_dims = get_media_dimensions(context.user_data['processed'][0])
    if not media_dims:
        await update.message.reply_text('Error: Could not get media dimensions.')
        return await cancel(update, context)

    font_name = context.user_data['text_watermark_font']
    font_path = next((f for f in context.application.bot_data['font_files'] if os.path.basename(f) == font_name), None)
    if not font_path:
        await update.message.reply_text(f"Error: Font '{font_name}' not found.")
        return States.ASK_ADD_MUSIC

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
        await update.message.reply_text(f"Error creating watermark preview: {e}")
        return await cancel(update, context)


async def handle_text_watermark_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Applies the text watermark to all media files if confirmed by the user."""
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

    # Finished with watermarks, ask the next question and transition
    has_video = any(is_video_file(p) for p in context.user_data.get('processed', []))
    if not has_video:
        logging.info("No videos in batch, skipping music step.")
        await update.message.reply_text("No videos found, skipping music step.")
        return await upload.combine_changes(update, context)
        
    await update.message.reply_text('Do you want to add music to the video(s)?', reply_markup=ReplyKeyboardMarkup([['Yes', 'No'], ['❌ Cancel']], one_time_keyboard=True))
    return States.ASK_ADD_MUSIC
