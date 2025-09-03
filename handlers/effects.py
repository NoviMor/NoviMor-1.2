import asyncio
import logging
import os

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InputMediaVideo
from telegram.ext import ContextTypes

from add_video_effects import EffectsEngine
from state_machine import States
from handlers.common import is_video_file, cancel

async def ask_video_effects(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the user to select video effects."""
    context.user_data['selected_effects'] = []
    effects_list = [
        'Cinematic Color (LUT)', 'Ken Burns', # New effects will go here
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
    if choice == '❌ Cancel':
        return await cancel(update, context)
        
    selected = context.user_data.get('selected_effects', [])

    if 'Done' in choice:
        if not selected:
            await update.message.reply_text("No effects selected. Please enter the final caption.", reply_markup=ReplyKeyboardRemove())
            return States.CAPTION
        else:
            keyboard = [['🚀 High Quality', '⚡️ Draft Preview'], ['❌ Cancel']]
            await update.message.reply_text(
                "How would you like to render the preview?\n\n"
                "🚀 **High Quality:** Slower, but shows the final result.\n"
                "⚡️ **Draft Preview:** Much faster, but lower quality.",
                reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True),
                parse_mode='Markdown'
            )
            return States.ASK_RENDER_QUALITY

    # --- Sub-conversation for LUT ---
    if choice == 'Cinematic Color (LUT)':
        keyboard = [['📁 Built-in', '📤 Upload Custom'], ['❌ Cancel']]
        await update.message.reply_text(
            "Do you want to use a built-in cinematic LUT or upload your own .cube file?",
            reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
        )
        return States.ASK_LUT_TYPE

    # --- Sub-conversation for Contrast ---
    if choice == 'Contrast / Brightness':
        keyboard = [['Low', 'Medium', 'High'], ['❌ Cancel']]
        await update.message.reply_text(
            "Please choose a level for Contrast / Brightness:",
            reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
        )
        return States.ASK_CONTRAST_LEVEL

    # --- Standard effect selection ---
    # To handle tuples (effect, param) vs strings
    if any(eff == choice or (isinstance(eff, tuple) and eff[0] == choice) for eff in selected):
        # Remove the effect
        selected = [eff for eff in selected if not (eff == choice or (isinstance(eff, tuple) and eff[0] == choice))]
        context.user_data['selected_effects'] = selected
        effect_names = [eff[0] if isinstance(eff, tuple) else eff for eff in selected]
        await update.message.reply_text(f"Removed '{choice}'. Current effects: {', '.join(effect_names) if effect_names else 'None'}.")
    elif len(selected) < 3:
        # Add the effect (as a string, parameters will be added in sub-conversations)
        selected.append(choice)
        context.user_data['selected_effects'] = selected
        effect_names = [eff[0] if isinstance(eff, tuple) else eff for eff in selected]
        await update.message.reply_text(f"Added '{choice}'. Current effects: {', '.join(effect_names)}.")
    else:
        await update.message.reply_text("You can only select up to 3 effects.")

    return States.CHOOSE_EFFECTS

async def _return_to_effects_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Displays the main effects menu and current selections without clearing them."""
    effects_list = [
        'Cinematic Color (LUT)', 'Ken Burns',
        'Black & White', 'Color Saturation', 'Contrast / Brightness',
        'Chromatic Aberration', 'Pixelated Effect',
        'Invert Colors', 'Speed Control', 'Rotate',
        'VHS Look', 'Film Grain', 'Glitch',
        'Rolling Shutter', 'Neon Glow',
        'Cartoon / Painterly', 'Vignette', 'Fade In/Out'
    ]
    keyboard = [effects_list[i:i + 3] for i in range(0, len(effects_list), 3)]
    keyboard.append(['✅ Done Selecting', '❌ Cancel'])
    
    selected = context.user_data.get('selected_effects', [])
    
    if not selected:
        effect_text = "Current effects: None."
    else:
        effect_lines = []
        for i, eff in enumerate(selected):
            name = f"{eff[0]} ({eff[1]})" if isinstance(eff, tuple) else eff
            effect_lines.append(f"{i+1}. {name}")
        effect_text = "Current effects:\n" + "\n".join(effect_lines)

    await update.message.reply_text(
        f"{effect_text}\n\n"
        "Select another effect to add to the list, click an existing one to remove it, or press 'Done Selecting'.",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )
    return States.CHOOSE_EFFECTS

# --- LUT Sub-conversation Handlers ---

async def ask_lut_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's choice of built-in vs. custom LUT."""
    choice = update.message.text
    if 'Built-in' in choice:
        lut_dir = 'assets/luts'
        if not os.path.exists(lut_dir) or not os.listdir(lut_dir):
            await update.message.reply_text("Sorry, there are no built-in LUTs available at the moment.")
            return await ask_video_effects(update, context) # Go back to main effects menu
        
        luts = [f for f in os.listdir(lut_dir) if f.endswith('.cube')]
        keyboard = [[lut.replace('.cube', '')] for lut in luts]
        keyboard.append(['❌ Cancel'])
        await update.message.reply_text(
            "Please choose a built-in LUT:",
            reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
        )
        return States.CHOOSE_BUILTIN_LUT
    elif 'Upload' in choice:
        await update.message.reply_text("Please upload your .cube file as a document.")
        return States.RECEIVE_LUT_FILE
    else:
        return await cancel(update, context)

async def choose_builtin_lut(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's selection from the list of built-in LUTs."""
    lut_name = update.message.text
    lut_path = os.path.join('assets/luts', f"{lut_name}.cube")

    if not os.path.exists(lut_path):
        await update.message.reply_text("Invalid selection. Please try again.")
        return await _return_to_effects_menu(update, context)

    selected = context.user_data.get('selected_effects', [])
    selected.append(('Cinematic Color (LUT)', lut_path))
    
    # Go back to the main effects menu to select more or finish
    return await _return_to_effects_menu(update, context)

async def receive_lut_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receives and validates a custom .cube file from the user."""
    if not update.message.document or not update.message.document.file_name.lower().endswith('.cube'):
        await update.message.reply_text("That's not a .cube file. Please upload a valid LUT file as a document.")
        return States.RECEIVE_LUT_FILE

    doc = await update.message.document.get_file()
    downloads_path = context.application.bot_data['downloads_path']
    lut_path = os.path.join(downloads_path, f"custom_{doc.file_id}.cube")
    await doc.download_to_drive(lut_path)

    selected = context.user_data.get('selected_effects', [])
    selected.append(('Cinematic Color (LUT)', lut_path))

    # Go back to the main effects menu
    return await _return_to_effects_menu(update, context)

# --- Parameterization Handlers ---

async def set_contrast_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Sets the chosen level for the Contrast/Brightness effect."""
    level = update.message.text.lower()
    if level not in ['low', 'medium', 'high']:
        await update.message.reply_text("Invalid level. Please choose Low, Medium, or High.")
        return States.ASK_CONTRAST_LEVEL

    selected = context.user_data.get('selected_effects', [])
    # Remove if already exists to prevent duplicates
    selected = [eff for eff in selected if not (isinstance(eff, tuple) and eff[0] == 'Contrast / Brightness')]
    selected.append(('Contrast / Brightness', level))
    context.user_data['selected_effects'] = selected
    
    return await _return_to_effects_menu(update, context)


async def handle_render_quality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the user's quality choice and starts the rendering process."""
    choice = update.message.text
    quality = 'draft' if 'Draft' in choice else 'final'
    
    effect_names = [eff[0] if isinstance(eff, tuple) else eff for eff in context.user_data.get('selected_effects', [])]
    await update.message.reply_text(f"Applying effects: {', '.join(effect_names)}. Please wait, this may take a moment...", reply_markup=ReplyKeyboardRemove())

    return await process_and_confirm_effects(update, context, quality=quality)


async def process_and_confirm_effects(update: Update, context: ContextTypes.DEFAULT_TYPE, quality: str = 'final') -> int:
    """Applies the selected effects and sends a preview for confirmation."""
    engine = EffectsEngine()
    effects_applied_files = []
    
    for i, file_path in enumerate(context.user_data['final_files']):
        if is_video_file(file_path):
            output_path = os.path.join(context.application.bot_data['downloads_path'], f"effects_{i}_{os.path.basename(file_path)}")
            try:
                # This should be run in a separate thread to avoid blocking
                path = await asyncio.to_thread(
                    engine.apply_effects_in_sequence,
                    video_path=file_path,
                    effects=context.user_data['selected_effects'],
                    output_path=output_path,
                    quality=quality # Pass the quality parameter
                )
                effects_applied_files.append(path)
            except Exception as e:
                logging.error(f"Error applying effects to {file_path}: {e}")
                await update.message.reply_text(f"❌ An error occurred while applying effects to {os.path.basename(file_path)}.")
                effects_applied_files.append(file_path)  # Append original if effect fails
        else:
            effects_applied_files.append(file_path)  # Keep non-video files as they are

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
    else:  # 'No, restart effects'
        await update.message.reply_text("Restarting effect selection...")
        return await ask_video_effects(update, context)
