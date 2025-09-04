import asyncio
import logging
import os

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InputMediaVideo
from telegram.ext import ContextTypes

from add_video_effects import EffectsEngine
from state_machine import States
from handlers.common import is_video_file, cancel

# --- Main Effect Selection Handlers ---

async def ask_video_effects(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Asks the user to select video effects."""
    if 'selected_effects' not in context.user_data:
        context.user_data['selected_effects'] = []
    return await _return_to_effects_menu(update, context)


async def choose_effects(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles user's selection of video effects, branching to sub-conversations for parameterized effects."""
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

    # --- Parameterized Effect Routing ---
    parameterized_effects = {
        'Cinematic Color (LUT)': (States.ASK_LUT_TYPE, [['📁 Built-in', '📤 Upload Custom']]),
        'Ken Burns': (States.ASK_KENBURNS_LEVEL, [['Low', 'Medium', 'High']]),
        'Contrast / Brightness': (States.ASK_CONTRAST_LEVEL, [['Low', 'Medium', 'High']]),
        'Color Saturation': (States.ASK_SATURATION_LEVEL, [['Low', 'Medium', 'High']]),
        'Chromatic Aberration': (States.ASK_ABERRATION_LEVEL, [['Low', 'Medium', 'High']]),
        'Pixelated Effect': (States.ASK_PIXELATE_LEVEL, [['Low', 'Medium', 'High']]),
        'Speed Control': (States.ASK_SPEED_LEVEL, [['low', 'medium', 'high']]),
        'Rotate': (States.ASK_ROTATE_OPTION, [['15°', '45°', '90°']]),
        'Film Grain': (States.ASK_GRAIN_LEVEL, [['Low', 'Medium', 'High']]),
        'Glitch': (States.ASK_GLITCH_LEVEL, [['Low', 'Medium', 'High']]),
        'Rolling Shutter': (States.ASK_SHUTTER_LEVEL, [['Low', 'Medium', 'High']]),
        'Neon Glow': (States.ASK_NEON_LEVEL, [['Low', 'Medium', 'High']]),
        'Cartoon / Painterly': (States.ASK_CARTOON_LEVEL, [['Subtle', 'Normal', 'Strong']]),
        'Vignette': (States.ASK_VIGNETTE_LEVEL, [['Low', 'Medium', 'High']]),
        'Fade In/Out': (States.ASK_FADE_DURATION, [['Short', 'Medium', 'Long']]),
    }

    if choice in parameterized_effects:
        state, options = parameterized_effects[choice]
        keyboard = options + [['❌ Cancel']]
        await update.message.reply_text(
            f"Please choose a level for {choice}:",
            reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
        )
        return state

    # --- Standard (non-parameterized) effect selection ---
    if any(eff == choice or (isinstance(eff, tuple) and eff[0] == choice) for eff in selected):
        selected = [eff for eff in selected if not (eff == choice or (isinstance(eff, tuple) and eff[0] == choice))]
        context.user_data['selected_effects'] = selected
    elif len(selected) < 3:
        selected.append(choice)
        context.user_data['selected_effects'] = selected
    else:
        await update.message.reply_text("You can only select up to 3 effects.")

    return await _return_to_effects_menu(update, context)


async def _return_to_effects_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Displays the main effects menu and current selections without clearing them."""
    effects_list = [
        'Cinematic Color (LUT)', 'Ken Burns', 'Black & White', 
        'Color Saturation', 'Contrast / Brightness', 'Chromatic Aberration', 
        'Pixelated Effect', 'Invert Colors', 'Speed Control', 
        'Rotate', 'VHS Look', 'Film Grain', 
        'Glitch', 'Rolling Shutter', 'Neon Glow', 
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
            if isinstance(eff, tuple):
                # Format with parameter, e.g., "Contrast / Brightness (high)"
                name = f"{eff[0]} ({eff[1]})"
                # For LUT, show filename
                if eff[0] == 'Cinematic Color (LUT)':
                    name = f"{eff[0]} ({os.path.basename(eff[1])}, {eff[2]})"
                effect_lines.append(f"{i+1}. {name}")
            else:
                effect_lines.append(f"{i+1}. {eff}")
        effect_text = "Current effects:\n" + "\n".join(effect_lines)

    await update.message.reply_text(
        f"{effect_text}\n\n"
        "Select another effect, click an existing one to remove it, or press 'Done Selecting'.",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )
    return States.CHOOSE_EFFECTS

# --- Sub-conversation Handlers ---

def create_level_setter(effect_name: str, option_map: dict, default_value: str):
    """A factory to create a handler for setting an effect's level."""
    async def level_setter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        choice = update.message.text
        level = option_map.get(choice, default_value)
        selected = context.user_data.get('selected_effects', [])
        selected = [eff for eff in selected if not (isinstance(eff, tuple) and eff[0] == effect_name)]
        selected.append((effect_name, level))
        context.user_data['selected_effects'] = selected
        return await _return_to_effects_menu(update, context)
    return level_setter

# Create handlers for each parameterized effect
set_contrast_level = create_level_setter('Contrast / Brightness', {'Low': 'low', 'Medium': 'medium', 'High': 'high'}, 'medium')
set_saturation_level = create_level_setter('Color Saturation', {'Low': 'low', 'Medium': 'medium', 'High': 'high'}, 'medium')
set_aberration_level = create_level_setter('Chromatic Aberration', {'Low': 'low', 'Medium': 'medium', 'High': 'high'}, 'medium')
set_pixelate_level = create_level_setter('Pixelated Effect', {'Low': 'low', 'Medium': 'medium', 'High': 'high'}, 'medium')
set_speed_level = create_level_setter('Speed Control', {'low': 'low', 'medium': 'medium', 'high': 'high'}, 'medium')
set_rotate_option = create_level_setter('Rotate', {'15°': 'low', '45°': 'medium', '90°': 'high'}, 'high')
set_grain_level = create_level_setter('Film Grain', {'Low': 'low', 'Medium': 'medium', 'High': 'high'}, 'medium')
set_glitch_level = create_level_setter('Glitch', {'Low': 'low', 'Medium': 'medium', 'High': 'high'}, 'medium')
set_shutter_level = create_level_setter('Rolling Shutter', {'Low': 'low', 'Medium': 'medium', 'High': 'high'}, 'medium')
set_neon_level = create_level_setter('Neon Glow', {'Low': 'low', 'Medium': 'medium', 'High': 'high'}, 'medium')
set_cartoon_level = create_level_setter('Cartoon / Painterly', {'Subtle': 'low', 'Normal': 'medium', 'Strong': 'high'}, 'medium')
set_vignette_level = create_level_setter('Vignette', {'Low': 'low', 'Medium': 'medium', 'High': 'high'}, 'medium')
set_fade_duration = create_level_setter('Fade In/Out', {'Short': 'low', 'Medium': 'medium', 'Long': 'high'}, 'medium')
set_kenburns_level = create_level_setter('Ken Burns', {'Low': 'low', 'Medium': 'medium', 'High': 'high'}, 'medium')

# LUT Handlers
async def ask_lut_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    choice = update.message.text
    if 'Built-in' in choice:
        lut_dir = 'assets/luts'
        if not os.path.exists(lut_dir) or not os.listdir(lut_dir):
            await update.message.reply_text("Sorry, there are no built-in LUTs available.")
            return await _return_to_effects_menu(update, context)
        luts = [f.replace('.cube', '') for f in os.listdir(lut_dir) if f.endswith('.cube')]
        keyboard = [[lut] for lut in luts] + [['❌ Cancel']]
        await update.message.reply_text("Please choose a built-in LUT:", reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True))
        return States.CHOOSE_BUILTIN_LUT
    elif 'Upload' in choice:
        await update.message.reply_text("Please upload your .cube file as a document.")
        return States.RECEIVE_LUT_FILE
    return await cancel(update, context)

async def choose_builtin_lut(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lut_name = update.message.text
    lut_path = os.path.join('assets/luts', f"{lut_name}.cube")
    if not os.path.exists(lut_path):
        await update.message.reply_text("Invalid selection.")
        return await _return_to_effects_menu(update, context)
    context.user_data['current_lut_path'] = lut_path
    keyboard = [['Low', 'Medium', 'High'], ['❌ Cancel']]
    await update.message.reply_text("Please choose the blend level for the LUT:", reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True))
    return States.ASK_LUT_BLEND_LEVEL

async def receive_lut_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message.document or not update.message.document.file_name.lower().endswith('.cube'):
        await update.message.reply_text("That's not a .cube file. Please upload a valid LUT file.")
        return States.RECEIVE_LUT_FILE
    doc = await update.message.document.get_file()
    downloads_path = context.application.bot_data['downloads_path']
    lut_path = os.path.join(downloads_path, f"custom_{doc.file_id}.cube")
    await doc.download_to_drive(lut_path)
    context.user_data['current_lut_path'] = lut_path
    keyboard = [['Low', 'Medium', 'High'], ['❌ Cancel']]
    await update.message.reply_text("Please choose the blend level for the LUT:", reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True))
    return States.ASK_LUT_BLEND_LEVEL

async def set_lut_blend_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    level = update.message.text.lower()
    if level not in ['low', 'medium', 'high']:
        await update.message.reply_text("Invalid level. Please choose Low, Medium, or High.")
        return States.ASK_LUT_BLEND_LEVEL
    
    lut_path = context.user_data.get('current_lut_path')
    if not lut_path:
        await update.message.reply_text("Error: LUT path not found. Returning to menu.")
        return await _return_to_effects_menu(update, context)

    selected = context.user_data.get('selected_effects', [])
    selected = [eff for eff in selected if not (isinstance(eff, tuple) and eff[0] == 'Cinematic Color (LUT)')]
    selected.append(('Cinematic Color (LUT)', lut_path, level))
    context.user_data['selected_effects'] = selected
    
    return await _return_to_effects_menu(update, context)

# --- Final Processing Handlers ---

async def handle_render_quality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    choice = update.message.text
    quality = 'draft' if 'Draft' in choice else 'final'
    effect_names = [eff[0] if isinstance(eff, tuple) else eff for eff in context.user_data.get('selected_effects', [])]
    await update.message.reply_text(f"Applying effects: {', '.join(effect_names)}. Please wait, this may take a moment...", reply_markup=ReplyKeyboardRemove())
    return await process_and_confirm_effects(update, context, quality=quality)

async def process_and_confirm_effects(update: Update, context: ContextTypes.DEFAULT_TYPE, quality: str = 'final') -> int:
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
                    output_path=output_path,
                    quality=quality
                )
                effects_applied_files.append(path)
            except Exception as e:
                logging.error(f"Error applying effects to {file_path}: {e}")
                await update.message.reply_text(f"❌ An error occurred while applying effects to {os.path.basename(file_path)}.")
                effects_applied_files.append(file_path)
        else:
            effects_applied_files.append(file_path)
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
    if 'Yes' in update.message.text:
        context.user_data['final_files'] = context.user_data['final_files_with_effects']
        await update.message.reply_text('Effects confirmed. Please enter the final caption.', reply_markup=ReplyKeyboardRemove())
        return States.CAPTION
    else:
        await update.message.reply_text("Restarting effect selection...")
        return await ask_video_effects(update, context)
