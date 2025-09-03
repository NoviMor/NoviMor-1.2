from telegram.ext import MessageHandler, CommandHandler, filters, ConversationHandler

from state_machine import States
from handlers import auth, common, media, watermark, music, effects, upload

def get_conversation_handler() -> ConversationHandler:
    """
    Builds the main conversation handler by assembling handlers from sub-modules.
    """
    return ConversationHandler(
        entry_points=[CommandHandler('start', auth.start)],
        states={
            # A state to handle restarting the conversation
            States.START: [MessageHandler(filters.ALL, auth.start)],

            # Authentication
            States.AUTH_2FA: [MessageHandler(filters.TEXT & ~filters.COMMAND, auth.handle_2fa)],
            States.AUTH_SMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, auth.handle_sms)],

            # Media Handling
            States.MEDIA_TYPE: [MessageHandler(filters.Regex('^📤 Album$|^📎 Single$'), media.handle_media_type)],
            States.RECEIVE_MEDIA: [
                MessageHandler(filters.PHOTO | filters.VIDEO | filters.ANIMATION, media.handle_media),
                MessageHandler(filters.TEXT & filters.Regex(r'^🏁 Done$'), media.process_media)
            ],
            States.CONFIRM: [MessageHandler(filters.Regex('^✅ Yes, continue$|^❌ No, Upload As Is$'), media.handle_confirmation)],
            
            # Image Watermark
            States.ASK_IMAGE_WATERMARK: [MessageHandler(filters.Regex('^Yes$|^No$'), watermark.ask_image_watermark)],
            States.RECEIVE_IMAGE_WATERMARK: [MessageHandler(filters.PHOTO, watermark.receive_image_watermark)],
            States.CHOOSE_IMG_WATERMARK_POSITION: [MessageHandler(filters.Regex('^(top|middle|bottom)-(left|center|right)$'), watermark.handle_img_position)],
            States.CHOOSE_IMG_WATERMARK_SCALE: [MessageHandler(filters.Regex(r'^\d+$'), watermark.handle_img_scale)],
            States.CHOOSE_IMG_WATERMARK_OPACITY: [MessageHandler(filters.Regex(r'^\d+$'), watermark.generate_and_preview_image_watermark)],
            States.CONFIRM_IMG_WATERMARK: [MessageHandler(filters.Regex('^✅ Yes, Confirm$|^❌ No, Retry$'), watermark.handle_img_watermark_confirmation)],
            
            # Text Watermark
            States.ASK_TEXT_WATERMARK: [MessageHandler(filters.Regex('^Yes$|^No$'), watermark.handle_ask_text_watermark)],
            States.RECEIVE_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, watermark.receive_text)],
            States.CHOOSE_FONT: [MessageHandler(filters.TEXT & ~filters.COMMAND, watermark.handle_font)],
            States.CHOOSE_FONT_SIZE: [MessageHandler(filters.Regex(r'^\d+$'), watermark.handle_font_size)],
            States.CHOOSE_COLOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, watermark.handle_color)],
            States.CHOOSE_TEXT_POSITION: [MessageHandler(filters.Regex('^top–center$|^middle–center$|^bottom–center$'), watermark.generate_and_preview_text_watermark)],
            States.CONFIRM_TEXT_WATERMARK: [MessageHandler(filters.Regex('^✅ Yes, Confirm$|^❌ No, Retry$'), watermark.handle_text_watermark_confirmation)],
            
            # Music
            States.ASK_ADD_MUSIC: [MessageHandler(filters.Regex('^Yes$|^No$'), music.ask_add_music)],
            States.RECEIVE_MUSIC: [MessageHandler(filters.AUDIO, music.receive_music)],
            States.RECEIVE_MUSIC_START_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, music.receive_music_start_time)],
            States.CONFIRM_MUSIC: [MessageHandler(filters.Regex('^✅ Yes, Confirm$|^❌ No, Retry$'), music.handle_music_confirmation)],
            
            # Combination & Final Processing
            States.CONFIRM_COMBINED_MEDIA: [MessageHandler(filters.Regex('^✅ Yes, continue$|^❌ No, restart edits$'), upload.handle_combined_media_confirmation)],
            States.CONFIRM_FINAL_MEDIA: [MessageHandler(filters.Regex('^✅ Yes, looks good$|^❌ No, restart edits$|^Add Video Effects$'), upload.handle_final_confirmation)],
            
            # Video Effects
            States.ASK_VIDEO_EFFECTS: [MessageHandler(filters.Regex('Add Video Effects'), effects.ask_video_effects)],
            States.CHOOSE_EFFECTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, effects.choose_effects)],
            States.CONFIRM_EFFECTS: [MessageHandler(filters.Regex('^✅ Yes, upload$|^❌ No, restart effects$'), effects.handle_effects_confirmation)],

            # LUT Sub-conversation
            States.ASK_LUT_TYPE: [MessageHandler(filters.Regex('^📁 Built-in$|^📤 Upload Custom$'), effects.ask_lut_type)],
            States.CHOOSE_BUILTIN_LUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, effects.choose_builtin_lut)],
            States.RECEIVE_LUT_FILE: [MessageHandler(filters.Document.ALL, effects.receive_lut_file)],

            # Parameterization Sub-conversations
            States.ASK_CONTRAST_LEVEL: [MessageHandler(filters.Regex('^Low$|^Medium$|^High$'), effects.set_contrast_level)],

            # Render Quality
            States.ASK_RENDER_QUALITY: [MessageHandler(filters.Regex('^🚀 High Quality$|^⚡️ Draft Preview$'), effects.handle_render_quality)],
            
            # Finalize
            States.CAPTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, upload.handle_caption_and_upload)],
        },
        fallbacks=[
            CommandHandler('cancel', common.cancel),
            MessageHandler(filters.Regex('^❌ Cancel$'), common.cancel)
        ],
        conversation_timeout=1440,  # 24 minutes
        allow_reentry=True
    )
