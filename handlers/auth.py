import asyncio
import logging
from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler

from auth_manager import AuthManager
from state_machine import States
from handlers.common import send_welcome_message, cancel

MAX_AUTH_ATTEMPTS = 3


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the /start command and initiates the authentication process."""
    logging.info("'/start' command received. Initiating authentication.")
    context.user_data['auth_attempts'] = 0
    ig_manager: AuthManager = context.application.bot_data['ig_manager']

    # Use asyncio.to_thread for the blocking login call
    success, status = await asyncio.to_thread(ig_manager.login)

    if success:
        await update.message.reply_text("✅ Connection to Telegram and Instagram is successful.")
        return await send_welcome_message(update, context)
    if status == "2FA_REQUIRED":
        await update.message.reply_text("🔐 Please enter your 2FA code (from your authenticator app).")
        return States.AUTH_2FA
    elif status == "SMS_REQUIRED":
        await update.message.reply_text("📱 Please enter the SMS code sent to your phone.")
        return States.AUTH_SMS
    else:
        await update.message.reply_text(f"❌ Instagram login failed: {ig_manager.login_error_message}")
        return ConversationHandler.END


async def handle_2fa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the 2FA code provided by the user."""
    if update.message.text == '❌ Cancel':
        return await cancel(update, context)

    context.user_data['auth_attempts'] += 1
    if context.user_data.get('auth_attempts', 0) > MAX_AUTH_ATTEMPTS:
        await update.message.reply_text("❌ Too many incorrect attempts. Halting operation.")
        return ConversationHandler.END

    code = update.message.text.strip()
    ig_manager: AuthManager = context.application.bot_data['ig_manager']
    success, status = await asyncio.to_thread(ig_manager.login, two_factor_code=code)

    if success:
        await update.message.reply_text("✅ Instagram connection successful!")
        return await send_welcome_message(update, context)
    else:
        remaining_attempts = MAX_AUTH_ATTEMPTS - context.user_data['auth_attempts']
        await update.message.reply_text(f"❌ Incorrect 2FA code. Please try again. ({remaining_attempts} attempts remaining)")
        return States.AUTH_2FA


async def handle_sms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles the SMS verification code provided by the user."""
    if update.message.text == '❌ Cancel':
        return await cancel(update, context)

    context.user_data['auth_attempts'] += 1
    if context.user_data.get('auth_attempts', 0) > MAX_AUTH_ATTEMPTS:
        await update.message.reply_text("❌ Too many incorrect attempts. Halting operation.")
        return ConversationHandler.END

    code = update.message.text.strip()
    ig_manager: AuthManager = context.application.bot_data['ig_manager']
    success, status = await asyncio.to_thread(ig_manager.login, verification_code=code)

    if success:
        await update.message.reply_text("✅ Instagram connection successful!")
        return await send_welcome_message(update, context)
    else:
        remaining_attempts = MAX_AUTH_ATTEMPTS - context.user_data['auth_attempts']
        await update.message.reply_text(f"❌ Incorrect SMS code. Please try again. ({remaining_attempts} attempts remaining)")
        return States.AUTH_SMS
