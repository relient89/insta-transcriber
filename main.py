"""
Instagram Video Transcriber — Telegram Bot
Парсит Instagram-профили, скачивает видео, транскрибирует через OpenAI Whisper.
"""

import asyncio
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import instaloader
from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("insta-transcriber")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise ValueError(
        "BOT_TOKEN не задан. Установите переменную окружения BOT_TOKEN "
        "с токеном вашего Telegram-бота (@BotFather)."
    )
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY не задан. Установите переменную окружения OPENAI_API_KEY "
        "с ключом от OpenAI API (https://platform.openai.com/api-keys)."
    )

DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "insta_downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_TG_MESSAGE_LEN = 4096
WHISPER_MAX_SIZE = 25 * 1024 * 1024  # 25 MB
ANTI_SPAM_DELAY = 0.6  # seconds between Telegram messages

bot = Bot(token=BOT_TOKEN, default=types.DefaultBotProperties(parse_mode=ParseMode.HTML))
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
router = Router()

# In-memory per-user state: {user_id: {"username": str}}
user_data: dict[int, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Instagram service
# ---------------------------------------------------------------------------
INSTAGRAM_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?instagram\.com/([A-Za-z0-9_.]+)/?",
)


def get_instagram_videos(username: str, count: int) -> list[dict[str, Any]]:
    """Fetch latest *count* video posts from a public Instagram profile.

    Runs synchronously (intended for ``run_in_executor``).
    Returns a list of dicts with keys:
        video_url, shortcode, date, caption, duration
    """
    loader = instaloader.Instaloader(
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        quiet=True,
    )

    try:
        profile = instaloader.Profile.from_username(loader.context, username)
    except instaloader.exceptions.ProfileNotExistsException:
        raise ValueError(f"Профиль @{username} не найден.")
    except Exception as exc:
        raise ValueError(f"Ошибка при загрузке профиля @{username}: {exc}")

    if profile.is_private:
        raise ValueError(
            f"Профиль @{username} является приватным. "
            "Бот может работать только с публичными профилями."
        )

    videos: list[dict[str, Any]] = []
    for post in profile.get_posts():
        if len(videos) >= count:
            break
        if not post.is_video:
            continue
        caption = (post.caption or "")[:150]
        videos.append(
            {
                "video_url": post.video_url,
                "shortcode": post.shortcode,
                "date": post.date_utc.strftime("%Y-%m-%d %H:%M"),
                "caption": caption,
                "duration": post.video_duration or 0,
            }
        )

    if not videos:
        raise ValueError(f"У @{username} не найдено видео-постов.")

    return videos


# ---------------------------------------------------------------------------
# Downloader helpers
# ---------------------------------------------------------------------------
async def download_video(url: str) -> Path:
    """Download a video from *url* and return the local file path."""
    dest = DOWNLOAD_DIR / f"{uuid.uuid4().hex}.mp4"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 64):
                    f.write(chunk)
    log.info("Downloaded video: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


async def extract_audio(video_path: Path) -> Path:
    """Extract mono 16 kHz MP3 audio from *video_path* using FFmpeg."""
    audio_path = video_path.with_suffix(".mp3")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
        str(audio_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg завершился с кодом {proc.returncode}")
    log.info("Extracted audio: %s (%.1f MB)", audio_path.name, audio_path.stat().st_size / 1e6)
    return audio_path


async def trim_audio(audio_path: Path, max_seconds: int = 600) -> Path:
    """Trim audio to *max_seconds* if the file exceeds Whisper size limit."""
    trimmed = audio_path.with_stem(audio_path.stem + "_trimmed")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", str(audio_path),
        "-t", str(max_seconds),
        "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
        str(trimmed),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg trim failed")
    log.info("Trimmed audio to %d s: %s", max_seconds, trimmed.name)
    return trimmed


def cleanup(*paths: Path) -> None:
    """Silently remove temporary files."""
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
async def transcribe_audio(audio_path: Path) -> dict[str, Any]:
    """Send *audio_path* to OpenAI Whisper and return transcription data."""
    target = audio_path
    trimmed: Path | None = None

    if audio_path.stat().st_size > WHISPER_MAX_SIZE:
        log.warning("Audio > 25 MB — trimming to 10 min")
        trimmed = await trim_audio(audio_path, max_seconds=600)
        target = trimmed

    try:
        with open(target, "rb") as f:
            response = await openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language="ru",
            )
        return {
            "text": response.text,
            "duration": getattr(response, "duration", 0),
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in (getattr(response, "segments", None) or [])
            ],
        }
    finally:
        if trimmed:
            cleanup(trimmed)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
async def process_single_video(video_info: dict[str, Any]) -> dict[str, Any]:
    """Full pipeline for one video: download → audio → transcribe → cleanup."""
    video_path: Path | None = None
    audio_path: Path | None = None
    try:
        video_path = await download_video(video_info["video_url"])
        audio_path = await extract_audio(video_path)
        transcription = await transcribe_audio(audio_path)
        return {
            "status": "success",
            "shortcode": video_info["shortcode"],
            "date": video_info["date"],
            "caption": video_info["caption"],
            "duration": video_info["duration"],
            "transcription": transcription,
        }
    except Exception as exc:
        log.exception("Error processing video %s", video_info.get("shortcode", "?"))
        return {
            "status": "error",
            "shortcode": video_info.get("shortcode", "?"),
            "date": video_info.get("date", ""),
            "caption": video_info.get("caption", ""),
            "duration": video_info.get("duration", 0),
            "error": str(exc),
        }
    finally:
        paths_to_clean = [p for p in (video_path, audio_path) if p]
        cleanup(*paths_to_clean)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def format_result(result: dict[str, Any], index: int) -> str:
    """Format a single video result for Telegram (HTML)."""
    shortcode = result["shortcode"]
    date = result["date"]
    caption = _escape_html(result.get("caption", ""))
    dur = _format_duration(result.get("duration", 0))

    if result["status"] == "error":
        return (
            f"{'─' * 30}\n"
            f"🎬 <b>Видео {index}</b> — {date}\n"
            f"🔗 https://instagram.com/p/{shortcode}\n"
            f"❌ <b>Ошибка:</b> {_escape_html(result['error'])}\n"
        )

    tr = result["transcription"]
    text = _escape_html(tr["text"]).strip() or "<i>(речь не распознана)</i>"
    tr_dur = _format_duration(tr.get("duration", 0))

    return (
        f"{'─' * 30}\n"
        f"🎬 <b>Видео {index}</b> — {date}\n"
        f"⏱ Длительность: {dur} | Аудио: {tr_dur}\n"
        f"🔗 https://instagram.com/p/{shortcode}\n"
        f"📝 {caption}\n\n"
        f"📄 <b>Транскрипция:</b>\n{text}\n"
    )


def format_summary(results: list[dict[str, Any]], username: str) -> str:
    """Format a summary block for all results."""
    total = len(results)
    ok = sum(1 for r in results if r["status"] == "success")
    err = total - ok
    return (
        f"📊 <b>Итоги — @{_escape_html(username)}</b>\n"
        f"Всего видео: {total} | ✅ Успешно: {ok} | ❌ Ошибок: {err}\n"
    )


def progress_bar(done: int, total: int) -> str:
    filled = int(done / total * 10) if total else 0
    return "🟩" * filled + "⬜" * (10 - filled)


# ---------------------------------------------------------------------------
# Message splitter
# ---------------------------------------------------------------------------
def split_message(text: str, limit: int = MAX_TG_MESSAGE_LEN) -> list[str]:
    """Split *text* into chunks that fit within Telegram message limit."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 <b>Instagram Video Transcriber</b>\n\n"
        "Отправь мне ссылку на публичный Instagram-профиль, и я:\n"
        "1. Найду последние видео\n"
        "2. Скачаю и извлеку аудио\n"
        "3. Распознаю речь через OpenAI Whisper\n"
        "4. Верну текстовую транскрипцию\n\n"
        "Пример: <code>instagram.com/durov</code>",
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "ℹ️ <b>Как использовать</b>\n\n"
        "1. Отправь ссылку на Instagram-профиль\n"
        "   <code>https://instagram.com/username</code>\n"
        "2. Выбери количество видео для транскрипции\n"
        "3. Дождись результата\n\n"
        "⚠️ Бот работает только с <b>публичными</b> профилями.\n"
        "Транскрипция одного видео занимает ~30–60 секунд.\n\n"
        "Команды:\n"
        "/start — приветствие\n"
        "/help — эта справка",
    )


@router.message(F.text.regexp(INSTAGRAM_URL_RE))
async def handle_instagram_link(message: Message) -> None:
    match = INSTAGRAM_URL_RE.search(message.text or "")
    if not match:
        return
    username = match.group(1).rstrip("/")

    # Ignore service pages
    if username.lower() in ("p", "reel", "stories", "explore", "accounts", "about"):
        await message.answer("⚠️ Отправьте ссылку на <b>профиль</b>, а не на пост или страницу.")
        return

    user_data[message.from_user.id] = {"username": username}

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="3 видео (тест)", callback_data="count_3"),
                InlineKeyboardButton(text="10 видео", callback_data="count_10"),
                InlineKeyboardButton(text="20 видео", callback_data="count_20"),
            ]
        ]
    )
    await message.answer(
        f"🔍 Профиль: <b>@{_escape_html(username)}</b>\n"
        "Сколько последних видео транскрибировать?",
        reply_markup=keyboard,
    )


@router.callback_query(F.data.startswith("count_"))
async def handle_count_callback(callback: CallbackQuery) -> None:
    await callback.answer()

    uid = callback.from_user.id
    data = user_data.get(uid)
    if not data or "username" not in data:
        await callback.message.answer("⚠️ Сначала отправьте ссылку на Instagram-профиль.")
        return

    username = data["username"]
    count = int(callback.data.split("_")[1])

    status_msg = await callback.message.answer(
        f"⏳ Загружаю профиль <b>@{_escape_html(username)}</b>…\n"
        f"{progress_bar(0, count)} 0/{count}"
    )

    # --- Step 1: Parse Instagram (sync, in executor) ---
    loop = asyncio.get_running_loop()
    try:
        videos = await loop.run_in_executor(
            None, get_instagram_videos, username, count,
        )
    except ValueError as exc:
        await status_msg.edit_text(f"❌ {_escape_html(str(exc))}")
        return
    except Exception as exc:
        log.exception("Instagram fetch error")
        await status_msg.edit_text(f"❌ Ошибка при парсинге Instagram: {_escape_html(str(exc))}")
        return

    total = len(videos)
    await status_msg.edit_text(
        f"📥 Найдено видео: {total}. Начинаю обработку…\n"
        f"{progress_bar(0, total)} 0/{total}"
    )

    # --- Step 2: Process each video sequentially ---
    results: list[dict[str, Any]] = []
    for i, video in enumerate(videos, 1):
        result = await process_single_video(video)
        results.append(result)

        try:
            await status_msg.edit_text(
                f"⚙️ Обработка: {i}/{total}\n"
                f"{progress_bar(i, total)} {i}/{total}\n"
                f"Текущее: <code>{video['shortcode']}</code> — "
                f"{'✅' if result['status'] == 'success' else '❌'}"
            )
        except Exception:
            pass  # Telegram may throttle edits

        await asyncio.sleep(ANTI_SPAM_DELAY)

    # --- Step 3: Send results ---
    try:
        await status_msg.edit_text(
            f"✅ Обработка завершена!\n{progress_bar(total, total)} {total}/{total}"
        )
    except Exception:
        pass

    # Summary
    summary = format_summary(results, username)
    await callback.message.answer(summary)
    await asyncio.sleep(ANTI_SPAM_DELAY)

    # Individual results
    for i, result in enumerate(results, 1):
        text = format_result(result, i)
        for part in split_message(text):
            await callback.message.answer(part)
            await asyncio.sleep(ANTI_SPAM_DELAY)

    # Cleanup user_data
    user_data.pop(uid, None)


@router.message()
async def fallback(message: Message) -> None:
    await message.answer(
        "💡 Отправьте ссылку на Instagram-профиль.\n"
        "Пример: <code>instagram.com/durov</code>\n\n"
        "/help — справка",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    dp = Dispatcher()
    dp.include_router(router)
    log.info("Bot started — polling…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
