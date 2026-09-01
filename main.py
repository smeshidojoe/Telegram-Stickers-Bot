import asyncio
import logging
import math
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ContentType
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    InputSticker,
    Message,
)
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise ValueError("BOT_TOKEN не задан — ни в .env, ни в окружении")

log = logging.getLogger("stickerbot")

bot = Bot(TOKEN)
dp = Dispatcher()

# --- Ограничения Telegram для видеостикеров ---
STICKER_SIDE = 512
STICKER_MAX_BYTES = 256 * 1024
STICKER_MAX_SECONDS = 3.0

# Bot API не отдаёт файлы больше 20 МБ.
DOWNLOAD_LIMIT = 20 * 1024 * 1024

# Проценты скорости: шаг 5, 0% = как в оригинале.
PCT_STEP = 5
PCT_MIN = -90
PCT_MAX = 900

# Обрезка: шаг и минимальная длина фрагмента.
TRIM_STEP = 0.5
MIN_SEGMENT = 0.5

# reverse держит все кадры фрагмента в памяти (512x512 RGBA ~ 1 МБ на кадр),
# поэтому для эффектов ограничиваем длину.
EFFECT_MAX_SECONDS = 10.0

EFFECTS = ["none", "reverse", "boomerang"]
EFFECT_LABELS = {"none": "нет", "reverse": "реверс", "boomerang": "бумеранг"}

STICKER_EMOJI = "🎬"
PACK_TITLE = "Круглые стикеры"

# Лестница качества: поднимаем crf / роняем fps, пока не влезем в 256 КБ.
QUALITY_LADDER = [(32, 30), (40, 30), (48, 24), (56, 20), (63, 15)]

# Кандидаты на GPU-декод, в порядке проверки; неподходящие для системы просто
# не пройдут пробу. vaapi — для Linux и контейнеров, d3d11va/dxva2 — для Windows.
# Энкод VP9+alpha возможен только на CPU (libvpx-vp9): ни NVENC, ни QSV,
# ни VAAPI не умеют VP9 с альфа-каналом.
HWACCEL_CANDIDATES = ["cuda", "vaapi", "qsv", "d3d11va", "dxva2"]

SESSION_TTL = 3600.0
FFMPEG_TIMEOUT = 300.0

# Рабочие файлы — в tmp/ рядом с кодом, по подпапке на задачу.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_ROOT = os.path.join(PROJECT_DIR, "tmp")
MASK_PATH = os.path.join(TMP_ROOT, "mask.png")
CHECKER_PATH = os.path.join(TMP_ROOT, "checker.png")

JOB_CODE_RE = re.compile(r"^[0-9a-f]{4}(-[0-9a-f]{4}){3}$")

# Файл, по свежести которого HEALTHCHECK судит, жив ли цикл событий.
HEARTBEAT_PATH = os.path.join(TMP_ROOT, "heartbeat")
HEARTBEAT_PERIOD = 30.0

# Не даём пачке видео забить все ядра.
ffmpeg_semaphore = asyncio.Semaphore(2)

_hwaccel: str | None = None
_hwaccel_probed = False
_bot_username: str | None = None


def new_job_code() -> str:
    """Код задачи вида aaaa-bbbb-cccc-dddd — он же имя подпапки в tmp/."""
    h = uuid.uuid4().hex[:16]
    return "-".join(h[i:i + 4] for i in range(0, 16, 4))


@dataclass
class Session:
    file_id: str
    duration: float
    source_fps: float
    job_dir: str
    input_path: str
    pct: int = 0
    start: float = 0.0
    length: float = 0.0
    effect: str = "none"
    sticker_file_id: str | None = None
    created_at: float = field(default_factory=time.monotonic)


sessions: dict[str, Session] = {}


def drop_session(sid: str) -> None:
    session = sessions.pop(sid, None)
    if session:
        shutil.rmtree(session.job_dir, ignore_errors=True)


def purge_sessions() -> None:
    now = time.monotonic()
    for sid in [k for k, s in sessions.items() if now - s.created_at > SESSION_TTL]:
        drop_session(sid)


def sweep_tmp() -> None:
    """Подчищает папки задач, оставшиеся от прошлых запусков."""
    os.makedirs(TMP_ROOT, exist_ok=True)

    for name in os.listdir(TMP_ROOT):
        path = os.path.join(TMP_ROOT, name)
        if os.path.isdir(path) and JOB_CODE_RE.match(name):
            shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------
# ffmpeg
# --------------------------------------------------------------------------


async def run(cmd: list[str], timeout: float = FFMPEG_TIMEOUT) -> tuple[int, str]:
    """Запускает процесс, возвращает (код возврата, stderr).

    Таймаут обязателен: подвисший ffmpeg иначе держит семафор навсегда.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "ffmpeg превысил лимит {0:.0f} с".format(timeout)

    return proc.returncode, stderr.decode(errors="ignore")


async def probe(path: str) -> tuple[float, float]:
    """Возвращает (длительность в секундах, fps) первого видеопотока."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate:format=duration",
        "-of", "default=noprint_wrappers=1",
        path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()

    duration, fps = 0.0, 30.0

    for line in stdout.decode(errors="ignore").splitlines():
        key, _, value = line.partition("=")
        if key == "duration" and value not in ("", "N/A"):
            duration = float(value)
        elif key == "r_frame_rate" and "/" in value:
            num, den = value.split("/")
            if float(den):
                fps = float(num) / float(den)

    return duration, fps


async def ensure_assets() -> None:
    """Рисует круглую маску и фон-шашечки — по одному разу на весь процесс.

    Маску считаем в 4x и уменьшаем с lanczos, чтобы край был сглаженным.
    Это единственные два места, где остаётся медленный geq, и отрабатывает
    он ровно по одному кадру.
    """
    os.makedirs(TMP_ROOT, exist_ok=True)

    if not os.path.exists(MASK_PATH):
        big = STICKER_SIDE * 4
        c = big / 2
        r = c - 2

        code, err = await run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi",
            "-i", "color=c=black:s={0}x{0}:d=1".format(big),
            "-vf", (
                "format=gray,"
                "geq=lum='if(lte((X-{c})*(X-{c})+(Y-{c})*(Y-{c}),{r}*{r}),255,0)',"
                "scale={side}:{side}:flags=lanczos"
            ).format(c=c, r=r, side=STICKER_SIDE),
            "-frames:v", "1",
            MASK_PATH,
        ])

        if code != 0:
            raise RuntimeError("не удалось создать маску: " + err.strip()[-400:])

    if not os.path.exists(CHECKER_PATH):
        code, err = await run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi",
            "-i", "color=c=black:s={0}x{0}:d=1".format(STICKER_SIDE),
            "-vf", (
                "format=gray,"
                "geq=lum='if(eq(mod(floor(X/32)+floor(Y/32),2),0),235,205)',"
                "format=rgb24"
            ),
            "-frames:v", "1",
            CHECKER_PATH,
        ])

        if code != 0:
            raise RuntimeError("не удалось создать фон: " + err.strip()[-400:])


async def detect_hwaccel(sample: str) -> str | None:
    """Один раз подбирает рабочий GPU-декодер, иначе None (чистый CPU)."""
    global _hwaccel, _hwaccel_probed

    if _hwaccel_probed:
        return _hwaccel

    for name in HWACCEL_CANDIDATES:
        code, _ = await run([
            "ffmpeg", "-y", "-v", "error",
            "-hwaccel", name,
            "-i", sample,
            "-frames:v", "1",
            "-f", "null", "-",
        ])
        if code == 0:
            _hwaccel = name
            break

    _hwaccel_probed = True
    log.info("hwaccel для декода: %s", _hwaccel or "нет, только CPU")
    return _hwaccel


CROP_CHAIN = (
    "fps={fps},"
    "scale={side}:{side}:force_original_aspect_ratio=increase,"
    "crop={side}:{side},"
    "format=rgba"
)


def speed_factor(pct: int) -> float:
    return 1.0 + pct / 100.0


def effect_chain(effect: str, speed: float) -> str:
    """Хвост цепочки после кадрирования: эффект, затем скорость.

    reverse сбрасывает таймстемпы, поэтому перед concat выравниваем их
    через setpts=PTS-STARTPTS — иначе бумеранг склеится с дырой.
    """
    speedup = "setpts=PTS/{0:.6f}[v]".format(speed)

    if effect == "reverse":
        return "reverse,setpts=PTS-STARTPTS," + speedup

    if effect == "boomerang":
        return (
            "split[a][b];"
            "[b]reverse,setpts=PTS-STARTPTS[r];"
            "[a][r]concat=n=2:v=1:a=0," + speedup
        )

    return speedup


def build_command(
    session: "Session",
    output_path: str,
    crf: int,
    fps: int,
    hwaccel: str | None,
) -> list[str]:
    cmd = ["ffmpeg", "-y", "-v", "error"]

    if hwaccel:
        cmd += ["-hwaccel", hwaccel]

    filters = (
        "[0:v]{crop},{tail};"
        "[1:v]format=gray,scale={side}:{side}[m];"
        # shortest=0 + repeatlast=1 (дефолт): однокадровая маска
        # растягивается на всё видео и не обрывает его.
        "[v][m]alphamerge,format=yuva420p[o]"
    ).format(
        crop=CROP_CHAIN.format(fps=fps, side=STICKER_SIDE),
        tail=effect_chain(session.effect, speed_factor(session.pct)),
        side=STICKER_SIDE,
    )

    cmd += [
        # -ss и -t до -i: ffmpeg перематывает, а не декодирует всё подряд.
        "-ss", "{0:.3f}".format(session.start),
        "-t", "{0:.3f}".format(session.length),
        "-i", session.input_path,
        "-i", MASK_PATH,
        "-filter_complex", filters,
        "-map", "[o]",
        "-an",
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-auto-alt-ref", "0",   # обязательно для VP9 с альфой
        "-crf", str(crf),
        "-b:v", "0",
        "-row-mt", "1",
        "-cpu-used", "4",
        "-deadline", "good",
        output_path,
    ]

    return cmd


async def convert(session: Session, output_path: str) -> None:
    """Кодирует стикер, подбирая качество под лимит 256 КБ."""
    await ensure_assets()
    hwaccel = await detect_hwaccel(session.input_path)

    async with ffmpeg_semaphore:
        for crf, fps in QUALITY_LADDER:
            code, err = await run(
                build_command(session, output_path, crf, fps, hwaccel)
            )

            if code != 0 and hwaccel:
                # GPU-декод отвалился на этом конкретном файле — идём на CPU.
                log.warning("hwaccel %s не справился, откат на CPU", hwaccel)
                code, err = await run(
                    build_command(session, output_path, crf, fps, None)
                )

            if code != 0:
                raise RuntimeError(err.strip()[-800:])

            size = os.path.getsize(output_path)

            if size <= STICKER_MAX_BYTES:
                return

            log.info("crf=%s fps=%s дал %s байт, жмём сильнее", crf, fps, size)

    # Лестница кончилась — отдаём самый сжатый вариант, дальше решит Telegram.


async def render_preview(session: Session) -> str:
    """Один кадр с начала фрагмента: круг поверх шашечек.

    PNG уйдёт фотографией, а фотографии Telegram расплющивает в JPEG,
    так что прозрачность подкладываем сами — иначе был бы чёрный квадрат.
    """
    await ensure_assets()

    path = os.path.join(session.job_dir, "preview.png")

    filters = (
        "[0:v]scale={side}:{side}:force_original_aspect_ratio=increase,"
        "crop={side}:{side},format=rgba[v];"
        "[1:v]format=gray,scale={side}:{side}[m];"
        "[v][m]alphamerge[fg];"
        "[2:v][fg]overlay=0:0:format=auto,format=rgb24[o]"
    ).format(side=STICKER_SIDE)

    code, err = await run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", "{0:.3f}".format(session.start),
        "-i", session.input_path,
        "-i", MASK_PATH,
        "-i", CHECKER_PATH,
        "-filter_complex", filters,
        "-map", "[o]",
        "-frames:v", "1",
        path,
    ], timeout=60.0)

    if code != 0:
        raise RuntimeError("превью не получилось: " + err.strip()[-400:])

    return path


# --------------------------------------------------------------------------
# Расчёты
# --------------------------------------------------------------------------


def out_duration(session: Session) -> float:
    result = session.length / speed_factor(session.pct)
    return result * 2 if session.effect == "boomerang" else result


def fmt(seconds: float) -> str:
    return "{0:.1f} с".format(seconds)


def clamp(pct: int) -> int:
    return max(PCT_MIN, min(PCT_MAX, pct))


def max_segment(session: Session) -> float:
    """Верхний предел длины фрагмента с учётом прожорливости reverse."""
    room = session.duration - session.start

    if session.effect != "none":
        room = min(room, EFFECT_MAX_SECONDS)

    return max(MIN_SEGMENT, room)


def clamp_trim(session: Session) -> None:
    session.start = max(0.0, min(session.start, session.duration - MIN_SEGMENT))
    session.length = max(MIN_SEGMENT, min(session.length, max_segment(session)))


def pct_to_fit(session: Session) -> int:
    """Минимальный процент ускорения, при котором стикер влезает в 3 с."""
    base = session.length * (2 if session.effect == "boomerang" else 1)

    if base <= STICKER_MAX_SECONDS:
        return 0

    needed = (base / STICKER_MAX_SECONDS - 1) * 100
    return clamp(int(math.ceil(needed / PCT_STEP) * PCT_STEP))


# --------------------------------------------------------------------------
# Клавиатуры и тексты
# --------------------------------------------------------------------------


def mode_keyboard(sid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Обычная", callback_data="go:" + sid),
            InlineKeyboardButton(text="Настроить", callback_data="custom:" + sid),
        ],
        [InlineKeyboardButton(text="Отмена", callback_data="cancel:" + sid)],
    ])


def settings_keyboard(sid: str, session: Session) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(text="−25%", callback_data="d:{0}:-25".format(sid)),
            InlineKeyboardButton(text="−5%", callback_data="d:{0}:-5".format(sid)),
            InlineKeyboardButton(text="0%", callback_data="d:{0}:0".format(sid)),
            InlineKeyboardButton(text="+5%", callback_data="d:{0}:5".format(sid)),
            InlineKeyboardButton(text="+25%", callback_data="d:{0}:25".format(sid)),
        ],
    ]

    if out_duration(session) > STICKER_MAX_SECONDS:
        rows.append([
            InlineKeyboardButton(
                text="Уместить в 3 с (+{0}%)".format(pct_to_fit(session)),
                callback_data="fit:" + sid,
            )
        ])

    rows.append([
        InlineKeyboardButton(text="◀ старт", callback_data="s:{0}:-1".format(sid)),
        InlineKeyboardButton(text="старт ▶", callback_data="s:{0}:1".format(sid)),
        InlineKeyboardButton(text="− длина", callback_data="l:{0}:-1".format(sid)),
        InlineKeyboardButton(text="+ длина", callback_data="l:{0}:1".format(sid)),
    ])

    rows.append([
        InlineKeyboardButton(
            text="Эффект: " + EFFECT_LABELS[session.effect],
            callback_data="e:" + sid,
        )
    ])

    rows.append([
        InlineKeyboardButton(text="Готово", callback_data="go:" + sid),
        InlineKeyboardButton(text="Отмена", callback_data="cancel:" + sid),
    ])

    return InlineKeyboardMarkup(inline_keyboard=rows)


def pack_keyboard(sid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="➕ В мой пак", callback_data="pack:" + sid)
    ]])


def mode_text(session: Session) -> str:
    lines = [
        "Длина видео: " + fmt(session.duration),
        "На превью — первый кадр будущего стикера.",
        "",
        "Какая будет скорость?",
    ]

    if session.duration > STICKER_MAX_SECONDS:
        lines += [
            "",
            "Видео длиннее 3 с. Telegram может не принять такой стикер — "
            "тогда пришлю файлом. «Настроить» даёт обрезку и ускорение.",
        ]

    return "\n".join(lines)


def settings_text(session: Session) -> str:
    result = out_duration(session)
    sign = "+" if session.pct > 0 else ""

    lines = [
        "Скорость: {0}{1}%".format(sign, session.pct),
        "Фрагмент: {0:.1f}–{1:.1f} с из {2}".format(
            session.start, session.start + session.length, fmt(session.duration)
        ),
        "Эффект: " + EFFECT_LABELS[session.effect],
        "",
        "Исходная длина: " + fmt(session.length),
        "Выходная длина: " + fmt(result),
    ]

    if result > STICKER_MAX_SECONDS:
        lines += ["", "Дольше 3 с — Telegram может отказать в стикере."]

    if session.effect != "none" and session.length >= EFFECT_MAX_SECONDS:
        lines += [
            "",
            "С эффектом фрагмент ограничен {0:.0f} с.".format(EFFECT_MAX_SECONDS),
        ]

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Хендлеры: приём видео
# --------------------------------------------------------------------------


@dp.message(CommandStart())
@dp.message(Command("help"))
async def handle_start(message: Message):
    await message.answer(
        "Пришли видео или кружок — верну круглый видеостикер.\n\n"
        "Перед конвертацией покажу превью и спрошу скорость. "
        "В «Настроить» есть шаг 5% по скорости, обрезка фрагмента, "
        "реверс и бумеранг. Готовый стикер можно добавить в свой пак."
    )


async def start_session(message: Message, file_id: str, file_size: int | None):
    if file_size and file_size > DOWNLOAD_LIMIT:
        await message.answer(
            "Файл больше 20 МБ — Telegram не даёт боту его скачать."
        )
        return

    purge_sessions()

    status = await message.answer("Читаю видео…")

    sid = new_job_code()
    job_dir = os.path.join(TMP_ROOT, sid)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, "input.mp4")

    try:
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, destination=input_path)

        duration, fps = await probe(input_path)

        if duration <= 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            await status.edit_text("В файле не нашлось видеопотока.")
            return

        session = Session(
            file_id=file_id,
            duration=duration,
            source_fps=fps,
            job_dir=job_dir,
            input_path=input_path,
            length=duration,
        )
        clamp_trim(session)
        sessions[sid] = session

        preview = await render_preview(session)

    except Exception as exc:
        log.exception("не смог подготовить видео")
        sessions.pop(sid, None)
        shutil.rmtree(job_dir, ignore_errors=True)
        await status.edit_text("Не удалось прочитать видео: {0}".format(exc))
        return

    await status.delete()
    await message.answer_photo(
        FSInputFile(preview),
        caption=mode_text(session),
        reply_markup=mode_keyboard(sid),
    )


@dp.message(F.content_type == ContentType.VIDEO)
async def handle_video(message: Message):
    await start_session(message, message.video.file_id, message.video.file_size)


@dp.message(F.content_type == ContentType.VIDEO_NOTE)
async def handle_video_note(message: Message):
    await start_session(
        message, message.video_note.file_id, message.video_note.file_size
    )


@dp.message(F.content_type == ContentType.ANIMATION)
async def handle_animation(message: Message):
    await start_session(
        message, message.animation.file_id, message.animation.file_size
    )


@dp.message(F.document.mime_type.startswith("video/"))
async def handle_video_document(message: Message):
    await start_session(
        message, message.document.file_id, message.document.file_size
    )


# --------------------------------------------------------------------------
# Хендлеры: настройка
# --------------------------------------------------------------------------


def get_session(sid: str) -> Session | None:
    purge_sessions()
    return sessions.get(sid)


async def refresh(
    callback: CallbackQuery, sid: str, session: Session, new_frame: bool
) -> None:
    """Перерисовывает панель настроек.

    Кадр перекодируем только когда сдвинулся старт: на смене скорости или
    эффекта картинка та же, хватает правки подписи.
    """
    text = settings_text(session)
    markup = settings_keyboard(sid, session)

    try:
        if new_frame:
            preview = await render_preview(session)
            await callback.message.edit_media(
                InputMediaPhoto(media=FSInputFile(preview), caption=text),
                reply_markup=markup,
            )
        else:
            await callback.message.edit_caption(caption=text, reply_markup=markup)
    except TelegramBadRequest as exc:
        # "message is not modified" — клик ничего не поменял, это не ошибка.
        if "not modified" not in str(exc):
            raise


@dp.callback_query(F.data.startswith("custom:"))
async def on_custom(callback: CallbackQuery):
    sid = callback.data.split(":")[1]
    session = get_session(sid)

    if not session:
        await callback.answer("Сессия устарела, пришли видео заново.", show_alert=True)
        return

    await refresh(callback, sid, session, new_frame=False)
    await callback.answer()


@dp.callback_query(F.data.startswith("d:"))
async def on_speed(callback: CallbackQuery):
    _, sid, raw = callback.data.split(":")
    session = get_session(sid)

    if not session:
        await callback.answer("Сессия устарела, пришли видео заново.", show_alert=True)
        return

    delta = int(raw)
    session.pct = 0 if delta == 0 else clamp(session.pct + delta)

    await refresh(callback, sid, session, new_frame=False)
    await callback.answer()


@dp.callback_query(F.data.startswith("fit:"))
async def on_fit(callback: CallbackQuery):
    sid = callback.data.split(":")[1]
    session = get_session(sid)

    if not session:
        await callback.answer("Сессия устарела, пришли видео заново.", show_alert=True)
        return

    session.pct = pct_to_fit(session)

    await refresh(callback, sid, session, new_frame=False)
    await callback.answer()


@dp.callback_query(F.data.startswith("s:"))
async def on_start_shift(callback: CallbackQuery):
    _, sid, raw = callback.data.split(":")
    session = get_session(sid)

    if not session:
        await callback.answer("Сессия устарела, пришли видео заново.", show_alert=True)
        return

    before = session.start
    session.start += int(raw) * TRIM_STEP
    clamp_trim(session)

    if session.start == before:
        await callback.answer("Дальше некуда.")
        return

    await refresh(callback, sid, session, new_frame=True)
    await callback.answer()


@dp.callback_query(F.data.startswith("l:"))
async def on_length_shift(callback: CallbackQuery):
    _, sid, raw = callback.data.split(":")
    session = get_session(sid)

    if not session:
        await callback.answer("Сессия устарела, пришли видео заново.", show_alert=True)
        return

    before = session.length
    session.length += int(raw) * TRIM_STEP
    clamp_trim(session)

    if session.length == before:
        await callback.answer("Дальше некуда.")
        return

    await refresh(callback, sid, session, new_frame=False)
    await callback.answer()


@dp.callback_query(F.data.startswith("e:"))
async def on_effect(callback: CallbackQuery):
    sid = callback.data.split(":")[1]
    session = get_session(sid)

    if not session:
        await callback.answer("Сессия устарела, пришли видео заново.", show_alert=True)
        return

    session.effect = EFFECTS[(EFFECTS.index(session.effect) + 1) % len(EFFECTS)]
    clamp_trim(session)

    await refresh(callback, sid, session, new_frame=False)
    await callback.answer()


@dp.callback_query(F.data.startswith("cancel:"))
async def on_cancel(callback: CallbackQuery):
    sid = callback.data.split(":")[1]
    drop_session(sid)

    await callback.message.delete()
    await callback.answer("Отменено.")


# --------------------------------------------------------------------------
# Хендлеры: сборка стикера и пак
# --------------------------------------------------------------------------


@dp.callback_query(F.data.startswith("go:"))
async def on_go(callback: CallbackQuery):
    sid = callback.data.split(":")[1]
    session = get_session(sid)

    if not session:
        await callback.answer("Сессия устарела, пришли видео заново.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_caption(
        caption="Конвертирую… ({0} → {1})".format(
            fmt(session.length), fmt(out_duration(session))
        )
    )

    output_path = os.path.join(session.job_dir, "sticker.webm")

    try:
        await convert(session, output_path)

        try:
            sent = await callback.message.answer_sticker(
                FSInputFile(output_path, filename="sticker.webm"),
                reply_markup=pack_keyboard(sid),
            )
        except TelegramBadRequest as exc:
            log.warning("Telegram отверг стикер: %s", exc)
            await callback.message.answer_document(
                FSInputFile(output_path, filename="sticker.webm"),
                caption=(
                    "Telegram не принял это как стикер: {0}, {1} КБ.\n"
                    "Лимиты — до 3 с и до 256 КБ. Файл приложил; "
                    "либо нажми «Уместить в 3 с» и попробуй снова."
                ).format(
                    fmt(out_duration(session)),
                    os.path.getsize(output_path) // 1024,
                ),
            )
            await refresh(callback, sid, session, new_frame=False)
            return

    except Exception as exc:
        log.exception("конвертация упала")
        await callback.message.edit_caption(
            caption="Не удалось конвертировать: {0}".format(exc)
        )
        return

    # Сессию держим ради кнопки «в пак», а рабочие файлы уже не нужны:
    # addStickerToSet принимает file_id уже отправленного стикера.
    session.sticker_file_id = sent.sticker.file_id
    session.created_at = time.monotonic()
    shutil.rmtree(session.job_dir, ignore_errors=True)

    await callback.message.delete()


async def pack_name(user_id: int) -> str:
    global _bot_username

    if _bot_username is None:
        _bot_username = (await bot.get_me()).username

    return "u{0}_by_{1}".format(user_id, _bot_username)


@dp.callback_query(F.data.startswith("pack:"))
async def on_pack(callback: CallbackQuery):
    sid = callback.data.split(":")[1]
    session = get_session(sid)

    if not session or not session.sticker_file_id:
        await callback.answer("Стикер устарел, сделай новый.", show_alert=True)
        return

    user_id = callback.from_user.id
    name = await pack_name(user_id)

    sticker = InputSticker(
        sticker=session.sticker_file_id,
        format="video",
        emoji_list=[STICKER_EMOJI],
    )

    try:
        try:
            await bot.add_sticker_to_set(user_id=user_id, name=name, sticker=sticker)
        except TelegramBadRequest as exc:
            # Пака ещё нет — первый стикер его создаёт.
            if "STICKERSET_INVALID" not in str(exc).upper():
                raise
            await bot.create_new_sticker_set(
                user_id=user_id,
                name=name,
                title=PACK_TITLE,
                stickers=[sticker],
            )
    except TelegramBadRequest as exc:
        log.warning("пак не обновился: %s", exc)
        await callback.answer("Не вышло: {0}".format(exc), show_alert=True)
        return

    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer(
        "Готово: https://t.me/addstickers/{0}".format(name)
    )
    await callback.answer("Добавлено в пак")

    drop_session(sid)


@dp.message()
async def handle_other(message: Message):
    await message.answer("Пришли видео или кружок — сделаю круглый стикер.")


async def heartbeat_loop():
    """Отмечает, что цикл событий крутится, а не залип на блокирующем вызове."""
    while True:
        try:
            with open(HEARTBEAT_PATH, "w") as fh:
                fh.write(str(time.time()))
        except OSError as exc:
            log.warning("heartbeat не записался: %s", exc)

        await asyncio.sleep(HEARTBEAT_PERIOD)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sweep_tmp()
    await ensure_assets()

    beat = asyncio.create_task(heartbeat_loop())

    try:
        await dp.start_polling(bot)
    finally:
        beat.cancel()


if __name__ == "__main__":
    asyncio.run(main())
