# -*- coding: utf-8 -*-
"""
Vibereader LINE Webhook Bot (YouTube -> Transcript -> OpenAI Summary)

Features
- LINE webhook endpoint: POST /webhook
- Extract YouTube URL, fetch transcript, do Map-Reduce summarization
- Reply immediately then push final summary
- Background thread to avoid LINE webhook timeout
- Usage logging to /tmp/usage_log.jsonl
- Error logging to /tmp/error_log.txt
"""

import os
import re
import json
import time
import hmac
import base64
import hashlib
import traceback
import threading
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import requests
from flask import Flask, request, abort
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import random
import queue
from youtube_transcript_api._errors import YouTubeRequestFailed

load_dotenv()

# 同一時間最多幾個 transcript 抓取（建議 1~2）
TRANSCRIPT_SEM = threading.Semaphore(int(os.getenv("TRANSCRIPT_CONCURRENCY", "1")))

# 針對 YouTube transcript 做快取（記憶體版，部署在單一 Render instance 很有用）
TRANSCRIPT_CACHE: Dict[str, Tuple[float, str]] = {}  # video_id -> (ts, text)
CACHE_TTL_SEC = int(os.getenv("TRANSCRIPT_CACHE_TTL_SEC", "86400"))  # 1 day

# -----------------------------
# Background job queue (避免 thread 暴衝)
# -----------------------------
JOB_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=int(os.getenv("JOB_QUEUE_MAXSIZE", "200")))
WORKER_COUNT = int(os.getenv("WORKER_COUNT", "1"))  # 建議 1~2（先 1 最穩）
WORKERS_STARTED = False
WORKERS_LOCK = threading.Lock()
# -----------------------------
# Load env
# -----------------------------


LINE_TOKEN = (os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or "").strip()
LINE_CHANNEL_SECRET = (os.getenv("LINE_CHANNEL_SECRET") or "").strip()  # optional
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

if not LINE_TOKEN:
    raise RuntimeError("缺少 LINE_CHANNEL_ACCESS_TOKEN，請在環境變數或 .env 設定")
if not OPENAI_API_KEY:
    raise RuntimeError("缺少 OPENAI_API_KEY，請在環境變數或 .env 設定")

client = OpenAI(api_key=OPENAI_API_KEY)

# Model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Where to log in container (Render)
USAGE_LOG_PATH = os.getenv("USAGE_LOG_PATH", "/tmp/usage_log.jsonl")
ERROR_LOG_PATH = os.getenv("ERROR_LOG_PATH", "/tmp/error_log.txt")


# -----------------------------
# Prompt templates
# -----------------------------
SUMMARY_TEMPLATE = """
你是一位資深總體經濟分析師。請根據提供的字幕內容，用繁體中文產出「10 分鐘內讀完」的影片摘要。
規則：
- 嚴禁杜撅。若字幕未明確提到，請寫「（原文未提供細節）」。
- 所有數字、時間、百分比、政策條件，務必保留並放在【關鍵數字與假設】中。
- 在【市場含義】中用「因為…所以…」描述因果鏈，不要只下結論。
- 內容以「總經/利率/通膨/就業/美元/風險資產」視角為主。

輸出格式（請嚴格照順序）：
【三行總結】
- ...
- ...
- ...

【重點條列】
- ...（8–12 點，每點最多 2 行）

【關鍵數字與假設】
- 指標/數字：...（條件：... / 時點：...）

【市場含義】
- 股市：...
- 債市：...
- 匯率：...

【風險與追蹤清單】
- ...（3–6 點）
""".strip()

MAP_INSTRUCTIONS = """
你是總體經濟分析師助理。請把這一段字幕提煉成「段落重點」，要求：
- 嚴禁杜撅；不確定就寫（原文未提供細節）
- 只保留這段內真正出現的論點與數字
- 以條列輸出 5–8 點，每點一句話，盡量保留數字/條件/時間
輸出只要條列，不要加其他標題。
""".strip()


# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)


# -----------------------------
# Logging helpers
# -----------------------------
def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_error(where: str, err_text: str) -> None:
    """Log error to stdout + /tmp/error_log.txt (append)."""
    print(f"=== ERROR @ {where} ({_ts()}) ===")
    print(err_text)

    try:
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n[{_ts()}] {where}\n{err_text}\n")
    except Exception:
        # Even if file logging fails, stdout already has the traceback
        pass


def make_json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return make_json_safe(obj.__dict__)
    return str(obj)


def log_usage(tag: str, usage: Dict[str, Any], extra: Dict[str, Any] = None) -> None:
    rec = {
        "ts": _ts(),
        "tag": tag,
        "model": OPENAI_MODEL,
        "usage": make_json_safe(usage),
    }
    if extra:
        rec["extra"] = make_json_safe(extra)

    try:
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # If file logging fails, don't break the workflow
        pass


# -----------------------------
# OpenAI helper
# -----------------------------
def openai_call(instructions: str, input_text: str) -> Tuple[str, Dict[str, Any]]:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=input_text,
    )

    text = (getattr(resp, "output_text", "") or "").strip()

    usage = getattr(resp, "usage", None)
    if usage is None:
        usage_dict = {}
    elif isinstance(usage, dict):
        usage_dict = usage
    else:
        usage_dict = usage.__dict__ if hasattr(usage, "__dict__") else {"usage": str(usage)}

    return text, usage_dict


# -----------------------------
# LINE signature verification (optional)
# -----------------------------
def verify_line_signature(raw_body: bytes, signature_b64: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        return True  # skip verification if not set

    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature_b64 or "")


# -----------------------------
# YouTube helpers
# -----------------------------
def normalize_url_tail(url: str) -> str:
    # remove common trailing punctuations from chat
    return url.rstrip(")】]〉》”’\"'。，,.;:!！？")


def extract_video_id(url: str) -> str:
    url = normalize_url_tail(url)

    # Common patterns:
    # - https://www.youtube.com/watch?v=VIDEOID
    # - https://youtu.be/VIDEOID
    # - https://www.youtube.com/shorts/VIDEOID
    patterns = [
        r"(?:v=)([A-Za-z0-9_\-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_\-]{11})",
        r"(?:/shorts/)([A-Za-z0-9_\-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)

    raise ValueError("這不是有效的 YouTube 影片連結（找不到 videoId），請貼完整網址")


def fetch_cc_transcript_text(video_id: str) -> str:
    # 1) cache hit
    now = time.time()
    cached = TRANSCRIPT_CACHE.get(video_id)
    if cached:
        ts, text = cached
        if now - ts < CACHE_TTL_SEC and text:
            return text

    # 2) concurrency guard
    with TRANSCRIPT_SEM:
        # Double-check cache (避免等待 semaphore 後又重抓)
        cached = TRANSCRIPT_CACHE.get(video_id)
        if cached:
            ts, text = cached
            if time.time() - ts < CACHE_TTL_SEC and text:
                return text

        # 3) retry with exponential backoff + jitter (針對 429 / sorry / 5xx 特別有效)
        max_attempts = int(os.getenv("YT_TRANSCRIPT_MAX_ATTEMPTS", "6"))
        base_sleep = float(os.getenv("YT_TRANSCRIPT_BASE_SLEEP", "1.2"))
        last_err = None

        for attempt in range(1, max_attempts + 1):
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

                preferred_langs = ["zh-Hant", "zh-TW", "zh-HK", "zh", "zh-Hans", "en"]
                transcript = None

                for lang in preferred_langs:
                    try:
                        transcript = transcript_list.find_manually_created_transcript([lang])
                        break
                    except Exception:
                        pass

                if transcript is None:
                    for lang in preferred_langs:
                        try:
                            transcript = transcript_list.find_generated_transcript([lang])
                            break
                        except Exception:
                            pass

                if transcript is None:
                    transcript = next(iter(transcript_list), None)
                    if transcript is None:
                        raise RuntimeError("No transcripts available")

                data = transcript.fetch()

                lines = []
                for item in data:
                    t = (item.get("text") or "").strip()
                    if t:
                        lines.append(t)
                text = "\n".join(lines)

                # cache store
                TRANSCRIPT_CACHE[video_id] = (time.time(), text)
                return text

            except Exception as e:
                last_err = e
                err_str = str(e).lower()

                # 判斷是否是 YouTube/Google 限流類型（429 / sorry / too many requests）
                is_rate_limited = (
                    "too many requests" in err_str
                    or "google.com/sorry" in err_str
                    or "429" in err_str
                )

                # 不是限流：就不要一直重試（例如影片沒字幕、被關閉）
                if not is_rate_limited and attempt >= 2:
                    break

                # backoff with jitter
                sleep_s = min(60.0, base_sleep * (2 ** (attempt - 1)))
                sleep_s = sleep_s * (0.7 + random.random() * 0.6)  # 0.7~1.3x
                print(f"[yt_transcript] attempt {attempt}/{max_attempts} failed; sleep {sleep_s:.1f}s; err={e}")
                time.sleep(sleep_s)

        raise last_err if last_err else RuntimeError("Transcript fetch failed")




def compress_transcript(text: str, max_chars: int = 200000) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen = set()
    uniq = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            uniq.append(ln)
    joined = "\n".join(uniq)
    return joined[:max_chars]


def chunk_text_by_chars(text: str, chunk_size: int = 4500, overlap: int = 350) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def openai_map_reduce_summary(transcript_text: str, youtube_url: str) -> str:
    cleaned = compress_transcript(transcript_text, max_chars=200000)
    chunks = chunk_text_by_chars(cleaned, chunk_size=4500, overlap=350)

    if not chunks:
        return "抓不到字幕內容（可能影片未提供字幕或字幕取得失敗）。"

    partials: List[str] = []
    for idx, ch in enumerate(chunks, start=1):
        input_text = (
            f"來源影片：{youtube_url}\n"
            f"這是字幕第 {idx}/{len(chunks)} 段。\n\n"
            f"{ch}"
        )
        part, usage = openai_call(MAP_INSTRUCTIONS, input_text)
        log_usage(f"map_{idx}", usage, extra={"chunks": len(chunks)})
        if part:
            partials.append(f"【段落 {idx}】\n{part}")

        time.sleep(0.15)

    if not partials:
        return "字幕摘要失敗（未取得任何段落摘要）。"

    reduce_input = "\n\n".join(partials)
    final_input = (
        f"來源影片：{youtube_url}\n"
        "以下是各段字幕的段落重點（已分段整理）。\n"
        "請依照模板產出最終摘要。\n\n"
        f"{reduce_input}"
    )

    final_text, usage = openai_call(SUMMARY_TEMPLATE, final_input)
    log_usage("reduce_final", usage, extra={"chunks": len(chunks)})

    return final_text.strip() or "摘要產生失敗（模型未輸出內容）。"


# -----------------------------
# LINE Messaging API helpers
# -----------------------------
def line_reply(reply_token: str, text: str) -> None:
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text[:4900]}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    r.raise_for_status()


def line_push(user_id: str, text: str) -> None:
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {"to": user_id, "messages": [{"type": "text", "text": text[:4900]}]}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()

def _worker_loop(worker_id: int) -> None:
    print(f"[worker-{worker_id}] started")
    while True:
        event = JOB_QUEUE.get()  # 會阻塞等待
        try:
            process_event(event)
        except Exception:
            log_error(f"worker_loop_{worker_id}", traceback.format_exc())
        finally:
            JOB_QUEUE.task_done()


def ensure_workers_started() -> None:
    global WORKERS_STARTED
    if WORKERS_STARTED:
        return
    with WORKERS_LOCK:
        if WORKERS_STARTED:
            return
        for wid in range(1, WORKER_COUNT + 1):
            t = threading.Thread(target=_worker_loop, args=(wid,), daemon=True)
            t.start()
        WORKERS_STARTED = True
        print(f"[queue] workers started: {WORKER_COUNT}")


# -----------------------------
# Background processing
# -----------------------------
def process_event(event: dict) -> None:
    """
    Heavy work in background:
    - Extract URL
    - Fetch transcript
    - Map-Reduce summarize
    - Push result
    """
    # 方便定位：每次事件打一個短 id
    eid = (event.get("replyToken") or "")[:8] or "no_token"

    def _log(step: str, extra: str = ""):
        # Render Logs 會看到
        print(f"[{eid}] {step} {extra}".strip())

    try:
        reply_token = event.get("replyToken", "") or ""
        source = event.get("source", {}) or {}
        user_id = source.get("userId", "") or ""

        msg = event.get("message", {}) or {}
        user_text = (msg.get("text") or "").strip()
        _log("received", f"text_len={len(user_text)} user_id={'Y' if user_id else 'N'}")

        # Extract URL
        m = re.search(r"(https?://[^\s]+)", user_text)
        if not m:
            if reply_token:
                line_reply(reply_token, "請貼上 YouTube 影片連結，我會幫你做摘要。")
            _log("no_url")
            return

        youtube_url = normalize_url_tail(m.group(1))
        _log("url_extracted", youtube_url)

        # Reply ASAP (reply token only once)
        if reply_token:
            try:
                line_reply(reply_token, "收到連結，開始抓字幕並整理摘要。完成後我會再把摘要推播給你。")
                _log("replied_ok")
            except Exception:
                _log("replied_failed")
                log_error("line_reply", traceback.format_exc())

        # Heavy work
        _log("extract_video_id")
        video_id = extract_video_id(youtube_url)
        _log("video_id", video_id)

        _log("fetch_transcript_start")
        try:
            transcript = fetch_cc_transcript_text(video_id)
        except Exception as e:
            # 針對 YouTube transcript 常見錯誤給更友善的訊息
            err = traceback.format_exc()
            log_error("fetch_transcript", err)

            msg = "抓不到字幕（可能影片未提供字幕、字幕被關閉、或 YouTube 暫時拒絕存取）。你可以換一支影片或稍後再試。"
            if user_id:
                line_push(user_id, msg)
            _log("fetch_transcript_failed", str(e))
            return

        _log("fetch_transcript_ok", f"chars={len(transcript)}")

        _log("summarize_start")
        summary = openai_map_reduce_summary(transcript, youtube_url)
        _log("summarize_ok", f"chars={len(summary)}")

        # Push final summary
        if not user_id:
            log_error("process_event", "No userId found; cannot push summary.")
            _log("no_user_id")
            return

        # 如果摘要太長，切段推送（避免只收到前 4900）
        parts = []
        s = summary.strip()
        while s:
            parts.append(s[:4900])
            s = s[4900:]

        _log("push_start", f"parts={len(parts)}")
        for i, part in enumerate(parts, start=1):
            line_push(user_id, part if len(parts) == 1 else f"({i}/{len(parts)})\n{part}")
            time.sleep(0.2)

        _log("push_ok")

    except Exception:
        err = traceback.format_exc()
        log_error("process_event", err)
        _log("fatal_error")

        # Best-effort push an error message
        try:
            user_id = (event.get("source", {}) or {}).get("userId", "") or ""
            if user_id:
                line_push(user_id, "抱歉，處理摘要時發生錯誤。我已在伺服器端記錄錯誤訊息，請稍後再試或換一支影片連結。")
                _log("pushed_error_msg")
        except Exception:
            log_error("line_push_error_fallback", traceback.format_exc())
            _log("push_error_msg_failed")



# -----------------------------
# Routes
# -----------------------------
@app.route("/healthz", methods=["GET"])
def healthz():
    return "OK", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    raw_body = request.get_data()  # bytes
    signature = request.headers.get("X-Line-Signature", "")

    if not verify_line_signature(raw_body, signature):
        abort(400)

    body = request.get_json(silent=True)
    if not body:
        abort(400)

    try:
        ensure_workers_started()

        events = body.get("events", []) or []
        for event in events:
            if event.get("type") != "message":
                continue
            msg = event.get("message", {}) or {}
            if msg.get("type") != "text":
                continue

            try:
                JOB_QUEUE.put_nowait(event)
                print(f"[queue] enqueued event; qsize={JOB_QUEUE.qsize()}")
            except queue.Full:
                # Queue 滿了：短時間塞太多
                reply_token = event.get("replyToken", "") or ""
                if reply_token:
                    try:
                        line_reply(reply_token, "目前同時處理的請求較多，請稍後再試一次 🙏")
                    except Exception:
                        log_error("line_reply_queue_full", traceback.format_exc())
                print("[queue] queue full; dropped event")

        return "OK", 200

    except Exception:
        log_error("webhook", traceback.format_exc())
        return "ERROR", 500


# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    # Local dev server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
