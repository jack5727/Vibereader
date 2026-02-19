# -*- coding: utf-8 -*-
"""
Vibereader LINE Webhook Bot (YouTube -> Transcript -> OpenAI Summary)

✅ Features
- LINE webhook endpoint: POST /webhook
- Extract YouTube URL, fetch transcript, do Map-Reduce summarization (better quality for ~60min videos)
- Reply immediately ("開始處理") then push final summary
- Background thread to avoid LINE webhook timeout
- Usage logging to usage_log.jsonl (per OpenAI call)
- Error logging to error_log.txt

🔧 Required .env (or environment variables)
- LINE_CHANNEL_ACCESS_TOKEN
- OPENAI_API_KEY

(Optional but recommended)
- LINE_CHANNEL_SECRET   # if set, will verify X-Line-Signature
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

import requests
from flask import Flask, request, abort
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from typing import Optional


# -----------------------------
# Load env
# -----------------------------
load_dotenv()

LINE_TOKEN = (os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or "").strip()
LINE_CHANNEL_SECRET = (os.getenv("LINE_CHANNEL_SECRET") or "").strip()  # optional but recommended
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

if not LINE_TOKEN:
    raise RuntimeError("缺少 LINE_CHANNEL_ACCESS_TOKEN，請在 .env 設定")
if not OPENAI_API_KEY:
    raise RuntimeError("缺少 OPENAI_API_KEY，請在 .env 設定")

client = OpenAI(api_key=OPENAI_API_KEY)

# Model
OPENAI_MODEL = "gpt-4o-mini"

# Summary template (macro economist, zh-TW)
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
# Helpers: OpenAI calls + usage logging
# -----------------------------
def make_json_safe(obj):
    """把任何物件轉成可被 json.dumps 的結構（dict/list/str/int/...）"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    # 有些 SDK 物件有 __dict__
    if hasattr(obj, "__dict__"):
        return make_json_safe(obj.__dict__)
    # 最後兜底：轉字串
    return str(obj)

def openai_call(instructions: str, input_text: str) -> tuple[str, dict]:
    """Call OpenAI once. Return (output_text, usage_dict)."""
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


def log_usage(tag: str, usage: dict, extra: dict = None) -> None:
    """把每次 token 使用量寫到 usage_log.jsonl（容錯：可序列化轉換）"""
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "model": OPENAI_MODEL,
        "usage": make_json_safe(usage),
    }
    if extra:
        rec["extra"] = make_json_safe(extra)

    with open("usage_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")



# -----------------------------
# LINE signature verification (optional)
# -----------------------------
def verify_line_signature(raw_body: bytes, signature_b64: str) -> bool:
    """
    Verify X-Line-Signature if LINE_CHANNEL_SECRET is set.
    LINE signature = base64(hmac_sha256(channel_secret, body))
    """
    if not LINE_CHANNEL_SECRET:
        return True  # skip verification if secret not provided

    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature_b64 or "")


# -----------------------------
# YouTube transcript helpers
# -----------------------------
def extract_video_id(url: str) -> str:
    """Extract YouTube videoId (11 chars)."""
    m = re.search(r"(v=|youtu\.be/)([A-Za-z0-9_\-]{11})", url)
    if not m:
        raise ValueError("這不是有效的 YouTube 影片連結，請貼完整網址")
    return m.group(2)


def fetch_cc_transcript_text(video_id: str) -> str:
    """
    Fetch transcript. Prefer Chinese (Traditional) then fallback.
    """
    api = YouTubeTranscriptApi()
    data = api.fetch(video_id, languages=["zh-Hant", "zh-TW", "zh", "en"])
    # youtube_transcript_api may return objects; keep safe access
    lines = []
    for item in data:
        t = getattr(item, "text", None)
        if t is None and isinstance(item, dict):
            t = item.get("text")
        if t:
            lines.append(t)
    return "\n".join(lines)


def compress_transcript(text: str, max_chars: int = 200000) -> str:
    """
    Light cleanup:
    - strip empty lines
    - remove exact duplicates
    - keep up to max_chars (very large; mainly to avoid insane memory)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen = set()
    uniq = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            uniq.append(ln)
    joined = "\n".join(uniq)
    return joined[:max_chars]


def chunk_text_by_chars(text: str, chunk_size: int = 4500, overlap: int = 350) -> list[str]:
    """
    Chunk by characters, with overlap to avoid cutting important context.
    Suitable for subtitle text.
    """
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
    """
    Map-Reduce summary:
    1) Map: summarize each chunk (5-8 bullets)
    2) Reduce: merge chunk summaries into final structured report
    """
    cleaned = compress_transcript(transcript_text, max_chars=200000)
    chunks = chunk_text_by_chars(cleaned, chunk_size=4500, overlap=350)

    if not chunks:
        return "抓不到字幕內容（可能影片未提供字幕或字幕取得失敗）。"

    partials: list[str] = []
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

        # Small pause to reduce rate-limit spikes (optional)
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
    """Reply to user via Reply API (token usable only once)."""
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text[:4900]}],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    r = requests.post(url, headers=headers, data=data, timeout=30)
    r.raise_for_status()


def line_push(user_id: str, text: str) -> None:
    """Push message to user via Push API."""
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {"to": user_id, "messages": [{"type": "text", "text": text[:4900]}]}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    r = requests.post(url, headers=headers, data=data, timeout=60)
    r.raise_for_status()


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
    try:
        reply_token = event.get("replyToken", "")
        source = event.get("source", {}) or {}
        user_id = source.get("userId", "")

        msg = event.get("message", {}) or {}
        user_text = (msg.get("text") or "").strip()

        m = re.search(r"(https?://\S+)", user_text)
        if not m:
            if reply_token:
                line_reply(reply_token, "請貼上 YouTube 影片連結，我會幫你摘要。")
            return

        youtube_url = m.group(1)

        # Reply quickly (reply token can be used only once)
        if reply_token:
            line_reply(reply_token, "收到連結，正在抓字幕並整理摘要（約 1–3 分鐘，視影片長度與字幕而定）")

        # Heavy steps
        video_id = extract_video_id(youtube_url)
        transcript = fetch_cc_transcript_text(video_id)
        summary = openai_map_reduce_summary(transcript, youtube_url)

        # Push final summary
        if user_id:
            line_push(user_id, summary)
        else:
            # fallback: cannot push, no userId
            # (rare, but keep safe)
            pass

    except Exception:
        err = traceback.format_exc()
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write(err)
        # Try push an error message if possible
        try:
            user_id = (event.get("source", {}) or {}).get("userId", "")
            if user_id:
                line_push(user_id, "抱歉，處理摘要時發生錯誤。我已記錄錯誤訊息（error_log.txt）。")
        except Exception:
            pass


# -----------------------------
# Webhook
# -----------------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    # Verify signature (optional)
    raw_body = request.get_data()  # bytes
    signature = request.headers.get("X-Line-Signature", "")

    if not verify_line_signature(raw_body, signature):
        abort(400)

    body = request.get_json(silent=True)
    if not body:
        abort(400)

    try:
        events = body.get("events", []) or []

        for event in events:
            if event.get("type") != "message":
                continue
            msg = event.get("message", {}) or {}
            if msg.get("type") != "text":
                continue

            # Run in background thread so webhook returns immediately (LINE verify stable)
            threading.Thread(target=process_event, args=(event,), daemon=True).start()

        return "OK", 200

    except Exception:
        err = traceback.format_exc()
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write(err)
        return "ERROR", 500


# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    # Local dev server
    app.run(host="0.0.0.0", port=8000, debug=True)
