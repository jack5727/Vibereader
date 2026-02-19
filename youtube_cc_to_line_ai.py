# -*- coding: utf-8 -*-

import os
import re
import json
import sys
import traceback

import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SUMMARY_TEMPLATE = """
你是一位資深總體經濟分析師，寫給專業投資人看的影片重點摘要。
請用繁體中文，語氣像 sell-side / buy-side macro note，內容要可在 10 分鐘內讀完。
務必「保留數字與條件假設」，不要杜撰；若原文不確定請標註「（原文未提供細節）」。

輸出格式請嚴格照以下段落與標題（不要加其他標題）：

【A. 三行總結】（3 bullet）
- ...
- ...
- ...

【B. 宏觀主軸與推論鏈】（6–10 bullet；寫清楚因果：原因→機制→結果）
- ...
（每點不超過 2 句）

【C. 關鍵數據與假設】（只列原文提到者；沒有就寫「（本段無具體數據）」）
- 指標/數字/期間：...（來源：影片內容）
- ...

【D. 市場含義】（分三小段，各 2–4 bullet）
(1) 股市/風險資產：
- ...
(2) 債市/利率：
- ...
(3) 匯率/美元：
- ...

【E. 風險、反例與需驗證點】（3–6 bullet）
- ...

【F. 一週內可追蹤的觀察清單】（5 bullet，盡量具體到“看什麼/怎麼判斷”）
- ...

【G. 我的行動建議】（3 bullet，偏研究/配置/風控，不要給個股買賣指令）
- ...
""".strip()

load_dotenv()

LINE_TOKEN = (os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or "").strip()
LINE_USER_ID = (os.getenv("LINE_USER_ID") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

if not LINE_TOKEN or not LINE_USER_ID:
    raise RuntimeError("請先在 .env 設定 LINE_CHANNEL_ACCESS_TOKEN 與 LINE_USER_ID")

if not OPENAI_API_KEY:
    raise RuntimeError("請先在 .env 設定 OPENAI_API_KEY")

bad_positions = [(i, ord(ch)) for i, ch in enumerate(OPENAI_API_KEY) if ord(ch) > 127]
if bad_positions:
    raise RuntimeError(
        f"OPENAI_API_KEY 含非 ASCII 字元，位置/碼點：{bad_positions[:5]}..."
        "請重新複製貼上 key（不要引號/空白/全形字/換行）"
    )

client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_MODEL = "gpt-4o-mini"

def extract_video_id(youtube_url: str) -> str:
    m = re.search(r"(v=|youtu\.be/)([A-Za-z0-9_\-]{11})", youtube_url)
    if not m:
        raise ValueError("解析不到 videoId，請確認你貼的是 YouTube 影片連結")
    return m.group(2)

def line_push(text: str) -> None:
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json; charset=utf-8"
    }
    payload = {"to": LINE_USER_ID, "messages": [{"type": "text", "text": text[:4900]}]}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    r = requests.post(url, headers=headers, data=data, timeout=60)
    r.raise_for_status()

def fetch_cc_transcript_text(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    transcript_data = api.fetch(video_id, languages=["zh-Hant", "zh-TW", "zh", "en"])
    return "\n".join([item.text for item in transcript_data])

def dedupe_lines(text: str) -> list:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen = set()
    unique = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            unique.append(ln)
    return unique

def compress_transcript_for_single_call(text: str, max_chars: int = 24000) -> str:
    lines = dedupe_lines(text)
    joined = "\n".join(lines)
    if len(joined) <= max_chars:
        return joined

    avg_len = max(10, int(len(joined) / max(1, len(lines))))
    approx_lines = max(200, min(len(lines), max_chars // (avg_len + 1)))
    if approx_lines >= len(lines):
        return joined[:max_chars]

    step = len(lines) / approx_lines
    picked = []
    i = 0.0
    while int(i) < len(lines) and len(picked) < approx_lines:
        picked.append(lines[int(i)])
        i += step

    compressed = "\n".join(picked)
    return compressed[:max_chars]

def openai_single_summary(transcript_text: str, youtube_url: str) -> str:
    compressed = compress_transcript_for_single_call(transcript_text, max_chars=24000)
    input_text = (
        f"來源影片：{youtube_url}\n"
        "以下為影片字幕（已去重與壓縮，仍涵蓋全片主要段落）。\n"
        "請依照模板輸出最終摘要。\n\n"
        f"{compressed}"
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SUMMARY_TEMPLATE,
        input=input_text,
    )
    return (resp.output_text or "").strip()

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=GtOtp8g6UkY"

    try:
        video_id = extract_video_id(youtube_url)
        transcript_text = fetch_cc_transcript_text(video_id)
        final_summary = openai_single_summary(transcript_text, youtube_url)
        line_push(final_summary + f"\n\n來源：{youtube_url}")
        print("SUCCESS: sent summary to LINE")

    except Exception:
        err = traceback.format_exc()
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write(err)
        print("ERROR: failed. Please open error_log.txt for the full traceback.")
        try:
            line_push("Program failed. Please check error_log.txt in your project folder.")
        except Exception:
            pass