# 匯入 os：用來讀取環境變數（.env 裡的 token、userId）
import os

# 匯入 re：用來從 YouTube 連結中解析 videoId
import re

# 匯入 requests：用來呼叫 LINE Push API
import requests

# 匯入 load_dotenv：用來讀取 .env 檔
from dotenv import load_dotenv

# 匯入 YouTubeTranscriptApi：用來抓 YouTube CC 字幕
from youtube_transcript_api import YouTubeTranscriptApi


# 讀取 .env 檔，讓裡面的變數可以被 os.getenv() 取得
load_dotenv()

# 從 .env 取得 LINE Channel Access Token
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# 從 .env 取得你的 LINE userId
LINE_USER_ID = os.getenv("LINE_USER_ID")


def extract_video_id(youtube_url: str) -> str:
    """從 YouTube 網址中解析出 11 碼 videoId"""

    # 用正規表達式找出網址中的 videoId（支援 watch?v= 和 youtu.be/ 兩種）
    match = re.search(r"(v=|youtu\.be/)([A-Za-z0-9_\-]{11})", youtube_url)

    # 如果沒有找到，就丟出錯誤
    if not match:
        raise ValueError("解析不到 videoId，請確認你貼的是 YouTube 影片連結")

    # 回傳第 2 個括號群組，也就是 11 碼 videoId
    return match.group(2)


def fetch_transcript_text(video_id: str) -> str:
    """用 youtube-transcript-api 抓字幕（你目前的版本需要用 api.fetch）"""

    # 建立 API 物件（你目前安裝的版本是用物件方法 fetch）
    api = YouTubeTranscriptApi()

    # 抓字幕（依序嘗試：繁中 → 中文 → 英文）
    transcript_data = api.fetch(
        video_id,
        languages=["zh-Hant", "zh-TW", "zh", "en"]
    )

    # transcript_data 是 list，每筆像 {"text": "...", "start": 12.3, "duration": 3.4}
    # 把每一句的 text 串起來變成完整文字
    # transcript_data 裡每個 item 是物件，不是 dict，所以用 item.text 取字幕文字
    full_text = "\n".join([item.text for item in transcript_data])
    # 回傳字幕全文
    return full_text


def simple_summary(text: str) -> str:
    """先做不靠 AI 的 MVP 摘要（之後我們再換成 LLM 摘要）"""

    # 取前 800 字當作前段大意（先驗證流程）
    preview = text[:800]

    # 總經常見關鍵字（你可以依你的需求增減）
    keywords = ["通膨", "CPI", "PCE", "利率", "降息", "升息", "聯準會", "Fed",
                "美元", "殖利率", "公債", "就業", "失業率", "衰退", "GDP", "PMI", "AI"]

    # 找出字幕中命中的關鍵字
    hits = [k for k in keywords if k in text]

    # 把命中的關鍵字串起來，如果沒有就顯示提示
    hits_text = "、".join(hits) if hits else "（未偵測到常見總經關鍵字）"

    # 組合成摘要文字
    return (
        "📌 YouTube CC 字幕摘要（MVP）\n\n"
        "A) 前段大意（字幕前 800 字）：\n"
        f"{preview}\n\n"
        "B) 命中關鍵字：\n"
        f"{hits_text}\n"
    )


def line_push(text: str) -> None:
    """推播訊息到你的 LINE"""

    # LINE Push Message API 的網址
    url = "https://api.line.me/v2/bot/message/push"

    # HTTP headers：用 token 做授權
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json"
    }

    # 推播 payload：推給誰、推什麼
    payload = {
        "to": LINE_USER_ID,
        "messages": [
            {
                "type": "text",
                # LINE 文字長度限制，保守切到 4900 字
                "text": text[:4900]
            }
        ]
    }

    # 送出 POST 請求
    response = requests.post(url, headers=headers, json=payload, timeout=60)

    # 若不是成功狀態碼，會丟出錯誤（方便除錯）
    response.raise_for_status()


if __name__ == "__main__":
    # 你要測試的 YouTube 影片連結（你提供的這支）
    youtube_url = "https://www.youtube.com/watch?v=GtOtp8g6UkY"

    # 從連結解析出 videoId
    video_id = extract_video_id(youtube_url)

    try:
        # 1) 抓字幕全文
        transcript_text = fetch_transcript_text(video_id)

        # 2) 做 MVP 摘要
        summary = simple_summary(transcript_text)

        # 3) 推播到 LINE（附上來源連結）
        line_push(summary + f"\n來源：{youtube_url}")

        # 4) 終端機提示成功
        print("✅ 成功：已抓到 CC 字幕並推播到 LINE")

    except Exception as e:
        # 如果抓字幕或推播失敗，就推播錯誤訊息（方便你遠端也知道失敗原因）
        line_push(f"⚠️ 失敗：{youtube_url}\n錯誤：{e}")

        # 同時在終端機印出錯誤
        print("❌ 發生錯誤：", e)