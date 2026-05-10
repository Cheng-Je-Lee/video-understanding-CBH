"""
pipeline_core.py — Comic Book Hypothesis 核心函數庫
====================================================
所有 stage 的函數集中在這裡，供 run.py 呼叫。
不直接執行，只被 import。
"""

import os
import gc
import json
import time
import queue
import pickle
import threading
import subprocess

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════
# 路徑常數
# ══════════════════════════════════════════════════════

BASE_DIR     = "/content"
VIDEO_DIR    = f"{BASE_DIR}/videos"
AUDIO_DIR    = f"{BASE_DIR}/audios"
KF_DIR_V1    = f"{BASE_DIR}/keyframes_v1"
KF_DIR_V2    = f"{BASE_DIR}/keyframes_v2"
COMICS_V1    = f"{BASE_DIR}/comics_v1"
COMICS_V2    = f"{BASE_DIR}/comics_v2"
COMICS_V3    = f"{BASE_DIR}/comics_v3"
MEMORIES_DIR = f"{BASE_DIR}/memories"
PROFILES_DIR = f"{BASE_DIR}/video_profiles"
RESULTS_DIR  = f"{BASE_DIR}/results"

for d in [VIDEO_DIR, AUDIO_DIR, KF_DIR_V1, KF_DIR_V2,
          COMICS_V1, COMICS_V2, COMICS_V3,
          MEMORIES_DIR, PROFILES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════
# Stage 1 — 模型載入（lazy，只在第一次呼叫時載入）
# ══════════════════════════════════════════════════════

_models = {}

def get_clip():
    if "clip" not in _models:
        import torch
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        _models["clip"]       = clip_model
        _models["preprocess"] = preprocess
        _models["device"]     = device
        print(f"  CLIP 載入完成（{device}）")
    return _models["clip"], _models["preprocess"], _models["device"]

def get_whisper():
    if "whisper" not in _models:
        import whisper
        _models["whisper"] = whisper.load_model("base")
        print("  Whisper 載入完成")
    return _models["whisper"]

def get_phi3():
    if "phi3" not in _models:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        model_id = "microsoft/Phi-3-vision-128k-instruct"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        phi3 = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            _attn_implementation="eager"
        )
        _models["phi3"]      = phi3
        _models["processor"] = processor
        print("  Phi-3 Vision 載入完成")
    return _models["phi3"], _models["processor"]


# ══════════════════════════════════════════════════════
# Stage 2 — 下載影片 + 抽取音訊
# ══════════════════════════════════════════════════════

def download_video(label, url):
    """下載 YouTube 影片，回傳 video_path。已存在則跳過。"""
    import yt_dlp
    video_path = f"{VIDEO_DIR}/{label}.mp4"
    if os.path.exists(video_path):
        print(f"  影片已存在，跳過下載")
        return video_path
    ydl_opts = {
        "format":  "mp4[height<=480]/best[height<=480]",
        "outtmpl": f"{VIDEO_DIR}/{label}.%(ext)s",
        "quiet":   True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"  影片下載完成：{video_path}")
    return video_path

def extract_audio(label, video_path):
    """用 ffmpeg 抽取 16kHz mono 音訊，回傳 audio_path。已存在則跳過。"""
    audio_path = f"{AUDIO_DIR}/{label}.wav"
    if os.path.exists(audio_path):
        print(f"  音訊已存在，跳過抽取")
        return audio_path
    result = subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-y", audio_path
    ], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失敗：{result.stderr.decode()[:200]}")
    print(f"  音訊抽取完成：{audio_path}")
    return audio_path


# ══════════════════════════════════════════════════════
# Stage 3 — CLIP 語意編碼 + 關鍵幀選取
# ══════════════════════════════════════════════════════

def _encode_video_range(video_path, target_seconds):
    """
    內部函數：對指定的秒數列表進行 CLIP 編碼。
    target_seconds 是一個 set 或 list，包含要編碼的秒數。
    回傳 (embeddings, timestamps)。
    """
    import torch
    clip_model, preprocess, device = get_clip()

    target_set = set(target_seconds)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps))

    embeddings, timestamps = [], []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            second = frame_idx // frame_interval
            if second in target_set:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor  = preprocess(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = clip_model.encode_image(tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy())
                timestamps.append(second)
            # 若已超過 target_seconds 的最大值，提前結束
            if target_seconds and second > max(target_set) + 2:
                break
        frame_idx += 1
    cap.release()

    if not embeddings:
        return np.zeros((0, 512)), []
    return np.vstack(embeddings), timestamps


def encode_video_probe(video_path, label):
    """
    Stage 3a：S2 探針 CLIP 編碼。
    只編碼前 100 幀：0~20 秒 + 60~100 秒。
    不需要知道影片總長度，適合冷啟動。

    回傳 (probe_embeddings, probe_timestamps)。
    """
    # S2 窗口：0~20s + 60~100s
    target_seconds = list(range(0, 21)) + list(range(60, 101))
    emb, ts = _encode_video_range(video_path, target_seconds)
    print(f"  S2 探針 CLIP：{len(emb)} 幀（0~20s + 60~100s）")
    return emb, ts


def encode_video_full(video_path, label):
    """
    Stage 3c：全片 CLIP 編碼，每秒一幀。
    在 S2 探針完成之後才呼叫，確保 threshold/r₀/λ 已初始化。

    回傳 (embeddings, timestamps)。
    """
    import torch
    clip_model, preprocess, device = get_clip()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps))

    embeddings, timestamps = [], []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            second  = frame_idx // frame_interval
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor  = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())
            timestamps.append(second)
        frame_idx += 1
    cap.release()

    embeddings = np.vstack(embeddings)
    print(f"  全片 CLIP 編碼完成：{len(embeddings)} 幀")
    return embeddings, timestamps


def encode_video(video_path, label):
    """
    向下相容保留。等同於 encode_video_full()。
    新 notebook 請改用 encode_video_probe() + encode_video_full()。
    """
    return encode_video_full(video_path, label)

def select_keyframes_v1(embeddings, timestamps, threshold=None):
    """
    V1：相鄰幀餘弦變化 > threshold 即選為關鍵幀。

    threshold 來源：
      傳入 probe_stats["threshold_v1"] → S2 探針驅動（推薦，論文架構一致）
      不傳入（None）               → 從全片 embeddings 自算（向下相容）
    """
    change = np.array([
        1 - cosine_similarity(embeddings[j:j+1], embeddings[j+1:j+2])[0][0]
        for j in range(len(embeddings) - 1)
    ])
    if threshold is None:
        threshold = np.mean(change) + np.std(change)
        print(f"  V1 threshold（全片自算）：{threshold:.4f}")
    else:
        print(f"  V1 threshold（S2 探針）：{threshold:.4f}")
    times = [timestamps[j] for j, r in enumerate(change) if r > threshold]
    print(f"  V1 關鍵幀：{len(times)} 個")
    return times

def select_keyframes_v2(embeddings, timestamps, window=5, threshold=None):
    """
    V2：cosine 突變 AND 滑動平均漂移，帶單邊 Temporal Tolerance。

    threshold 來源：
      傳入 probe_stats["threshold_v2"] → S2 探針驅動（推薦，論文架構一致）
      不傳入（None）               → 從全片 embeddings 自算（向下相容）

    原版問題：
      cosine 突變峰值出現在場景切換當下 t，
      drift 峰值因滑動窗口累積延遲，出現在 t + delay。
      兩者直接取交集會因時間偏差而得到空集合。

    修正：單邊容忍窗口 δ = window_size。
      對每個 cosine 突變幀 t，在 [t, t+δ] 範圍內搜尋是否有 drift 峰值。
      drift 只會落後於 cosine，不會超前，所以只往後看。
    """
    # ── cosine 相鄰突變 ──
    change = np.array([
        1 - cosine_similarity(embeddings[j:j+1], embeddings[j+1:j+2])[0][0]
        for j in range(len(embeddings) - 1)
    ])
    if threshold is None:
        cosine_threshold = np.mean(change) + np.std(change)
        print(f"  V2 cosine threshold（全片自算）：{cosine_threshold:.4f}")
    else:
        cosine_threshold = threshold
        print(f"  V2 cosine threshold（S2 探針）：{cosine_threshold:.4f}")
    path1 = set(j for j, r in enumerate(change) if r > cosine_threshold)

    # ── 滑動平均漂移（drift threshold 仍從全片自算，因為 probe_stats 沒有 drift 的校正值）──
    drift = []
    for i in range(len(embeddings)):
        start = max(0, i - window)
        wm    = embeddings[start:i+1].mean(axis=0, keepdims=True)
        wm    = wm / np.linalg.norm(wm)
        drift.append(1 - cosine_similarity(embeddings[i:i+1], wm)[0][0])
    drift = np.array(drift)
    drift_threshold = np.mean(drift) + np.std(drift)
    path2 = set(j for j, r in enumerate(drift) if r > drift_threshold)

    # ── 單邊 Temporal Tolerance：δ = window_size ──
    delta = window
    indices = []
    for t_idx in sorted(path1):
        search_range = range(t_idx, min(t_idx + delta + 1, len(embeddings)))
        if any(j in path2 for j in search_range):
            indices.append(t_idx)

    times = [timestamps[i] for i in indices]

    print(f"  V2 關鍵幀：{len(times)} 個")
    print(f"    cosine 突變={len(path1)}, drift 峰值={len(path2)}")
    print(f"    原版交集={len(path1 & path2)}, Temporal Tolerance(δ={delta}s)={len(indices)}")
    return times


def save_screenshots(video_path, label, keyframe_times, kf_dir):
    """截取關鍵幀存圖"""
    out_dir = f"{kf_dir}/{label}"
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for t in keyframe_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t * fps)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{out_dir}/frame_{t:04d}s.jpg", frame)
    cap.release()


# ══════════════════════════════════════════════════════
# Stage 4 — Whisper 轉錄 + 語言偵測 + video_profile
# ══════════════════════════════════════════════════════

def transcribe(label, audio_path, confidence_threshold=0.7):
    """Whisper 轉錄，含語言信心門檻過濾。回傳 (segments, lang, conf, audio_type)"""
    whisper_model = get_whisper()
    result        = whisper_model.transcribe(audio_path, verbose=False)
    lang          = result.get("language", "unknown")
    conf          = result.get("language_probability", 1.0)
    print(f"  語言：{lang}（信心：{conf:.2f}）")

    if conf < confidence_threshold:
        print(f"  ⚠ 信心低，強制用 'en' 重跑")
        result = whisper_model.transcribe(audio_path, language="en", verbose=False)
        lang, conf = "en", 1.0

    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result["segments"] if s["text"].strip()
    ]

    # [STUB] 非英文翻譯（方向一）
    if lang not in ("en", "unknown") and segments:
        print(f"  [STUB] '{lang}' 非英文，翻譯功能未實作，保留原文")

    if   len(segments) == 0: audio_type = "silent"
    elif len(segments) <  5: audio_type = "sparse"
    else:                    audio_type = "rich"

    print(f"  audio_type：{audio_type}（{len(segments)} 段）")
    return segments, lang, conf, audio_type

def build_video_profile(label, lang, conf, audio_type, video_id=None, url=None):
    """建立 video_profile.json，stub 欄位保留預設值"""
    profile = {
        "video_id":              video_id or label,
        "label":                 label,
        "url":                   url or "",
        "language":              lang,
        "language_confidence":   round(conf, 3),
        "audio_type":            audio_type,
        "semantic_volatility":   "unknown",   # [STUB] 方向二
        "recommended_algorithm": "V2",         # [STUB] 方向二
        "threshold":             1.0,          # [STUB] 方向二 Bayesian opt
        "exclude_from_analysis": False,
        "notes":                 "no whisper segments" if audio_type == "silent" else ""
    }
    path = f"{PROFILES_DIR}/{label}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return profile


# ══════════════════════════════════════════════════════
# Stage 5 — 漫畫格組裝
# ══════════════════════════════════════════════════════

def get_dialogue(segments, start_time, end_time):
    text = " ".join(s["text"] for s in segments
                    if s["start"] >= start_time and s["start"] < end_time)
    return text if text else "[no dialogue]"

def apply_temporal_padding(panels, fill_interval=30, video_path=None, label=None, kf_dir=None):
    """
    動態採樣補幀（Dynamic Sampling Padding）。

    當相鄰兩個關鍵幀間距超過 fill_interval 秒時，
    每隔 fill_interval 秒從影片截取即時畫面作為補幀。

    比舊版「複製上一幀」更好：視覺資訊隨時間真實更新。
    同時解決 V2 關鍵幀為 0 的問題：整部影片每 30 秒至少有一張截圖。

    若 video_path 為 None，退回舊版行為（複製上一幀）。
    """
    # ── 處理 V2 關鍵幀為 0 的特殊情況 ──
    # 若 panels 為空且有影片路徑，每 fill_interval 秒截一張圖
    if not panels and video_path and os.path.exists(video_path):
        cap     = cv2.VideoCapture(video_path)
        total_s = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        print(f"  ⚠ 關鍵幀為 0，改用動態採樣（每 {fill_interval} 秒截圖）")
        dummy_panels = []
        for ft in range(0, total_s, fill_interval):
            dummy_panels.append({
                "time":    ft,
                "image":   f"frame_{ft:04d}s.jpg",
                "text":    "",
                "is_fill": True,
                "is_dynamic": True,
            })
        panels = dummy_panels

    # ── 截取即時畫面 ──
    def grab_frame(t):
        """從影片截取第 t 秒的畫面，存成 jpg，回傳檔名。"""
        if not video_path or not os.path.exists(video_path):
            return None
        if not kf_dir or not label:
            return None
        out_dir = f"{kf_dir}/{label}"
        os.makedirs(out_dir, exist_ok=True)
        fname   = f"frame_{t:04d}s.jpg"
        fpath   = f"{out_dir}/{fname}"
        if os.path.exists(fpath):
            return fname
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, t * fps)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(fpath, frame)
            return fname
        return None

    # ── 主要填充邏輯 ──
    filled = []
    for i, panel in enumerate(panels):
        filled.append(panel)
        if i < len(panels) - 1:
            gap = panels[i+1]["time"] - panel["time"]
            if gap > fill_interval:
                for ft in range(panel["time"] + fill_interval, panels[i+1]["time"], fill_interval):
                    if video_path:
                        # 動態採樣：截取即時畫面
                        fname = grab_frame(ft)
                        filled.append({
                            "time":       ft,
                            "image":      fname,
                            "text":       "",
                            "is_fill":    True,
                            "is_dynamic": True,
                        })
                    else:
                        # 退回舊版：複製上一幀
                        filled.append({**panel, "time": ft, "is_fill": True})
    return filled

def build_panels(label, keyframe_times, segments):
    panels = []
    kf     = sorted(keyframe_times)
    for i, t in enumerate(kf):
        start = kf[i-1] if i > 0 else 0
        panels.append({
            "time":    t,
            "image":   f"frame_{t:04d}s.jpg",
            "text":    get_dialogue(segments, start, t),
            "is_fill": False
        })
    return panels

def build_v3_panels(segments, interval=10):
    if not segments:
        return []
    max_t  = int(segments[-1]["end"]) + interval
    panels = []
    for t in range(0, max_t, interval):
        text = " ".join(s["text"] for s in segments
                        if s["start"] >= t and s["start"] < t + interval)
        if text:
            panels.append({"time": t, "image": None, "text": text, "is_fill": False})
    return panels

def write_index(panels, out_dir, label, version, has_image=True, video_id=None, url=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/index.txt", "w", encoding="utf-8") as f:
        f.write(f"=== {label} ({version}) ===\n")
        f.write(f"label    : {label}\n")
        f.write(f"method   : {version}\n")
        if video_id:
            f.write(f"video_id : {video_id}\n")
        if url:
            f.write(f"url      : {url}\n")
        f.write(f"\n")
        for p in panels:
            fill = " [fill]" if p.get("is_fill") else ""
            f.write(f"[{p['time']}s]{fill}\n")
            if has_image and p.get("image"):
                f.write(f"圖片：{p['image']}\n")
            f.write(f"台詞：{p['text']}\n")
            f.write("-" * 40 + "\n")


# ══════════════════════════════════════════════════════
# Stage 6 — Phi-3 Vision 逐格理解
# ══════════════════════════════════════════════════════

def extract_response(full_response):
    if "right now?" in full_response:
        answer = full_response.split("right now?")[-1].strip()
    elif "assistant" in full_response:
        answer = full_response.split("assistant")[-1].strip()
    else:
        answer = full_response.strip()
    if "." in answer:
        answer = answer.split(".")[0].strip() + "."
    return answer

def load_index(index_path):
    panels, current_time, current_img = [], None, None
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and "s]" in line:
                try:    current_time = int(line.split("s]")[0][1:])
                except: pass
            elif line.startswith("圖片："):
                current_img = line[3:]
            elif line.startswith("台詞：") and current_time is not None:
                panels.append({"time": current_time, "image": current_img, "text": line[3:]})
                current_img = None
    return panels

def run_phi3_on_panels(label, method, panels, kf_dir):
    """
    用 Phi-3 處理 panels list，回傳 (memories, timing_logs)。
    使用 queue + worker thread 模擬即時播放節奏。
    時間模擬為方向五（即時考試）的基礎設計，維持不動。
    """
    import torch
    phi3, processor = get_phi3()

    memories, timing_logs = [], []
    pq = queue.Queue()

    def worker():
        while True:
            try:
                item = pq.get(timeout=2)
                if item is None:
                    break
                panel, sent_time = item
                t         = panel["time"]
                dialogue  = panel["text"]
                img_fname = panel.get("image")
                start     = time.time()

                use_image = bool(img_fname and method != "v3")
                if use_image:
                    img_path  = f"{kf_dir}/{label}/{img_fname}"
                    use_image = os.path.exists(img_path)

                if use_image:
                    img = Image.open(img_path)
                    messages = [{"role": "user", "content":
                        f"<|image_1|>\nYou are watching a video in real time. "
                        f"Current time: {t} seconds.\nDialogue: {dialogue}\n"
                        f"In one sentence, what is the key narrative event happening right now?"}]
                    prompt = processor.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(prompt, [img], return_tensors="pt").to("cuda")
                else:
                    messages = [{"role": "user", "content":
                        f"You are watching a video in real time. "
                        f"Current time: {t} seconds.\nDialogue: {dialogue}\n"
                        f"In one sentence, what is the key narrative event happening right now?"}]
                    prompt = processor.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor.tokenizer(prompt, return_tensors="pt").to("cuda")

                with torch.no_grad():
                    output = phi3.generate(**inputs, max_new_tokens=60, do_sample=False)

                description  = extract_response(
                    processor.tokenizer.decode(output[0], skip_special_tokens=True))
                finish       = time.time()
                process_time = finish - start
                delay        = finish - sent_time

                memories.append({"time": t, "description": description, "dialogue": dialogue})
                timing_logs.append({
                    "frame_time": t, "process_time": process_time,
                    "delay": delay, "used_image": use_image
                })

                del inputs, output
                torch.cuda.empty_cache()
                gc.collect()

                mode = "IMG" if use_image else "TXT"
                fill = " [fill]" if panel.get("is_fill") else ""
                print(f"  [{method}][{t:4d}s]{fill} {process_time:.1f}s | {mode} | {description[:55]}")
                pq.task_done()
            except queue.Empty:
                continue

    worker_thread = threading.Thread(target=worker, daemon=False)
    worker_thread.start()

    start_real = time.time()
    for panel in panels:
        wait = panel["time"] - (time.time() - start_real)
        if wait > 0:
            time.sleep(wait)
        pq.put((panel, time.time()))

    pq.join()
    pq.put(None)
    worker_thread.join()

    return memories, timing_logs



def run_phi3_batch(label, method, panels, kf_dir):
    """
    Phi-3 批次處理：直接依序處理所有 panels，不模擬播放時間。
    速度最快，適合整合測試與一般實驗。
    """
    import torch
    phi3, processor = get_phi3()

    memories, timing_logs = [], []

    for panel in panels:
        t         = panel["time"]
        dialogue  = panel["text"]
        img_fname = panel.get("image")
        start     = time.time()

        use_image = bool(img_fname and method != "v3")
        if use_image:
            img_path  = f"{kf_dir}/{label}/{img_fname}"
            use_image = os.path.exists(img_path)

        if use_image:
            img = Image.open(img_path)
            messages = [{"role": "user", "content":
                f"<|image_1|>\nYou are watching a video in real time. "
                f"Current time: {t} seconds.\nDialogue: {dialogue}\n"
                f"In one sentence, what is the key narrative event happening right now?"}]
            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(prompt, [img], return_tensors="pt").to("cuda")
        else:
            messages = [{"role": "user", "content":
                f"You are watching a video in real time. "
                f"Current time: {t} seconds.\nDialogue: {dialogue}\n"
                f"In one sentence, what is the key narrative event happening right now?"}]
            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = phi3.generate(**inputs, max_new_tokens=60, do_sample=False)

        description  = extract_response(
            processor.tokenizer.decode(output[0], skip_special_tokens=True))
        process_time = time.time() - start

        memories.append({"time": t, "description": description, "dialogue": dialogue})
        timing_logs.append({
            "frame_time": t, "process_time": process_time, "used_image": use_image
        })

        del inputs, output
        torch.cuda.empty_cache()
        gc.collect()

        mode = "IMG" if use_image else "TXT"
        fill = " [fill]" if panel.get("is_fill") else ""
        print(f"  [{method}][{t:4d}s]{fill} {process_time:.1f}s | {mode} | {description[:55]}")

    return memories, timing_logs

def save_memory_txt(label, method, memories, url=None, video_id=None):
    """
    輸出純敘述流格式，方便直接貼給 LLM 答題。
    格式：
        [14s] A soothsayer warns Shen's parents about a white demon.
        [42s] Shen destroys the village in rage.
        ...
    """
    out_path = f"{MEMORIES_DIR}/{method}_{label}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== Video memory ===\n")
        f.write(f"label    : {label}\n")
        f.write(f"method   : {method}\n")
        f.write(f"video_id : {video_id if video_id else '(未指定)'}\n")
        f.write(f"url      : {url if url else ''}\n")
        f.write(f"frames   : {len(memories)}\n")
        f.write(f"\n")
        for m in sorted(memories, key=lambda x: x["time"]):
            f.write(f"[{m['time']}s] {m['description']}\n")
    print(f"  記憶存至：{out_path}（{len(memories)} 條）")
    return out_path


# ══════════════════════════════════════════════════════
# 記憶架構 — 結構化 JSON 存儲（方向六）
# ══════════════════════════════════════════════════════

def save_memory_json(label, method, memories, url=None, video_id=None, kf_dir=None):
    """
    輸出結構化 JSON 格式，供分層記憶查詢系統使用。
    與 save_memory_txt 並行輸出，txt 保持不動。

    每筆 panel 包含：
      - panel_id    : 唯一識別碼（video_id_method_timestamp）
      - timestamp   : 影片秒數
      - description : Phi-3 輸出的一句話敘述
      - dialogue    : 對應的 Whisper 台詞
      - frame_path  : 關鍵幀圖片路徑（漫畫格模式用，v3 為 None）
    """
    sorted_memories = sorted(memories, key=lambda x: x["time"])

    panels = []
    for m in sorted_memories:
        t   = m["time"]
        vid = video_id if video_id else "unknown"

        # 推算圖片路徑（v3 純文字沒有圖片）
        if method != "v3" and kf_dir:
            frame_path = f"{kf_dir}/{label}/frame_{t:04d}s.jpg"
            if not os.path.exists(frame_path):
                frame_path = None
        else:
            frame_path = None

        panels.append({
            "panel_id":    f"{vid}_{method}_{t:04d}",
            "timestamp":   t,
            "description": m["description"],
            "dialogue":    m.get("dialogue", ""),
            "frame_path":  frame_path,
        })

    output = {
        "meta": {
            "label":        label,
            "method":       method,
            "video_id":     video_id if video_id else "unknown",
            "url":          url if url else "",
            "total_panels": len(panels),
        },
        "panels": panels,
    }

    out_path = f"{MEMORIES_DIR}/{method}_{label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  JSON 記憶存至：{out_path}（{len(panels)} 筆）")
    return out_path


# ══════════════════════════════════════════════════════
# 記憶架構 — 短期記憶查詢系統（方向六）
# ══════════════════════════════════════════════════════

def load_memory_json(label, method):
    """
    從磁碟讀取結構化 JSON 記憶檔，回傳 (meta, panels)。
    panels 已按時間戳排序。
    """
    path = f"{MEMORIES_DIR}/{method}_{label}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到記憶檔：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta   = data["meta"]
    panels = sorted(data["panels"], key=lambda x: x["timestamp"])
    print(f"  載入記憶：{path}（{len(panels)} 筆）")
    return meta, panels


def query_short_term(panels, current_time, user_question, top_k=5):
    """
    給定當前時間戳和使用者問題，從過去的 panels 中撈出語意最相關的 top_k 筆。

    用 CLIP 文字編碼器將使用者問題 encode 成向量，
    和每個 panel 的 description + dialogue 做語意相似度比較，
    取相似度最高的 top_k 個 panel，依時間戳排序後回傳。

    回傳：
      selected_panels — 依時間戳排序的 top_k panels
      selected_scores — 對應的 cosine 相似度分數
    """
    import torch
    import clip as clip_lib

    clip_model, preprocess, device = get_clip()

    # 只取 current_time 之前的 panels
    past = [p for p in panels if p["timestamp"] <= current_time]
    if not past:
        return [], []
    if len(past) == 1:
        return past, [1.0]

    # 使用者問題 → 文字向量
    with torch.no_grad():
        q_token = clip_lib.tokenize([user_question], truncate=True).to(device)
        q_vec   = clip_model.encode_text(q_token)
        q_vec   = q_vec / q_vec.norm(dim=-1, keepdim=True)
        q_np    = q_vec.cpu().numpy()

    # 每個 panel 的 description + dialogue → 文字向量 → cosine 相似度
    scores = []
    for p in past:
        txt = f"{p['description']} {p['dialogue']}".strip()
        with torch.no_grad():
            t_token = clip_lib.tokenize([txt], truncate=True).to(device)
            t_vec   = clip_model.encode_text(t_token)
            t_vec   = t_vec / t_vec.norm(dim=-1, keepdim=True)
            t_np    = t_vec.cpu().numpy()
        scores.append(float(cosine_similarity(q_np, t_np)[0][0]))

    # 取 top_k，依時間戳排序回傳
    ranked          = sorted(zip(scores, past), key=lambda x: x[0], reverse=True)[:top_k]
    ranked_sorted   = sorted(ranked, key=lambda x: x[1]["timestamp"])
    selected_panels = [p for _, p in ranked_sorted]
    selected_scores = [s for s, _ in ranked_sorted]

    print(f"  短期查詢：從 {len(past)} 個 panels 取 top {len(selected_panels)}")
    print(f"  相似度範圍：{min(selected_scores):.3f} ~ {max(selected_scores):.3f}")
    return selected_panels, selected_scores


def format_for_llm(scene_panels, mode="text"):
    """
    將 scene_panels 格式化為 LLM 輸入包。

    mode="text"  — 模式 A：只用 Phi-3 文字描述 + 台詞（輕量）
    mode="image" — 模式 B：回傳圖片路徑列表 + 對應台詞（漫畫格模式）

    模式 A 輸出（字串，直接貼進 prompt）：
        [42s] Shen destroys the village. | "Nothing can stop me now."
        [89s] Po trains with Shifu.      | "I am the Dragon Warrior!"

    模式 B 輸出（list of dict，供多圖 API 使用）：
        [
          {"timestamp": 42, "frame_path": "...", "dialogue": "Nothing can stop me now."},
          {"timestamp": 89, "frame_path": "...", "dialogue": "I am the Dragon Warrior!"},
        ]
    """
    if mode == "text":
        lines = []
        for p in scene_panels:
            dialogue = f' | "{p["dialogue"]}"' if p["dialogue"] else ""
            lines.append(f'[{p["timestamp"]}s] {p["description"]}{dialogue}')
        return "\n".join(lines)

    elif mode == "image":
        result = []
        for p in scene_panels:
            result.append({
                "timestamp":  p["timestamp"],
                "frame_path": p["frame_path"],
                "dialogue":   p["dialogue"],
            })
        return result

    else:
        raise ValueError(f"未知的 mode：{mode}，請用 'text' 或 'image'")


# ══════════════════════════════════════════════════════
# 記憶架構 — 即時情境快照（方向六）
# ══════════════════════════════════════════════════════

INSTANT_DIR = f"{BASE_DIR}/instant"
os.makedirs(INSTANT_DIR, exist_ok=True)

def build_instant_context(video_path, current_time, segments, panels, context_window=2):
    """
    使用者發問當下，組合三層資料成即時情境快照。

    三層資料：
      1. 實時截圖     — 從影片抓 current_time 那一幀，存至 /content/instant/
      2. 前後台詞     — Whisper segments 裡，current_time 前後各 context_window 句
      3. 最近漫畫格   — JSON 記憶裡時間戳最接近且在 current_time 之前的 panel

    回傳 dict：
      {
        "timestamp":     154,
        "frame_path":    "/content/instant/frame_0154s.jpg",
        "context_lines": ["I am the Dragon Warrior!", "Fire! Destroy them all!"],
        "nearest_panel": { panel dict }
      }
    """
    # ── 1. 實時截圖 ──
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_time * fps)
    ret, frame = cap.read()
    cap.release()

    frame_path = None
    if ret:
        frame_path = f"{INSTANT_DIR}/frame_{current_time:04d}s.jpg"
        cv2.imwrite(frame_path, frame)
        print(f"  即時截圖：{frame_path}")
    else:
        print(f"  ⚠ 截圖失敗（time={current_time}s）")

    # ── 2. 前後台詞 ──
    # 找出和 current_time 重疊或最接近的 segments
    nearby = []
    for seg in segments:
        if seg["start"] <= current_time <= seg["end"]:
            nearby.append(seg)
    # 如果沒有完全重疊的，找最近的 context_window 句
    if not nearby:
        dists = [(abs(seg["start"] - current_time), seg) for seg in segments]
        dists.sort(key=lambda x: x[0])
        nearby = [s for _, s in dists[:context_window * 2]]

    # 取前後各 context_window 句
    if nearby:
        center_idx = segments.index(nearby[0]) if nearby[0] in segments else 0
        start_idx  = max(0, center_idx - context_window)
        end_idx    = min(len(segments), center_idx + context_window + 1)
        context_segs = segments[start_idx:end_idx]
    else:
        context_segs = []

    context_lines = [seg["text"].strip() for seg in context_segs if seg["text"].strip()]
    print(f"  前後台詞：{len(context_lines)} 句")

    # ── 3. 最近漫畫格 ──
    past_panels = [p for p in panels if p["timestamp"] <= current_time]
    nearest_panel = past_panels[-1] if past_panels else None
    if nearest_panel:
        print(f"  最近漫畫格：[{nearest_panel['timestamp']}s] {nearest_panel['description'][:50]}")
    else:
        print(f"  ⚠ 尚無漫畫格記憶")

    return {
        "timestamp":     current_time,
        "frame_path":    frame_path,
        "context_lines": context_lines,
        "nearest_panel": nearest_panel,
    }


def format_instant_for_llm(instant_context):
    """
    將即時情境快照格式化為 LLM 輸入文字。
    圖片路徑另外回傳，供多模態 API 使用。

    回傳：
      text      — 文字描述（直接貼進 prompt）
      frame_path — 截圖路徑（None 表示截圖失敗）
    """
    ctx = instant_context
    lines = []

    # 最近漫畫格
    if ctx["nearest_panel"]:
        p = ctx["nearest_panel"]
        lines.append(f"[前一個關鍵幀 {p['timestamp']}s]")
        lines.append(f"畫面：{p['description']}")
        if p["dialogue"]:
            lines.append(f"台詞：{p['dialogue']}")
        lines.append("")

    # 當下台詞
    if ctx["context_lines"]:
        lines.append(f"[當下台詞（{ctx['timestamp']}s 前後）]")
        for line in ctx["context_lines"]:
            lines.append(f"  {line}")

    text = "\n".join(lines)
    return text, ctx["frame_path"]

# ══════════════════════════════════════════════════════
# 記憶架構 — 長期記憶兩層架構（方向六）
# ══════════════════════════════════════════════════════

LONG_TERM_DIR = f"{BASE_DIR}/long_term"
os.makedirs(LONG_TERM_DIR, exist_ok=True)


def build_long_term_memory(label, method, panels, meta, gemini_api_key):
    """
    影片看完後，將所有 panels 送給 Gemini 壓縮成結構化長期記憶。
    輸出存至 /content/long_term/<video_id>.json。

    兩層結構：
      第一層：title_summary + key_characters + chapters（輕量，優先查詢）
      第二層：panels 原始資料路徑（摘要不足時回去撈）
    """
    from google import genai
    client = genai.Client(api_key=gemini_api_key)

    # ── 組合所有 panel 描述成純文字 ──
    panel_text = ""
    for p in panels:
        t = p.get("timestamp") or p.get("time", 0)
        desc = p.get("description") or p.get("text", "")
        dialogue = f' | "{p["dialogue"]}"' if p.get("dialogue") else ""
        panel_text += f"[{t}s] {desc}{dialogue}\n"

    prompt = f"""以下是一部影片的逐格描述，請整理成結構化摘要。
只輸出純 JSON，不要加任何說明文字或 markdown。

格式：
{{
  "title_summary": "一句話描述這部影片的主題",
  "key_characters": ["角色1", "角色2"],
  "chapters": [
    {{
      "start": 開始秒數,
      "end": 結束秒數,
      "summary": "這個段落發生了什麼"
    }}
  ]
}}

章節劃分原則：
- 每個章節代表一個敘事段落或場景轉換
- 章節數量約為 panel 總數的 1/5 到 1/3
- summary 控制在 30 字以內

影片逐格描述：
{panel_text}"""

    print(f"  送給 Gemini 壓縮（{len(panels)} 個 panels）...")
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    raw = response.text.strip()

    # 清理可能的 markdown 包裝
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        summary = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON 解析失敗：{e}")
        print(f"  原始回應：{raw[:200]}")
        summary = {
            "title_summary": "(解析失敗)",
            "key_characters": [],
            "chapters": []
        }

    # ── 組合完整長期記憶檔 ──
    video_id = meta.get("video_id", "unknown")
    output   = {
        "meta": {
            "label":        label,
            "method":       method,
            "video_id":     video_id,
            "url":          meta.get("url", ""),
            "total_panels": len(panels),
        },
        "title_summary":  summary.get("title_summary", ""),
        "key_characters": summary.get("key_characters", []),
        "chapters":       summary.get("chapters", []),
        "panels_path":    f"{MEMORIES_DIR}/{method}_{label}.json",  # 第二層 RAG 指針
    }

    out_path = f"{LONG_TERM_DIR}/{video_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  長期記憶存至：{out_path}")
    print(f"  章節數：{len(output['chapters'])}，角色：{output['key_characters']}")
    return out_path


def query_long_term(user_question, gemini_api_key, top_chapters=2):
    """
    三步走查詢長期記憶庫。

    Step 1：掃所有影片的 title_summary + key_characters，找最相關的影片
    Step 2：在該影片的 chapters 裡找最相關的段落
    Step 3：回傳摘要文字；若需要更多細節，回傳 panels_path 供第二層 RAG 使用

    回傳：
      result dict：
        "matched_video"   — 最相關的影片 meta
        "matched_chapters"— 最相關的章節列表
        "answer_text"     — 格式化後可直接貼進 LLM prompt 的文字
        "panels_path"     — 原始 panels JSON 路徑（第二層 RAG 用）
        "needs_rag"       — 是否建議進一步撈原始 panels
    """
    import torch
    import clip as clip_lib

    clip_model, preprocess, device = get_clip()

    # ── Step 1：掃所有長期記憶檔，找最相關的影片 ──
    lt_files = [
        f for f in os.listdir(LONG_TERM_DIR)
        if f.endswith(".json")
    ]
    if not lt_files:
        return {"answer_text": "（尚無長期記憶）", "needs_rag": False}

    # 使用者問題 → CLIP 文字向量
    with torch.no_grad():
        q_token = clip_lib.tokenize([user_question], truncate=True).to(device)
        q_vec   = clip_model.encode_text(q_token)
        q_vec   = q_vec / q_vec.norm(dim=-1, keepdim=True)
        q_np    = q_vec.cpu().numpy()

    # 每部影片的 title_summary + key_characters → 相似度
    video_scores = []
    video_data   = []
    for fname in lt_files:
        with open(f"{LONG_TERM_DIR}/{fname}", "r", encoding="utf-8") as f:
            data = json.load(f)
        search_text = data["title_summary"] + " " + " ".join(data["key_characters"])
        with torch.no_grad():
            t_token = clip_lib.tokenize([search_text], truncate=True).to(device)
            t_vec   = clip_model.encode_text(t_token)
            t_vec   = t_vec / t_vec.norm(dim=-1, keepdim=True)
            t_np    = t_vec.cpu().numpy()
        sim = float(cosine_similarity(q_np, t_np)[0][0])
        video_scores.append(sim)
        video_data.append(data)

    best_idx  = int(np.argmax(video_scores))
    best_data = video_data[best_idx]
    print(f"  長期查詢 Step1：最相關影片 = {best_data['meta']['label']}（sim={video_scores[best_idx]:.3f}）")

    # ── Step 2：在該影片的 chapters 裡找最相關段落 ──
    chapters = best_data.get("chapters", [])
    if not chapters:
        matched_chapters = []
    else:
        ch_scores = []
        for ch in chapters:
            with torch.no_grad():
                c_token = clip_lib.tokenize([ch["summary"]], truncate=True).to(device)
                c_vec   = clip_model.encode_text(c_token)
                c_vec   = c_vec / c_vec.norm(dim=-1, keepdim=True)
                c_np    = c_vec.cpu().numpy()
            ch_scores.append(float(cosine_similarity(q_np, c_np)[0][0]))

        ranked_ch     = sorted(zip(ch_scores, chapters), key=lambda x: x[0], reverse=True)[:top_chapters]
        matched_chapters = [ch for _, ch in sorted(ranked_ch, key=lambda x: x[1]["start"])]
        print(f"  長期查詢 Step2：top {len(matched_chapters)} 章節，時段 "
              f"{matched_chapters[0]['start']}s ~ {matched_chapters[-1]['end']}s")

    # ── Step 3：組合輸出 ──
    lines = []
    lines.append(f"影片：{best_data['title_summary']}")
    lines.append(f"角色：{', '.join(best_data['key_characters'])}")
    lines.append("")
    for ch in matched_chapters:
        lines.append(f"[{ch['start']}s ~ {ch['end']}s] {ch['summary']}")

    # 如果匹配的章節相似度不高，建議進一步撈原始 panels
    top_sim    = max(ch_scores) if chapters else 0
    needs_rag  = top_sim < 0.25

    return {
        "matched_video":    best_data["meta"],
        "matched_chapters": matched_chapters,
        "answer_text":      "\n".join(lines),
        "panels_path":      best_data.get("panels_path"),
        "needs_rag":        needs_rag,
    }

# ══════════════════════════════════════════════════════
# 長期記憶 — Phi-3 多層壓縮（方向六，方案 A）
# ══════════════════════════════════════════════════════

def build_long_term_memory_phi3(label, method, panels, meta,
                                 chunk_size=20,
                                 mid_max_tokens=200,
                                 final_max_tokens=600):
    """
    用 Phi-3 做多層滾動壓縮，生成長期記憶。

    流程（方案 A：取代）：
      panels 1~20   → Phi-3 → 壓縮-1
      壓縮-1 + panels 21~40 → Phi-3 → 壓縮-2
      壓縮-2 + panels 41~60 → Phi-3 → 壓縮-3
      ...
      壓縮-N → Phi-3 → 最終結構化 JSON

    參數：
      chunk_size      每組 panel 數量（預設 20）
      mid_max_tokens  中間壓縮的 token 上限（預設 200）
      final_max_tokens 最終摘要的 token 上限（預設 600）

    輸出：
      /content/long_term/<video_id>.json
    """
    import torch
    phi3, processor = get_phi3()

    def phi3_compress(prompt_text, max_new_tokens):
        """用 Phi-3 文字模式壓縮，回傳壓縮後的文字。"""
        messages = [{"role": "user", "content": prompt_text}]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = phi3.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        result = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        # 取 assistant 回應部分
        if "assistant" in result.lower():
            result = result.split("assistant")[-1].strip()
        del inputs, output
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        return result.strip()

    # ── 把 panels 轉成文字 ──
    def panels_to_text(panel_list):
        lines = []
        for p in panel_list:
            t    = p.get("timestamp") or p.get("time", 0)
            desc = p.get("description") or p.get("text", "")
            dial = p.get("dialogue", "")
            line = f"[{t}s] {desc}"
            if dial:
                line += f' | "{dial}"'
            lines.append(line)
        return "\n".join(lines)

    # ── 分組 ──
    chunks = [panels[i:i+chunk_size] for i in range(0, len(panels), chunk_size)]
    print(f"  Phi-3 多層壓縮：{len(panels)} 個 panels → {len(chunks)} 組（每組最多 {chunk_size} 個）")

    # ── 滾動壓縮 ──
    current_summary = ""

    for idx, chunk in enumerate(chunks):
        chunk_text = panels_to_text(chunk)

        if current_summary:
            prompt = (
                f"You are summarizing a video in segments.\n\n"
                f"Previous summary:\n{current_summary}\n\n"
                f"New events:\n{chunk_text}\n\n"
                f"Write an updated summary combining both. "
                f"Be concise, keep key events and characters. "
                f"Maximum {mid_max_tokens} tokens."
            )
        else:
            prompt = (
                f"Summarize the following video events concisely.\n\n"
                f"{chunk_text}\n\n"
                f"Keep key events and characters. "
                f"Maximum {mid_max_tokens} tokens."
            )

        current_summary = phi3_compress(prompt, max_new_tokens=mid_max_tokens)
        print(f"  壓縮 {idx+1}/{len(chunks)} 完成（{len(current_summary)} 字元）")

    # ── 最終結構化摘要 ──
    final_prompt = (
        f"Based on this video summary, output a JSON object only. "
        f"No explanation, no markdown, just raw JSON.\n\n"
        f"Summary:\n{current_summary}\n\n"
        f"Output format:\n"
        f'{{"title_summary": "one sentence description", '
        f'"key_characters": ["name1", "name2"], '
        f'"chapters": [{{"start": 0, "end": 60, "summary": "what happened"}}]}}'
    )

    print(f"  生成最終結構化摘要...")
    final_raw = phi3_compress(final_prompt, max_new_tokens=final_max_tokens)

    # 清理可能的 markdown 包裝
    if "```" in final_raw:
        parts = final_raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            if part.strip().startswith("{"):
                final_raw = part.strip()
                break

    try:
        summary = json.loads(final_raw)
    except json.JSONDecodeError:
        print(f"  ⚠ JSON 解析失敗，儲存原始文字")
        summary = {
            "title_summary": current_summary[:200],
            "key_characters": [],
            "chapters": []
        }

    # ── 組合輸出 ──
    video_id = meta.get("video_id", "unknown")
    output = {
        "meta": {
            "label":        label,
            "method":       method,
            "video_id":     video_id,
            "url":          meta.get("url", ""),
            "total_panels": len(panels),
            "compression":  "phi3_hierarchical",
        },
        "title_summary":  summary.get("title_summary", ""),
        "key_characters": summary.get("key_characters", []),
        "chapters":       summary.get("chapters", []),
        "panels_path":    f"{MEMORIES_DIR}/{method}_{label}.json",
    }

    os.makedirs(LONG_TERM_DIR, exist_ok=True)
    out_path = f"{LONG_TERM_DIR}/{video_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  長期記憶存至：{out_path}")
    print(f"  章節數：{len(output['chapters'])}，角色：{output['key_characters']}")
    return out_path

# ══════════════════════════════════════════════════════
# Stage 3-probe — S2 探針冷啟動（新增）
# ══════════════════════════════════════════════════════

def compute_s2_probe(probe_embeddings, probe_timestamps, k=1.0):
    """
    S2 探針：根據 encode_video_probe() 回傳的幀計算全片視覺特徵估計。

    設計原則：
      encode_video_probe() 已負責只取 0~20s + 60~100s 的幀，
      本函數直接使用這些幀計算統計量，不再做二次篩選。
      影片太短（幀數 < 10）時仍可正常運作，用全部傳入的幀計算。

    策略說明：
      S2 窗口的 μ 代表性 r=0.916（108 支 Video-MME 驗證）。
      不需要知道影片總長度，適合冷啟動場景。

    線性校正（補償 S2 系統性偏差）：
      full_μ ≈ S2_μ + 0.017
      full_σ ≈ S2_σ - 0.012

    回傳：
      probe_stats dict，包含校正後的 mu/sigma/autocorr，
      以及各方法的建議初始參數。
    """
    n = len(probe_embeddings)

    # 幀數不足時發出警告（影片極短）
    if n < 4:
        print(f"  ⚠ S2 探針幀數不足（{n} 幀），改用全幀統計，結果可能不穩定")

    # ── 計算相鄰幀 cosine 變化 ──
    changes = np.array([
        1 - cosine_similarity(probe_embeddings[j:j+1], probe_embeddings[j+1:j+2])[0][0]
        for j in range(n - 1)
    ]) if n > 1 else np.array([0.0])

    s2_mu    = float(np.mean(changes))
    s2_sigma = float(np.std(changes))

    # ── 計算 lag-1 autocorrelation ──
    if len(changes) > 2:
        mean_c   = np.mean(changes)
        num      = np.sum((changes[:-1] - mean_c) * (changes[1:] - mean_c))
        den      = np.sum((changes - mean_c) ** 2)
        autocorr = float(num / den) if den > 1e-8 else 0.0
    else:
        autocorr = 0.0

    # ── OLS 線性校正（108 支 Video-MME 回歸，R²: μ=0.775, σ=0.603, ac=0.359）──
    # full_μ       = 0.8076 × S2_μ  + 0.1839
    # full_σ       = 0.7533 × S2_σ  + 0.0178
    # full_autocorr= 0.4836 × S2_ac + 0.0398
    mu_corrected    = 0.8076 * s2_mu    + 0.1839
    sigma_corrected = max(0.7533 * s2_sigma + 0.0178, 0.001)   # 下限避免 sigma=0
    autocorr        = 0.4836 * autocorr + 0.0398

    print(f"  S2 探針：{n} 幀（已由 encode_video_probe 預篩選）")
    print(f"  S2 raw   μ={s2_mu:.4f}, σ={s2_sigma:.4f}, autocorr（校正前）={autocorr:.4f}")
    print(f"  OLS校正  μ={mu_corrected:.4f}, σ={sigma_corrected:.4f}, autocorr={autocorr:.4f}")

    # ── 各方法建議參數 ──
    # V1/V2/V4 全部使用 μ + kσ（統計控制界限上限，論文架構完全一致）
    # k 由外部傳入（預設 1.0），對應 notebook 區段二的 K 變數
    # V4 λ 需要影片總長 T，不在此計算，由 select_keyframes_v4 傳入後推導
    threshold_v1 = mu_corrected + k * sigma_corrected
    threshold_v2 = threshold_v1     # V2 cosine path 同閾值
    r0           = mu_corrected + k * sigma_corrected   # 同一條公式

    probe_stats = {
        "s2_frames":    n,
        "s2_mu_raw":    round(s2_mu,          4),
        "s2_sigma_raw": round(s2_sigma,        4),
        "autocorr":     round(autocorr,        4),
        "mu":           round(mu_corrected,    4),
        "sigma":        round(sigma_corrected, 4),
        "threshold_v1": round(threshold_v1,    4),
        "threshold_v2": round(threshold_v2,    4),
        "r0":           round(r0,              4),
        # lambda_decay 不在此預算，需要 video_duration_s，由 select_keyframes_v4 計算
    }
    return probe_stats


# ══════════════════════════════════════════════════════
# Stage 3-V4 — Temporal Decay Clustering 關鍵幀選取（新增）
# ══════════════════════════════════════════════════════

def select_keyframes_v4(embeddings, timestamps, probe_stats,
                        k=1.0, beta=1.0, video_duration_s=None):
    """
    V4：Temporal Decay Clustering（TDC）。

    核心概念：
      每個 cluster 中心有「時間衰減半徑」r_i(t) = r₀ × exp(-λ × (t - born_at))。
      當一幀和所有現有 cluster 的距離都大於其當前半徑時，判定為新語意區域，
      選為關鍵幀並建立新 cluster。
      同一場景反覆出現時，隨著 cluster 半徑衰減，最終會被重新選取。

    參數設計：
      r₀ = μ + k × σ
        基於統計控制界限（SPC），和 V1 threshold 使用同一數學語言。
        新幀與所有 cluster 中心的距離都超過 μ+kσ，才判定為新語意區域。

      λ = β × (1 - autocorr) / T
        單位：1/秒（量綱正確）。
        T = 影片總長（秒），讓 λ 對應到「整部影片尺度的衰減速率」。
        autocorr 高（重複場景多）→ λ 小 → 衰減慢 → 同場景不易重選。
        autocorr 低（場景不重複）→ λ 大 → 衰減快 → 新場景容易被選。

    超參數：
      k    — 控制 r₀ 寬鬆程度，預設 1.0（和 V1 的 k 意義一致）
      beta — 控制衰減速率縮放，預設 1.0
      video_duration_s — 影片總長（秒），由 get_video_duration() 取得

    回傳：關鍵幀時間戳列表。
    """
    # ── 計算 r₀ ──
    r0 = probe_stats["mu"] + k * probe_stats["sigma"]

    # ── 計算 λ ──
    autocorr = probe_stats["autocorr"]
    if video_duration_s and video_duration_s > 0:
        T = video_duration_s
    else:
        # 若未提供影片長度，用 timestamps 的最大值估計
        T = float(max(timestamps)) if timestamps else 60.0
        print(f"  ⚠ video_duration_s 未提供，用 timestamps 最大值估計 T={T:.1f}s")

    lambda_decay = beta * max(1.0 - autocorr, 0.0) / T

    print(f"  V4 TDC 參數：")
    print(f"    r₀ = μ+kσ = {probe_stats['mu']:.4f}+{k}×{probe_stats['sigma']:.4f} = {r0:.4f}")
    print(f"    λ  = β×(1-autocorr)/T = {beta}×(1-{autocorr:.4f})/{T:.1f} = {lambda_decay:.6f} /s")

    # ── Cluster 狀態列表 ──
    # 每個 cluster = {"center": np.array, "born_at": float（秒）, "n": int（樣本數）}
    clusters     = []
    keyframe_times = []

    for i, t in enumerate(timestamps):
        emb = embeddings[i:i+1]   # shape (1, D)

        if not clusters:
            # 第一幀：直接建立第一個 cluster
            clusters.append({"center": emb.copy(), "born_at": float(t), "n": 1})
            keyframe_times.append(t)
            continue

        # ── 計算每個 cluster 的衰減半徑和距離 ──
        min_dist     = float("inf")
        nearest_idx  = -1

        for ci, cluster in enumerate(clusters):
            age      = float(t) - cluster["born_at"]
            radius   = r0 * np.exp(-lambda_decay * age)
            dist     = 1 - cosine_similarity(emb, cluster["center"])[0][0]

            if dist < min_dist:
                min_dist    = dist
                nearest_idx = ci
                nearest_rad = radius

        # ── 判斷：是否為新語意區域 ──
        # 需要對最近的 cluster 做判斷，若距離 > 其當前半徑 → 新關鍵幀
        nearest_cluster = clusters[nearest_idx]
        age_nearest     = float(t) - nearest_cluster["born_at"]
        radius_nearest  = r0 * np.exp(-lambda_decay * age_nearest)

        if min_dist > radius_nearest:
            # 選為關鍵幀，建立新 cluster
            clusters.append({"center": emb.copy(), "born_at": float(t), "n": 1})
            keyframe_times.append(t)
        else:
            # 不選，做 online mean update（更新最近 cluster 中心）
            c = nearest_cluster
            n = c["n"]
            c["center"] = (c["center"] * n + emb) / (n + 1)
            c["center"] = c["center"] / np.linalg.norm(c["center"])   # re-normalize
            c["n"]      = n + 1

    print(f"  V4 關鍵幀：{len(keyframe_times)} 個（clusters 總數：{len(clusters)}）")
    return keyframe_times


# ══════════════════════════════════════════════════════
# Stage 7 — timing log 輸出（新增）
# ══════════════════════════════════════════════════════

def save_timing_log(label, method, video_id, url, timing_dict):
    """
    將各 stage 的耗時記錄存成 JSON，供後續效能分析使用。

    timing_dict 應包含：
      video_duration_s  — 影片總秒數
      t_download        — 下載耗時
      t_audio_extract   — 音訊抽取耗時
      t_probe           — S2 探針耗時
      t_clip_encode     — CLIP 全片編碼耗時
      t_select          — 關鍵幀選取耗時（含截圖）
      t_whisper         — Whisper 轉錄耗時
      t_panel_build     — 漫畫格組裝耗時
      t_phi3_total      — Phi-3 全部處理耗時
      t_phi3_per_frame_avg — Phi-3 平均每幀耗時
      n_keyframes       — 真實關鍵幀數（不含補幀）
      n_padding         — 動態補幀數
      n_panels          — 總漫畫格數
      keyframe_ratio    — n_panels / video_duration_s（每秒幀數，衡量蒸餾密度）
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log = {
        "label":    label,
        "method":   method,
        "video_id": video_id if video_id else label,
        "url":      url if url else "",
        **{k: round(v, 3) if isinstance(v, float) else v
           for k, v in timing_dict.items()}
    }

    out_path = f"{RESULTS_DIR}/timing_{method}_{label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"  Timing log 存至：{out_path}")
    return out_path


def get_video_duration(video_path):
    """取得影片總秒數。"""
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return round(n_frames / fps, 1)
    return 0.0
