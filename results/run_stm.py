"""
run_stm.py — CBH Short-term Memory (ASI Backward Tracing)
==========================================================
OVO-Bench ASI 短期記憶評測，本地執行版本。

用法：
    python run_stm.py --youtube-id fFjv93ACGo8 \
                      --video-list video_list.json \
                      --api-key YOUR_ANTHROPIC_API_KEY

    # 或透過環境變數傳入 API key（推薦）：
    export ANTHROPIC_API_KEY=YOUR_KEY
    python run_stm.py --youtube-id fFjv93ACGo8 --video-list video_list.json

輸出：
    ./ovo_stm_frames/<youtube_id>/item<id>/   — 選取的幀圖片 + 波形圖
    ./ovo_stm_results/<youtube_id>.json       — 答題結果與來源貢獻度分析

依賴套件：
    pip install yt-dlp scipy ultralytics anthropic spacy openai-whisper matplotlib
    pip install git+https://github.com/openai/CLIP.git
    python -m spacy download en_core_web_sm
    apt-get install -y ffmpeg   # Linux
    # brew install ffmpeg       # macOS
"""

import argparse
import base64
import json
import os
import re
import subprocess
import time
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
from PIL import Image
from scipy.stats import norm as scipy_norm
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import spacy
import whisper as whisper_lib
import anthropic

# ── 解析參數 ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="CBH Short-term Memory Eval")
parser.add_argument("--youtube-id",       required=True,
                    help="YouTube video ID（例如 fFjv93ACGo8）")
parser.add_argument("--video-list",       default="video_list.json",
                    help="OVO-Bench 題目 JSON 路徑（預設：video_list.json）")
parser.add_argument("--api-key",          default=None,
                    help="Anthropic API key（也可設環境變數 ANTHROPIC_API_KEY）")
parser.add_argument("--out-dir",          default="./ovo_stm_frames")
parser.add_argument("--result-dir",       default="./ovo_stm_results")
parser.add_argument("--video-dir",        default="./ovo_videos")
# 大窗口參數
parser.add_argument("--big-window-sigma", type=float, default=15.0)
parser.add_argument("--big-window-n",     type=int,   default=20)
parser.add_argument("--min-interval-big", type=float, default=0.5)
# 小窗口參數
parser.add_argument("--kf-window",        type=float, default=3.0)
parser.add_argument("--rt-window",        type=float, default=5.0)
parser.add_argument("--sigma-kf",         type=float, default=1.0)
parser.add_argument("--sigma-rt",         type=float, default=1.5)
parser.add_argument("--clip-top-k",       type=int,   default=3)
parser.add_argument("--min-interval-small",type=float,default=0.25)
parser.add_argument("--max-n-small",      type=int,   default=5)
# 動態大窗口方向
parser.add_argument("--main-side-n",      type=int,   default=5)
parser.add_argument("--other-side-n",     type=int,   default=1)
# 其他
parser.add_argument("--dedup-thresh",     type=float, default=0.95)
parser.add_argument("--alpha",            type=float, default=1.0)
parser.add_argument("--beta",             type=float, default=1.0)
args = parser.parse_args()

YOUTUBE_ID        = args.youtube_id
OUT_DIR           = args.out_dir
RESULT_DIR        = args.result_dir
VIDEO_DIR         = args.video_dir
BIG_WINDOW_SIGMA  = args.big_window_sigma
BIG_WINDOW_N      = args.big_window_n
MIN_INTERVAL_BIG  = args.min_interval_big
KF_WINDOW         = args.kf_window
RT_WINDOW         = args.rt_window
SIGMA_KF          = args.sigma_kf
SIGMA_RT          = args.sigma_rt
CLIP_TOP_K        = args.clip_top_k
MIN_INTERVAL_SMALL= args.min_interval_small
MAX_N_SMALL       = args.max_n_small
MAIN_SIDE_N       = args.main_side_n
OTHER_SIDE_N      = args.other_side_n
CLIP_DEDUP_THRESH = args.dedup_thresh
ALPHA             = args.alpha
BETA              = args.beta

ANTHROPIC_API_KEY = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("請提供 Anthropic API key（--api-key 或環境變數 ANTHROPIC_API_KEY）")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

print("=" * 60)
print(f"CBH Short-term Memory Eval (ASI)")
print(f"  YouTube ID      : {YOUTUBE_ID}")
print(f"  Video list      : {args.video_list}")
print(f"  Big window σ    : {BIG_WINDOW_SIGMA}  N: {BIG_WINDOW_N}")
print(f"  Main/Other side : {MAIN_SIDE_N} / {OTHER_SIDE_N}")
print("=" * 60)

# ══════════════════════════════════════════════════════
# 載入題目
# ══════════════════════════════════════════════════════
with open(args.video_list) as f:
    video_list = json.load(f)

all_items = [q for q in video_list if q["video_id"] == YOUTUBE_ID]
assert len(all_items) > 0, f"找不到 video_id={YOUTUBE_ID}"

print(f"\n✅ 找到 {len(all_items)} 道題目")
for i, item in enumerate(all_items):
    print(f"  題 {i+1:02d}  ID={item['id']}  ANS={item['gt']}")
    print(f"        Q: {item['question']}")

# ══════════════════════════════════════════════════════
# 下載影片
# ══════════════════════════════════════════════════════
RAW_PATH = f"{VIDEO_DIR}/{YOUTUBE_ID}_raw.mp4"
if not os.path.exists(RAW_PATH):
    print(f"\n下載影片中...")
    subprocess.run([
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        f"https://www.youtube.com/watch?v={YOUTUBE_ID}",
        "-o", RAW_PATH
    ], check=True)
    print(f"✅ 下載完成：{RAW_PATH}")
else:
    print(f"\n✅ 已有本地影片：{RAW_PATH}")

# ══════════════════════════════════════════════════════
# 載入模型
# ══════════════════════════════════════════════════════
print("\n載入模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  裝置：{device}")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("yolov8n.pt")
nlp = spacy.load("en_core_web_sm")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
print("✅ CLIP + YOLO + spaCy 載入完成")

# ══════════════════════════════════════════════════════
# CLIP 全片編碼 + V4 TDC（只跑一次）
# ══════════════════════════════════════════════════════
print("\n[CLIP 全片編碼]")
cap = cv2.VideoCapture(RAW_PATH)
fps_vid   = cap.get(cv2.CAP_PROP_FPS)
total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps_vid)
print(f"  影片長度：{total_sec}s  fps={fps_vid:.1f}")

embeddings, timestamps = [], []
for t in range(total_sec):
    cap.set(cv2.CAP_PROP_POS_FRAMES, t * fps_vid)
    ret, frame = cap.read()
    if not ret: continue
    img = clip_preprocess(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img).cpu().numpy()
    embeddings.append(emb[0])
    timestamps.append(t)
cap.release()
embeddings = np.array(embeddings)
print(f"✅ CLIP 編碼完成：{len(timestamps)} 幀")

# S2 探針 + OLS 校正（論文公式）
s2_mask = [i for i, t in enumerate(timestamps) if t <= 20 or (60 <= t <= 100)]
if len(s2_mask) < 2:
    s2_mask = list(range(len(timestamps)))
s2_embs  = embeddings[s2_mask]
s2_diffs = [1 - cosine_similarity(s2_embs[j:j+1], s2_embs[j+1:j+2])[0][0]
            for j in range(len(s2_embs)-1)]
s2_mu       = float(np.mean(s2_diffs))
s2_sigma    = float(np.std(s2_diffs))
s2_autocorr = float(np.corrcoef(
    s2_embs[:-1].mean(axis=1), s2_embs[1:].mean(axis=1)
)[0,1]) if len(s2_embs) > 2 else 0.0

# OLS 校正（論文公式）
full_mu    = 0.8076 * s2_mu    + 0.1839
full_sigma = max(0.7533 * s2_sigma + 0.0178, 0.001)

# V4 參數（k=0.3，論文公式）
r0  = full_mu + 0.3 * full_sigma
lam = BETA * (1 - s2_autocorr) / max(total_sec, 1)
print(f"S2: μ={s2_mu:.4f} σ={s2_sigma:.4f} autocorr={s2_autocorr:.4f}")
print(f"V4: r₀={r0:.4f} λ={lam:.6f}")

# V4 TDC 全片關鍵幀
clusters, kf_all = [], []
for emb, t in zip(embeddings, timestamps):
    if not clusters:
        clusters.append((emb, t)); kf_all.append(t); continue
    eff_dists = [
        (1 - cosine_similarity(emb.reshape(1,-1), c.reshape(1,-1))[0][0])
        - r0 * np.exp(-lam * (t - born))
        for c, born in clusters
    ]
    if min(eff_dists) > 0:
        clusters.append((emb, t)); kf_all.append(t)
    else:
        idx_n = int(np.argmin([
            1 - cosine_similarity(emb.reshape(1,-1), c.reshape(1,-1))[0][0]
            for c, _ in clusters
        ]))
        old_c, old_b = clusters[idx_n]
        new_c = (old_c + emb) / 2
        new_c = new_c / np.linalg.norm(new_c)
        clusters[idx_n] = (new_c, old_b)
print(f"✅ V4 TDC：{len(kf_all)} 個全片關鍵幀")

# Whisper 逐字稿（只跑一次）
print("\n[Whisper 轉錄]")
whisper_model   = whisper_lib.load_model("base")
whisper_result  = whisper_model.transcribe(RAW_PATH, language="en")
whisper_segments = whisper_result["segments"]
print(f"✅ Whisper 完成：{len(whisper_segments)} 段逐字稿")

# ══════════════════════════════════════════════════════
# 工具函數
# ══════════════════════════════════════════════════════
def extract_keywords(question):
    doc = nlp(question)
    nouns     = [t.text for t in doc if t.pos_ in ("NOUN","PROPN")]
    verbs     = [t.text for t in doc if t.pos_ == "VERB"]
    dirs      = [t.text for t in doc if t.text.lower() in
                 ("left","right","front","behind","above","below","forward","backward")]
    is_after  = any(t.text.lower() == "after"  for t in doc)
    is_before = any(t.text.lower() == "before" for t in doc)
    return nouns, verbs, dirs, is_after, is_before

def gaussian_sample_times(center, window, sigma, min_interval,
                           t_min=0, t_max=None, max_n=5):
    lo = max(t_min, center - window)
    hi = min(center + window, t_max) if t_max is not None else center + window
    candidates = np.arange(lo, hi + 0.05, 0.25)
    if len(candidates) == 0: return []
    densities = scipy_norm.pdf(candidates, loc=center, scale=sigma)
    densities = densities / densities.max()
    selected = []
    for idx in np.argsort(-densities):
        t = candidates[idx]
        if all(abs(t - s) >= min_interval for s in selected):
            selected.append(round(float(t), 2))
        if len(selected) >= max_n: break
    return sorted(selected)

def sparse_gaussian_sample(center, t_min, t_max, sigma, n_max, min_interval):
    candidates = np.arange(t_min, t_max + 0.1, 0.5)
    if len(candidates) == 0: return []
    densities = scipy_norm.pdf(candidates, loc=center, scale=sigma)
    densities = densities / densities.max()
    selected = []
    for idx in np.argsort(-densities):
        t = candidates[idx]
        if all(abs(t - s) >= min_interval for s in selected):
            selected.append(round(float(t), 1))
        if len(selected) >= n_max: break
    return sorted(selected)

def batch_save_frames(video_path, time_path_dict):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    saved = {}
    for t in sorted(time_path_dict.keys()):
        out_path = time_path_dict[t]
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(out_path, frame)
            saved[t] = out_path
    cap.release()
    return saved

def add_frame(frame_dict, t, path, source):
    t = round(float(t), 2)
    if t not in frame_dict:
        frame_dict[t] = {"time": t, "path": path, "sources": [source]}
    else:
        if source not in frame_dict[t]["sources"]:
            frame_dict[t]["sources"].append(source)

def find_window_boundary(T_event, direction, n, t_hard_limit):
    if direction == "after":
        kf_cand = sorted([t for t in kf_all if t > T_event])
        kf_bd   = kf_cand[n-1] if len(kf_cand) >= n else (kf_cand[-1] if kf_cand else T_event)
        kf_bd   = min(kf_bd, t_hard_limit)
        sem_mask = [i for i, t in enumerate(timestamps) if T_event < t <= t_hard_limit]
        if sem_mask:
            w_ts  = [timestamps[i] for i in sem_mask]
            sims  = cosine_similarity(q_emb, embeddings[sem_mask])[0]
            top_n = np.argsort(-sims)[:n]
            sem_bd = max(w_ts[i] for i in top_n)
        else:
            sem_bd = T_event
        return min(max(kf_bd, sem_bd), t_hard_limit)
    else:
        kf_cand = sorted([t for t in kf_all if t < T_event], reverse=True)
        kf_bd   = kf_cand[n-1] if len(kf_cand) >= n else (kf_cand[-1] if kf_cand else T_event)
        kf_bd   = max(kf_bd, t_hard_limit)
        sem_mask = [i for i, t in enumerate(timestamps) if t_hard_limit <= t < T_event]
        if sem_mask:
            w_ts  = [timestamps[i] for i in sem_mask]
            sims  = cosine_similarity(q_emb, embeddings[sem_mask])[0]
            top_n = np.argsort(-sims)[:n]
            sem_bd = min(w_ts[i] for i in top_n)
        else:
            sem_bd = T_event
        return max(min(kf_bd, sem_bd), t_hard_limit)

def plot_sampling_wave(item_id, T_event, realtime, win_min, win_max,
                        kf_selected, clip_centers, big_times, item_dir):
    t_axis = np.linspace(win_min, win_max, 1000)
    fig, ax = plt.subplots(figsize=(12, 4))
    d_D = scipy_norm.pdf(t_axis, loc=T_event, scale=BIG_WINDOW_SIGMA)
    d_D = d_D / d_D.max() * 0.4
    ax.fill_between(t_axis, d_D, alpha=0.3, color="#888888", label="D 大窗口")
    d_A = np.zeros_like(t_axis)
    for kf in kf_selected:
        d_A += scipy_norm.pdf(t_axis, loc=kf, scale=SIGMA_KF)
    if d_A.max() > 0: d_A = d_A / d_A.max()
    ax.plot(t_axis, d_A, color="#1A5EA8", linewidth=1.5, label="A 突變點窗口")
    d_C = np.zeros_like(t_axis)
    for tc in clip_centers:
        d_C += scipy_norm.pdf(t_axis, loc=tc, scale=SIGMA_KF)
    if d_C.max() > 0: d_C = d_C / d_C.max()
    ax.plot(t_axis, d_C, color="#E07B2A", linewidth=1.5, label="C 語意窗口")
    d_B = scipy_norm.pdf(t_axis, loc=realtime, scale=SIGMA_RT)
    if d_B.max() > 0: d_B = d_B / d_B.max() * 0.8
    ax.plot(t_axis, d_B, color="#6B9FD4", linewidth=1.5, label="B realtime窗口")
    d_total = d_D + d_A + d_C + d_B
    if d_total.max() > 0: d_total = d_total / d_total.max()
    ax.plot(t_axis, d_total, color="black", linewidth=2.0,
            linestyle="--", label="Total", zorder=5)
    ax.axvline(T_event,  color="red",   linewidth=1.5, label=f"T_event={T_event:.0f}s")
    ax.axvline(realtime, color="green", linewidth=1.5, label=f"realtime={realtime:.0f}s")
    ax.axvspan(win_min, win_max, alpha=0.05, color="gray")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized sampling density")
    ax.set_title(f"Sampling distribution — Item {item_id}", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=3, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    out_path = f"{item_dir}/sampling_wave.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return out_path

def encode_image(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def build_prompt(question, options, records, nouns, verbs, transcript_panels):
    n = len(records)
    image_list = "\n".join(
        f"  Image {i+1}: {r['time']:.1f}s  [{'+'.join(r['sources'])}]"
        for i, r in enumerate(records)
    )
    scores_fmt = "\n".join(f'    "Image {i+1}": 0.0' for i in range(n))
    opts = options + ["N/A"] * (4 - len(options))
    kw = ""
    if nouns: kw += f"   - Objects/Actions: {', '.join(nouns + verbs)}\n"
    transcript_str = ""
    for p in transcript_panels:
        if p["text"]:
            transcript_str += f"  [{p['start']:.1f}s ~ {p['end']:.1f}s]: {p['text']}\n"
    if not transcript_str: transcript_str = "  (no dialogue)\n"
    return f'''You are analyzing frames and transcript from a video to answer a multiple-choice question.
Frames span a wide temporal window around the key event.
  Images labeled [E1 YOLO_Tevent] and [E2 YOLO_RT] show detected objects at the event moment and at the question timestamp.

Images provided (chronological):
{image_list}

Transcript panels (speech between key frames):
{transcript_str}
Before answering:
1. OBSERVE: Look across ALL frames and read the transcript to understand scene progression.
2. REASON: Focus on these key elements:
{kw}   Trace these elements across frames and transcript.
3. CONCLUDE: Based on the temporal sequence, determine the most likely answer.

For each image, assign a relevance score 0.0-1.0.
Then give your final answer.

Question: {question}
A. {opts[0]}
B. {opts[1]}
C. {opts[2]}
D. {opts[3]}

Respond ONLY in this JSON format, no other text:
{{
  "scores": {{
{scores_fmt}
  }},
  "answer": "A"
}}'''

def ask_claude(res, max_retries=3):
    nouns, verbs, _, _, _ = extract_keywords(res["question"])
    prompt  = build_prompt(res["question"], res["options"],
                            res["frame_records"], nouns, verbs,
                            res["transcript_panels"])
    content = []
    for i, r in enumerate(res["frame_records"]):
        content.append({"type":"image","source":{"type":"base64",
                        "media_type":"image/jpeg","data":encode_image(r["path"])}})
        content.append({"type":"text","text":f"Image {i+1}: {r['time']:.1f}s"})
    content.append({"type":"text","text":prompt})
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5", max_tokens=1000,
                messages=[{"role":"user","content":content}])
            raw = re.sub(r"```json|```","",response.content[0].text.strip()).strip()
            return json.loads(raw)
        except Exception as e:
            print(f"  ⚠️  嘗試 {attempt+1}/{max_retries}：{e}")
            if attempt < max_retries - 1: time.sleep(3)
    return None

# ══════════════════════════════════════════════════════
# 逐題處理
# ══════════════════════════════════════════════════════
all_results = []

for item_idx, item in enumerate(all_items):
    item_id  = item["id"]
    realtime = float(item.get("realtime", -1))
    if realtime < 0: realtime = total_sec - 0.5
    question = item["question"]
    options  = item["options"]
    gt       = item["gt"]

    print(f"\n{'='*60}")
    print(f"題 {item_idx+1}/{len(all_items)}  ID={item_id}  realtime={realtime}s")
    print(f"Q: {question}")

    item_dir = f"{OUT_DIR}/{YOUTUBE_ID}/item{item_id}"
    os.makedirs(item_dir, exist_ok=True)

    # Step 1：spaCy + CLIP 搜尋 T_event
    nouns, verbs, dirs, is_after, is_before = extract_keywords(question)
    query    = " ".join(nouns + verbs)
    q_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        q_emb = clip_model.encode_text(q_tokens).cpu().numpy()
    sims_all = cosine_similarity(q_emb, embeddings)[0]
    T_event  = float(timestamps[int(np.argmax(sims_all))])
    print(f"  T_event={T_event}s  after={is_after}  before={is_before}")

    # Step 2：動態大窗口
    if is_after:
        win_max = find_window_boundary(T_event, "after",  MAIN_SIDE_N,  realtime)
        win_min = find_window_boundary(T_event, "before", OTHER_SIDE_N, 0)
    elif is_before:
        win_min = find_window_boundary(T_event, "before", MAIN_SIDE_N,  0)
        win_max = find_window_boundary(T_event, "after",  OTHER_SIDE_N, realtime)
    else:
        win_min = find_window_boundary(T_event, "before", MAIN_SIDE_N, 0)
        win_max = find_window_boundary(T_event, "after",  MAIN_SIDE_N, realtime)
    win_min = max(0, win_min)
    win_max = min(total_sec - 1, win_max)
    print(f"  大窗口：{win_min:.0f}s ~ {win_max:.0f}s")

    # Step 3：計算所有取樣時間點
    time_sources = defaultdict(list)

    # D 大窗口稀疏高斯
    big_times = sparse_gaussian_sample(T_event, win_min, win_max,
                                        BIG_WINDOW_SIGMA, BIG_WINDOW_N, MIN_INTERVAL_BIG)
    for t in big_times:
        time_sources[t].append((f"{item_dir}/D_t{t:.1f}s.jpg", "D 大窗口"))

    # A V4 突變點
    if is_after:
        kf_main  = sorted([t for t in kf_all if T_event < t <= win_max],
                           key=lambda t: abs(t - T_event))[:MAIN_SIDE_N]
        kf_other = sorted([t for t in kf_all if win_min <= t <= T_event],
                           key=lambda t: abs(t - T_event))[:OTHER_SIDE_N]
    elif is_before:
        kf_main  = sorted([t for t in kf_all if win_min <= t < T_event],
                           key=lambda t: abs(t - T_event))[:MAIN_SIDE_N]
        kf_other = sorted([t for t in kf_all if T_event < t <= win_max],
                           key=lambda t: abs(t - T_event))[:OTHER_SIDE_N]
    else:
        kf_main  = sorted([t for t in kf_all if win_min <= t <= win_max],
                           key=lambda t: abs(t - T_event))[:MAIN_SIDE_N]
        kf_other = []
    kf_selected = kf_main + [t for t in kf_other if t not in kf_main]
    for kf in kf_selected:
        for t in gaussian_sample_times(kf, KF_WINDOW, SIGMA_KF,
                                        MIN_INTERVAL_SMALL, t_min=win_min,
                                        t_max=win_max, max_n=MAX_N_SMALL):
            time_sources[t].append((f"{item_dir}/A_t{t:.2f}s.jpg", "A 突變點窗口"))

    # C CLIP 語意
    def clip_top(t_min, t_max, n):
        mask = [i for i, t in enumerate(timestamps) if t_min <= t <= t_max]
        if not mask: return []
        w_embs = embeddings[mask]
        w_ts   = [timestamps[i] for i in mask]
        sims   = cosine_similarity(q_emb, w_embs)[0]
        top    = np.argsort(-sims)[:n]
        return [(w_ts[i], sims[i]) for i in top]

    if is_after:
        clip_all = clip_top(T_event, win_max, MAIN_SIDE_N) + \
                   [x for x in clip_top(win_min, T_event, OTHER_SIDE_N)]
    elif is_before:
        clip_all = clip_top(win_min, T_event, MAIN_SIDE_N) + \
                   [x for x in clip_top(T_event, win_max, OTHER_SIDE_N)]
    else:
        clip_all = clip_top(win_min, win_max, MAIN_SIDE_N)
    clip_centers = [x[0] for x in clip_all]
    for t_center, sim in clip_all:
        for t in gaussian_sample_times(t_center, KF_WINDOW, SIGMA_KF,
                                        MIN_INTERVAL_SMALL, t_min=win_min,
                                        t_max=win_max, max_n=MAX_N_SMALL):
            time_sources[t].append((f"{item_dir}/C_t{t:.2f}s.jpg", f"C CLIP(sim={sim:.3f})"))

    # B realtime 單側高斯
    for t in gaussian_sample_times(realtime, RT_WINDOW, SIGMA_RT,
                                    MIN_INTERVAL_SMALL, t_min=win_min,
                                    t_max=realtime, max_n=MAX_N_SMALL):
        time_sources[t].append((f"{item_dir}/B_t{t:.2f}s.jpg", "B realtime窗口"))

    # E1/E2 原圖
    time_sources[round(T_event, 1)].append(
        (f"{item_dir}/E1_t{T_event:.1f}s_raw.jpg", "_E1_raw"))
    time_sources[round(realtime, 1)].append(
        (f"{item_dir}/E2_t{realtime:.1f}s_raw.jpg", "_E2_raw"))

    # Step 4：批次截圖
    time_path_dict = {t: entries[0][0] for t, entries in time_sources.items()}
    saved = batch_save_frames(RAW_PATH, time_path_dict)
    print(f"  批次截圖：{len(saved)} 幀")

    # Step 5：建立 frame_dict
    frame_dict = {}
    for t, entries in time_sources.items():
        if t not in saved: continue
        path = saved[t]
        for _, source in entries:
            if not source.startswith("_"):
                add_frame(frame_dict, t, path, source)

    # Step 6：YOLO 標注
    for label, t_yolo, prefix in [("E1 YOLO_Tevent", T_event, "E1"),
                                    ("E2 YOLO_RT", realtime, "E2")]:
        raw_path = saved.get(round(t_yolo, 1))
        if raw_path:
            yolo_out = raw_path.replace("_raw.jpg", "_yolo.jpg")
            try:
                yr = yolo_model(raw_path, conf=0.3, verbose=False)
                cv2.imwrite(yolo_out, yr[0].plot())
                detected = [yr[0].names[int(b.cls)] + f"({float(b.conf):.2f})"
                            for b in yr[0].boxes]
                print(f"  {prefix} YOLO：{detected if detected else '無物件'}")
            except Exception as e:
                print(f"  {prefix} YOLO 失敗：{e}"); yolo_out = raw_path
            offset = 0.01 if prefix == "E1" else 0.02
            frame_dict[round(t_yolo + offset, 2)] = {
                "time": round(t_yolo + offset, 2),
                "path": yolo_out, "sources": [label]
            }

    # Step 7：波形圖
    wave_path = plot_sampling_wave(
        item_id, T_event, realtime, win_min, win_max,
        kf_selected, clip_centers, big_times, item_dir
    )

    frame_records = sorted(frame_dict.values(), key=lambda x: x["time"])
    print(f"  最終幀數：{len(frame_records)} 張")

    # Step 8：逐字稿切段
    cut_points = sorted(set(
        [round(T_event, 1)]
        + [round(t, 1) for t in kf_selected if win_min <= t <= win_max]
        + [round(t, 1) for t in clip_centers if win_min <= t <= win_max]
    ))
    boundaries = [win_min] + cut_points + [win_max]
    transcript_panels = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i+1]
        texts = [s["text"].strip() for s in whisper_segments
                 if s["start"] < seg_end and s["end"] > seg_start]
        transcript_panels.append({
            "start": seg_start, "end": seg_end,
            "text": " ".join(texts) if texts else ""
        })

    all_results.append({
        "item_id": item_id, "realtime": realtime,
        "T_event": T_event, "question": question,
        "options": options, "gt": gt,
        "wave_path": wave_path,
        "frame_records": frame_records, "item_dir": item_dir,
        "transcript_panels": transcript_panels,
        "cut_points": cut_points,
    })

print(f"\n✅ 全部 {len(all_items)} 題處理完成")

# CLIP 去重
for res in all_results:
    records = res["frame_records"]
    keep  = [r for r in records if any(s.startswith("E") for s in r["sources"])]
    dedup = [r for r in records if not any(s.startswith("E") for s in r["sources"])]
    if len(dedup) <= 1:
        res["frame_records"] = dedup + keep; continue
    embs = []
    for r in dedup:
        img = clip_preprocess(Image.open(r["path"])).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(img).cpu().numpy()[0]
        embs.append(emb)
    kept = [0]
    for i in range(1, len(dedup)):
        max_sim = max(cosine_similarity(embs[i:i+1], embs[k:k+1])[0][0] for k in kept)
        if max_sim <= CLIP_DEDUP_THRESH:
            kept.append(i)
    res["frame_records"] = sorted(
        [dedup[k] for k in kept] + keep, key=lambda x: x["time"])
print("✅ 去重完成")

# ══════════════════════════════════════════════════════
# Claude API 答題
# ══════════════════════════════════════════════════════
print("\n[Claude API 答題]")
LLM_RESPONSES = {}
for item_idx, res in enumerate(all_results):
    item_id = res["item_id"]
    print(f"題 {item_idx+1}/{len(all_results)}  ID={item_id}  ({len(res['frame_records'])} 張圖)  ", end="")
    result = ask_claude(res)
    if result:
        LLM_RESPONSES[item_id] = result
        print(f"→ {result['answer']}")
    else:
        print("→ ❌ 失敗")
    time.sleep(1)
print(f"\n✅ 完成：{len(LLM_RESPONSES)}/{len(all_results)} 題")

# ══════════════════════════════════════════════════════
# 計分與輸出
# ══════════════════════════════════════════════════════
def get_category(source):
    if source.startswith("A"):   return "A 突變點窗口"
    elif source.startswith("B"): return "B realtime窗口"
    elif source.startswith("C"): return "C 問題語意"
    elif source.startswith("D"): return "D 大窗口"
    elif source.startswith("E"): return "E YOLO標注"
    else:                        return "F 基準"

cat_scores    = defaultdict(float)
cat_counts    = defaultdict(int)
correct_count = 0
detail_log    = []

for res in all_results:
    item_id = res["item_id"]
    if item_id not in LLM_RESPONSES: continue
    llm        = LLM_RESPONSES[item_id]
    llm_ans    = llm["answer"].strip().upper()
    correct    = res["gt"].strip().upper()
    is_correct = (llm_ans == correct)
    if is_correct: correct_count += 1
    records = res["frame_records"]
    item_detail = {
        "item_id": item_id, "question": res["question"],
        "gt": correct, "llm_answer": llm_ans, "correct": is_correct,
        "T_event": res["T_event"], "frame_contributions": []
    }
    for img_label, score in llm["scores"].items():
        try:
            idx = int(img_label.split()[-1]) - 1
            r   = records[idx]
        except (ValueError, IndexError): continue
        sources       = r["sources"]
        score_per_src = score / len(sources)
        for src in sources:
            cat = get_category(src)
            weighted = 0.0 if cat == "F 基準" else (
                +score_per_src if is_correct else -score_per_src * 0.5
            )
            if cat != "F 基準":
                cat_scores[cat] += weighted
                cat_counts[cat] += 1
            item_detail["frame_contributions"].append({
                "image": img_label, "time": r["time"], "source": src,
                "category": cat, "llm_score": score,
                "score_per_src": score_per_src, "weighted": weighted
            })
    detail_log.append(item_detail)
    status = "✅" if is_correct else "❌"
    print(f"{status} ID={item_id}  LLM={llm_ans}  ANS={correct}  T_event={res['T_event']:.0f}s")

total    = len([r for r in all_results if r["item_id"] in LLM_RESPONSES])
accuracy = correct_count / total if total else 0

print(f"\n{'='*55}")
print(f"正確率：{correct_count}/{total} = {accuracy:.1%}")
print("="*55)
cats = ["A 突變點窗口","B realtime窗口","C 問題語意","D 大窗口","E YOLO標注"]
for cat in cats:
    s = cat_scores[cat]; n = cat_counts[cat]
    avg = s/n if n else 0
    print(f"  {cat:16s}  累積={s:+.3f}  引用={n:3d}  平均={avg:+.4f}")

result_json = {
    "youtube_id": YOUTUBE_ID, "task": "ASI",
    "correct": correct_count, "total": total,
    "accuracy": round(accuracy, 4),
    "params": {
        "BIG_WINDOW_SIGMA": BIG_WINDOW_SIGMA, "BIG_WINDOW_N": BIG_WINDOW_N,
        "KF_WINDOW": KF_WINDOW, "RT_WINDOW": RT_WINDOW,
        "SIGMA_KF": SIGMA_KF, "SIGMA_RT": SIGMA_RT,
        "CLIP_TOP_K": CLIP_TOP_K, "MAIN_SIDE_N": MAIN_SIDE_N,
        "OTHER_SIDE_N": OTHER_SIDE_N, "ALPHA": ALPHA, "BETA": BETA,
    },
    "category_scores": {
        cat: {
            "cumulative": round(cat_scores[cat], 4),
            "count": cat_counts[cat],
            "avg": round(cat_scores[cat]/cat_counts[cat], 4)
                   if cat_counts[cat] else 0
        } for cat in cats
    },
    "detail": detail_log,
}

result_path = f"{RESULT_DIR}/{YOUTUBE_ID}.json"
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=2)
print(f"\n✅ 結果已存：{result_path}")
print(f"✅ 波形圖位置：{OUT_DIR}/{YOUTUBE_ID}/item*/sampling_wave.png")
