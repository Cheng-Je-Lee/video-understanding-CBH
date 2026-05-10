"""
run_instant.py — CBH Instant Perception (OJR / STU)
=====================================================
OVO-Bench Gaming 即時感知評測，本地執行版本。

用法：
    python run_instant.py --youtube-id lhL3SUb078g \
                          --video-list video_list.json \
                          --api-key YOUR_ANTHROPIC_API_KEY

    # 或透過環境變數傳入 API key（推薦）：
    export ANTHROPIC_API_KEY=YOUR_KEY
    python run_instant.py --youtube-id lhL3SUb078g --video-list video_list.json

輸出：
    ./ovo_instant_frames/<youtube_id>/item<id>/   — 選取的幀圖片
    ./ovo_results/<youtube_id>.json               — 答題結果與貢獻度分析

依賴套件：
    pip install yt-dlp scipy ultralytics anthropic spacy
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
import zipfile
from collections import defaultdict

import cv2
import numpy as np
import torch
import clip
from PIL import Image
from scipy.stats import norm
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import spacy
import anthropic

# ── 解析參數 ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="CBH Instant Perception Eval")
parser.add_argument("--youtube-id",  required=True,
                    help="YouTube video ID（例如 lhL3SUb078g）")
parser.add_argument("--video-list",  default="video_list.json",
                    help="OVO-Bench 題目 JSON 路徑（預設：video_list.json）")
parser.add_argument("--api-key",     default=None,
                    help="Anthropic API key（也可設環境變數 ANTHROPIC_API_KEY）")
parser.add_argument("--out-dir",     default="./ovo_instant_frames",
                    help="幀輸出資料夾（預設：./ovo_instant_frames）")
parser.add_argument("--result-dir",  default="./ovo_results",
                    help="結果輸出資料夾（預設：./ovo_results）")
parser.add_argument("--video-dir",   default="./ovo_videos",
                    help="影片暫存資料夾（預設：./ovo_videos）")
# 幀選取參數
parser.add_argument("--kf-window",   type=float, default=3.0)
parser.add_argument("--rt-window",   type=float, default=5.0)
parser.add_argument("--sigma-kf",    type=float, default=1.0)
parser.add_argument("--sigma-rt",    type=float, default=1.5)
parser.add_argument("--min-interval",type=float, default=0.5)
parser.add_argument("--clip-top-k",  type=int,   default=2)
parser.add_argument("--dedup-thresh",type=float, default=0.95)
parser.add_argument("--alpha",       type=float, default=1.0)
parser.add_argument("--beta",        type=float, default=1.0)
args = parser.parse_args()

YOUTUBE_ID    = args.youtube_id
KF_WINDOW     = args.kf_window
RT_WINDOW     = args.rt_window
SIGMA_KF      = args.sigma_kf
SIGMA_RT      = args.sigma_rt
MIN_INTERVAL  = args.min_interval
CLIP_TOP_K    = args.clip_top_k
DEDUP_THRESH  = args.dedup_thresh
ALPHA         = args.alpha
BETA          = args.beta
OUT_DIR       = args.out_dir
RESULT_DIR    = args.result_dir
VIDEO_DIR     = args.video_dir

# API key
ANTHROPIC_API_KEY = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("請提供 Anthropic API key（--api-key 或環境變數 ANTHROPIC_API_KEY）")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

print("=" * 60)
print(f"CBH Instant Perception Eval")
print(f"  YouTube ID : {YOUTUBE_ID}")
print(f"  Video list : {args.video_list}")
print(f"  KF window  : {KF_WINDOW}s  RT window: {RT_WINDOW}s")
print("=" * 60)

# ══════════════════════════════════════════════════════
# 載入題目
# ══════════════════════════════════════════════════════
with open(args.video_list) as f:
    video_list = json.load(f)

all_items = [q for q in video_list if q['video_id'] == YOUTUBE_ID]
assert len(all_items) > 0, f"找不到 video_id={YOUTUBE_ID}"
all_items = sorted(all_items, key=lambda x: float(x['realtime']))

print(f"\n✅ 找到 {len(all_items)} 道題目")
for i, item in enumerate(all_items):
    print(f"  題 {i+1:02d}  ID={item['id']}  task={item['task']}  realtime={item['realtime']}s")
    print(f"        Q: {item['question']}")

# ══════════════════════════════════════════════════════
# 下載影片
# ══════════════════════════════════════════════════════
RAW_PATH = f"{VIDEO_DIR}/{YOUTUBE_ID}_raw.mp4"
if not os.path.exists(RAW_PATH):
    print(f"\n下載影片中...")
    subprocess.run([
        "yt-dlp", "-f", "mp4", "-o", RAW_PATH,
        f"https://www.youtube.com/watch?v={YOUTUBE_ID}"
    ], check=True)
    print(f"✅ 下載完成")
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
pose_model = YOLO("yolov8n-pose.pt")
nlp = spacy.load("en_core_web_sm")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
print("✅ 模型載入完成")

# ══════════════════════════════════════════════════════
# 工具函數
# ══════════════════════════════════════════════════════
def gaussian_sample_times(center, window, sigma, min_interval, t_min=0, t_max=None):
    lo = max(t_min, center - window)
    hi = min(center + window, t_max) if t_max is not None else center + window
    candidates = np.arange(lo, hi + 0.1, 0.5)
    if len(candidates) == 0:
        return []
    densities = norm.pdf(candidates, loc=center, scale=sigma)
    densities = densities / densities.max()
    selected = []
    for idx in np.argsort(-densities):
        t = candidates[idx]
        if all(abs(t - s) >= min_interval for s in selected):
            selected.append(round(float(t), 1))
    return sorted(selected)

def save_frame(video_path, t_sec, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, t_sec * cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    cv2.imwrite(out_path, frame)
    return out_path

def add_to_frame_dict(frame_dict, t, path, source):
    t = round(float(t), 1)
    if t not in frame_dict:
        frame_dict[t] = {"time": t, "path": path, "sources": [source]}
    else:
        if source not in frame_dict[t]["sources"]:
            frame_dict[t]["sources"].append(source)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def extract_keywords(question):
    doc = nlp(question)
    nouns = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")]
    verbs = [t.text for t in doc if t.pos_ == "VERB"]
    dirs  = [t.text for t in doc if t.text.lower() in
             ("left","right","front","behind","above","below",
              "north","south","east","west","forward","backward")]
    return nouns, verbs, dirs

def build_prompt(question, options, records):
    nouns, verbs, directions = extract_keywords(question)
    kw = ""
    if nouns:      kw += f"   - Objects/Characters: {', '.join(nouns)}\n"
    if verbs:      kw += f"   - Actions: {', '.join(verbs)}\n"
    if directions: kw += f"   - Directions: {', '.join(directions)}\n"
    if not kw:     kw  = "   - (analyse the full scene)\n"

    n = len(records)
    image_list = "\n".join(
        f"  Image {i+1}: {r['time']:.1f}s  [{'+'.join(r['sources'])}]"
        for i, r in enumerate(records)
    )
    scores_fmt = "\n".join(f'    "Image {i+1}": 0.0' for i in range(n))
    opts = options + ["N/A"] * (4 - len(options))

    return f'''You are analyzing frames extracted from a video to answer a multiple-choice question.
The frames are in chronological order. Images labeled [D2 YOLO] show detected objects with bounding boxes. Images labeled [D3 Pose] show character skeleton keypoints.

Images provided:
{image_list}

Before answering, perform the following reasoning steps:
1. OBSERVE: Look across ALL frames for the complete scene context.
2. REASON: Focus on these key elements extracted from the question:
{kw}   Search for these elements in ALL frames.
3. CONCLUDE: Based on your observations, determine the most likely answer.

For each image, assign a relevance score 0.0-1.0 for answering the question.
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

def ask_claude(item_id, question, options, records, max_retries=3):
    prompt = build_prompt(question, options, records)
    content = []
    for i, r in enumerate(records):
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": encode_image(r["path"]),
            }
        })
        content.append({"type": "text", "text": f"Image {i+1}: {r['time']:.1f}s"})
    content.append({"type": "text", "text": prompt})

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1000,
                messages=[{"role": "user", "content": content}]
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            return json.loads(raw)
        except Exception as e:
            print(f"  ⚠️  嘗試 {attempt+1}/{max_retries} 失敗：{e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    return None

# ══════════════════════════════════════════════════════
# 逐題處理
# ══════════════════════════════════════════════════════
all_results = []

for item_idx, item in enumerate(all_items):
    item_id  = item["id"]
    realtime = float(item["realtime"])
    question = item["question"]
    options  = item["options"]
    gt       = item["gt"]

    print(f"\n{'='*55}")
    print(f"題 {item_idx+1}/{len(all_items)}  ID={item_id}  realtime={realtime}s")
    print(f"Q: {question}")

    item_dir = f"{OUT_DIR}/{YOUTUBE_ID}/item{item_id}"
    os.makedirs(item_dir, exist_ok=True)

    # Step 1：ffmpeg 切片
    chunk_path = f"{VIDEO_DIR}/{YOUTUBE_ID}_item{item_id}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", RAW_PATH,
        "-t", str(realtime + 1.0),
        "-c", "copy", chunk_path
    ], check=True, capture_output=True)

    # Step 2：逐秒 CLIP 編碼
    cap = cv2.VideoCapture(chunk_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    embeddings, timestamps = [], []
    for t in range(total_sec):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t * fps)
        ret, frame = cap.read()
        if not ret:
            continue
        img = clip_preprocess(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(img).cpu().numpy()
        embeddings.append(emb[0])
        timestamps.append(t)
    cap.release()
    embeddings = np.array(embeddings)
    print(f"  CLIP 編碼：{len(timestamps)} 幀")

    # Step 3：S2 探針 + OLS 校正 + V4 TDC
    s2_mask = [i for i, t in enumerate(timestamps) if t <= 20 or (60 <= t <= 100)]
    if len(s2_mask) < 2:
        s2_mask = list(range(len(timestamps)))
    s2_embs = embeddings[s2_mask]
    s2_diffs = [
        1 - cosine_similarity(s2_embs[j:j+1], s2_embs[j+1:j+2])[0][0]
        for j in range(len(s2_embs) - 1)
    ]
    s2_mu       = float(np.mean(s2_diffs))
    s2_sigma    = float(np.std(s2_diffs))
    s2_autocorr = float(np.corrcoef(
        s2_embs[:-1].mean(axis=1), s2_embs[1:].mean(axis=1)
    )[0, 1]) if len(s2_embs) > 2 else 0.0

    # OLS 校正（論文公式）
    full_mu    = 0.8076 * s2_mu    + 0.1839
    full_sigma = max(0.7533 * s2_sigma + 0.0178, 0.001)

    # V4 參數（k=0.3，論文公式）
    r0  = full_mu + 0.3 * full_sigma
    lam = BETA * (1 - s2_autocorr) / max(total_sec, 1)
    print(f"  S2: μ={s2_mu:.4f} σ={s2_sigma:.4f} autocorr={s2_autocorr:.4f}")
    print(f"  V4: r₀={r0:.4f} λ={lam:.6f}")

    # V4 TDC
    clusters, kf_times = [], []
    for emb, t in zip(embeddings, timestamps):
        if not clusters:
            clusters.append((emb, t)); kf_times.append(t); continue
        eff_dists = [
            (1 - cosine_similarity(emb.reshape(1,-1), c.reshape(1,-1))[0][0])
            - r0 * np.exp(-lam * (t - born))
            for c, born in clusters
        ]
        if min(eff_dists) > 0:
            clusters.append((emb, t)); kf_times.append(t)
        else:
            idx_n = int(np.argmin([
                1 - cosine_similarity(emb.reshape(1,-1), c.reshape(1,-1))[0][0]
                for c, _ in clusters
            ]))
            old_c, old_b = clusters[idx_n]
            new_c = (old_c + emb) / 2
            new_c = new_c / np.linalg.norm(new_c)
            clusters[idx_n] = (new_c, old_b)
    print(f"  V4 關鍵幀：{len(kf_times)} 個")

    # Step 4：選幀
    kf_before  = [t for t in kf_times if t < realtime]
    nearest_kf = max(kf_before) if kf_before else None
    frame_dict = {}

    # 來源 A：突變點高斯窗口
    if nearest_kf is not None:
        for t in gaussian_sample_times(nearest_kf, KF_WINDOW, SIGMA_KF,
                                        MIN_INTERVAL, t_max=realtime):
            p = save_frame(chunk_path, t, f"{item_dir}/t{t:.1f}s.jpg")
            if p: add_to_frame_dict(frame_dict, t, p, "A 突變點窗口")

    # 來源 B：realtime 高斯窗口
    for t in gaussian_sample_times(realtime, RT_WINDOW, SIGMA_RT,
                                    MIN_INTERVAL, t_max=realtime):
        p = save_frame(chunk_path, t, f"{item_dir}/t{t:.1f}s.jpg")
        if p: add_to_frame_dict(frame_dict, t, p, "B realtime窗口")

    # 來源 C：CLIP 語意 top-K
    q_tokens = clip.tokenize([question]).to(device)
    with torch.no_grad():
        q_emb = clip_model.encode_text(q_tokens).cpu().numpy()
    w_mask = [i for i, t in enumerate(timestamps)
              if (realtime - RT_WINDOW) <= t <= realtime]
    if w_mask:
        w_embs = embeddings[w_mask]
        w_ts   = [timestamps[i] for i in w_mask]
        sims_q = cosine_similarity(q_emb, w_embs)[0]
        for idx in np.argsort(-sims_q)[:CLIP_TOP_K]:
            t, sim = w_ts[idx], sims_q[idx]
            p = save_frame(chunk_path, t, f"{item_dir}/t{t:.1f}s.jpg")
            if p: add_to_frame_dict(frame_dict, t, p, f"C CLIP(sim={sim:.3f})")

    # 來源 D：realtime 三張圖（D1原圖 + D2 YOLO + D3 Pose）
    raw_realtime = f"{item_dir}/t{realtime:.1f}s_raw.jpg"
    p_raw = save_frame(chunk_path, realtime, raw_realtime)

    if p_raw:
        frame_dict[round(realtime, 1)] = {
            "time": round(realtime, 1), "path": p_raw,
            "sources": ["D1 realtime原圖"]
        }

        yolo_out = f"{item_dir}/t{realtime:.1f}s_yolo.jpg"
        try:
            yolo_results = yolo_model(p_raw, conf=0.3, verbose=False)
            cv2.imwrite(yolo_out, yolo_results[0].plot())
            detected = [yolo_results[0].names[int(b.cls)] + f"({float(b.conf):.2f})"
                        for b in yolo_results[0].boxes]
            print(f"  YOLO 偵測：{detected if detected else '無物件'}")
        except Exception as e:
            print(f"  YOLO 失敗：{e}"); yolo_out = p_raw
        frame_dict[round(realtime + 0.01, 2)] = {
            "time": round(realtime + 0.01, 2), "path": yolo_out,
            "sources": ["D2 YOLO物件標注"]
        }

        pose_out = f"{item_dir}/t{realtime:.1f}s_pose.jpg"
        try:
            pose_results = pose_model(p_raw, conf=0.3, verbose=False)
            cv2.imwrite(pose_out, pose_results[0].plot())
            print(f"  Pose 偵測：{len(pose_results[0].boxes)} 個角色")
        except Exception as e:
            print(f"  Pose 失敗：{e}"); pose_out = p_raw
        frame_dict[round(realtime + 0.02, 2)] = {
            "time": round(realtime + 0.02, 2), "path": pose_out,
            "sources": ["D3 Pose姿態標注"]
        }

    frame_records = sorted(frame_dict.values(), key=lambda x: x["time"])
    print(f"  選幀結果：{len(frame_records)} 張")

    all_results.append({
        "item_id": item_id, "realtime": realtime,
        "question": question, "options": options, "gt": gt,
        "frame_records": frame_records, "item_dir": item_dir,
    })

print(f"\n✅ 全部 {len(all_items)} 題處理完成，開始 CLIP 去重...")

# CLIP 去重
for res in all_results:
    records = res["frame_records"]
    if len(records) <= 1:
        continue
    keep_fixed = [r for r in records if any(
        s.startswith("D1") or s.startswith("D2") or s.startswith("D3")
        for s in r["sources"])]
    to_dedup = [r for r in records if not any(
        s.startswith("D1") or s.startswith("D2") or s.startswith("D3")
        for s in r["sources"])]
    if len(to_dedup) <= 1:
        res["frame_records"] = to_dedup + keep_fixed
        continue
    dedup_embs = []
    for r in to_dedup:
        img = clip_preprocess(Image.open(r["path"])).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(img).cpu().numpy()[0]
        dedup_embs.append(emb)
    kept = [0]
    for i in range(1, len(to_dedup)):
        max_sim = max(
            cosine_similarity(dedup_embs[i:i+1], dedup_embs[k:k+1])[0][0]
            for k in kept
        )
        if max_sim <= DEDUP_THRESH:
            kept.append(i)
    deduped = [to_dedup[k] for k in kept]
    res["frame_records"] = sorted(deduped + keep_fixed, key=lambda x: x["time"])

print("✅ 去重完成，開始 Claude API 答題...")

# ══════════════════════════════════════════════════════
# Claude API 答題
# ══════════════════════════════════════════════════════
LLM_RESPONSES = {}

for item_idx, res in enumerate(all_results):
    item_id  = res["item_id"]
    records  = res["frame_records"]
    question = res["question"]
    options  = res["options"]

    print(f"題 {item_idx+1}/{len(all_results)}  ID={item_id}  ({len(records)} 張圖)  ", end="")
    result = ask_claude(item_id, question, options, records)
    if result:
        LLM_RESPONSES[item_id] = result
        print(f"→ {result['answer']}")
    else:
        print("→ ❌ 失敗，跳過")
    time.sleep(1)

print(f"\n✅ 完成：{len(LLM_RESPONSES)}/{len(all_results)} 題")

# ══════════════════════════════════════════════════════
# 計分與輸出
# ══════════════════════════════════════════════════════
def get_category(source):
    if source.startswith("A"):   return "A 突變點窗口"
    elif source.startswith("B"): return "B realtime窗口"
    elif source.startswith("C"): return "C 問題語意"
    elif source.startswith("D2") or source.startswith("D3"): return "D YOLO+Pose"
    else: return "E realtime基準"

category_scores = defaultdict(float)
category_counts = defaultdict(int)
correct_count   = 0
detail_log      = []

for res in all_results:
    item_id = res["item_id"]
    if item_id not in LLM_RESPONSES:
        continue
    llm        = LLM_RESPONSES[item_id]
    llm_answer = llm["answer"].strip().upper()
    correct    = res["gt"].strip().upper()
    is_correct = (llm_answer == correct)
    if is_correct:
        correct_count += 1

    item_detail = {
        "item_id": item_id, "question": res["question"],
        "gt": correct, "llm_answer": llm_answer,
        "correct": is_correct, "frame_contributions": []
    }

    for img_label, score in llm["scores"].items():
        try:
            idx = int(img_label.split()[-1]) - 1
            r   = res["frame_records"][idx]
        except (ValueError, IndexError):
            continue
        sources       = r["sources"]
        score_per_src = score / len(sources)
        for src in sources:
            cat = get_category(src)
            weighted = 0.0 if cat == "E realtime基準" else (
                +score_per_src if is_correct else -score_per_src * 0.5
            )
            if cat != "E realtime基準":
                category_scores[cat] += weighted
                category_counts[cat] += 1
            item_detail["frame_contributions"].append({
                "image": img_label, "time": r["time"], "source": src,
                "category": cat, "llm_score": score,
                "score_per_src": score_per_src, "weighted": weighted,
            })
    detail_log.append(item_detail)
    status = "✅" if is_correct else "❌"
    print(f"{status} ID={item_id}  LLM={llm_answer}  ANS={correct}")

total    = len([r for r in all_results if r["item_id"] in LLM_RESPONSES])
accuracy = correct_count / total if total else 0

print(f"\n{'='*55}")
print(f"正確率：{correct_count}/{total} = {accuracy:.1%}")
print(f"{'='*55}")
cats = ["A 突變點窗口", "B realtime窗口", "C 問題語意", "D YOLO+Pose"]
for cat in cats:
    s   = category_scores[cat]
    n   = category_counts[cat]
    avg = s/n if n else 0
    print(f"  {cat:16s}  累積={s:+.3f}  引用={n:3d}  平均={avg:+.3f}")

# 存結果 JSON
result_json = {
    "youtube_id": YOUTUBE_ID,
    "correct": correct_count, "total": total,
    "accuracy": round(accuracy, 4),
    "params": {
        "KF_WINDOW": KF_WINDOW, "RT_WINDOW": RT_WINDOW,
        "SIGMA_KF": SIGMA_KF, "SIGMA_RT": SIGMA_RT,
        "CLIP_TOP_K": CLIP_TOP_K, "ALPHA": ALPHA, "BETA": BETA,
    },
    "category_scores": {
        cat: {
            "cumulative": round(category_scores[cat], 4),
            "count": category_counts[cat],
            "avg": round(category_scores[cat]/category_counts[cat], 4)
                   if category_counts[cat] else 0
        } for cat in cats
    },
    "detail": detail_log,
}

result_path = f"{RESULT_DIR}/{YOUTUBE_ID}.json"
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=2)
print(f"\n✅ 結果已存：{result_path}")
