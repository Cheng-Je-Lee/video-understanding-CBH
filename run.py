"""
run.py — Comic Book Hypothesis 單影片執行入口
==============================================
用法：
    python run.py <YouTube_URL> [options]

範例：
    python run.py https://www.youtube.com/watch?v=7Hk9jct2ozY
    python run.py https://www.youtube.com/watch?v=7Hk9jct2ozY --label my_video --methods v4
    python run.py https://www.youtube.com/watch?v=7Hk9jct2ozY --skip-phi3

流程：
    Stage 2   下載影片 + 抽取音訊
    Stage 3a  S2 冷啟動探針（前 100 幀）
    Stage 3b  計算探針統計量（μ / σ / autocorr）
    Stage 3c  全片 CLIP 編碼
    Stage 3d  V1 / V2 / V4 關鍵幀選取 + 截圖
    Stage 4   Whisper 轉錄 + 語言偵測
    Stage 5   漫畫格組裝 + Dynamic Sampling Padding
    Stage 6   Phi-3 Vision 逐格理解 → 記憶文字檔
    Stage 7   輸出摘要報告

輸出（預設在 /content/ 下，可透過 pipeline_core.py 的路徑常數修改）：
    /content/memories/<method>_<label>.txt
    /content/memories/<method>_<label>.json
    /content/video_profiles/<label>.json
    /content/results/<label>_summary.json
"""

import argparse
import json
import os
import sys
import time

# ── 解析參數 ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="Comic Book Hypothesis Pipeline")
parser.add_argument("url",
                    help="YouTube 影片網址")
parser.add_argument("--label",      default=None,
                    help="影片識別名稱（預設從 URL 自動產生）")
parser.add_argument("--video-id",   default=None,
                    help="Video-MME 資料集編號（例如 391）")
parser.add_argument("--methods",    default="v1,v2,v3,v4",
                    help="要跑的方法，逗號分隔（預設：v1,v2,v3,v4）")
parser.add_argument("--skip-phi3",  action="store_true",
                    help="跳過 Stage 6 Phi-3 處理（加快測試）")
parser.add_argument("--gemini-key", default=None,
                    help="Gemini API key，提供後自動產生長期記憶 JSON")
parser.add_argument("--k",          type=float, default=0.3,
                    help="V1 / V4 threshold 的 k 值（預設 0.3）")
parser.add_argument("--beta",       type=float, default=1.0,
                    help="V4 λ 的 β 值（預設 1.0）")
args = parser.parse_args()

# ── 影片識別名稱 ──────────────────────────────────────
url      = args.url
methods  = [m.strip().lower() for m in args.methods.split(",")]
video_id = args.video_id
K        = args.k
BETA     = args.beta
FILL_INTERVAL = 30

if args.label:
    label = args.label
else:
    vid = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]
    label = f"video_{vid}"

print("=" * 60)
print("Comic Book Hypothesis Pipeline")
print(f"  URL      : {url}")
print(f"  Label    : {label}")
print(f"  Video-ID : {video_id if video_id else '(未指定)'}")
print(f"  Methods  : {methods}")
print(f"  k        : {K}  beta : {BETA}")
print(f"  Phi-3    : {'跳過' if args.skip_phi3 else '批次'}")
print("=" * 60)

# ── import pipeline_core ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_core import (
    # 路徑常數
    KF_DIR_V1, KF_DIR_V2, MEMORIES_DIR, PROFILES_DIR, RESULTS_DIR,
    # Stage 2
    download_video, extract_audio,
    # Stage 3
    encode_video_probe, encode_video_full,
    compute_s2_probe,
    select_keyframes_v1, select_keyframes_v2, select_keyframes_v4,
    save_screenshots,
    # Stage 4
    transcribe, build_video_profile,
    # Stage 5
    build_panels, build_v3_panels,
    apply_temporal_padding, write_index,
    # Stage 6
    get_phi3, run_phi3_batch,
    save_memory_txt, save_memory_json,
    build_long_term_memory,
    # 工具
    save_timing_log, get_video_duration,
)

import os

# V4 目錄（pipeline_core 未定義，這裡補上）
KF_DIR_V4 = "/content/keyframes_v4"
COMICS_V1 = "/content/comics_v1"
COMICS_V2 = "/content/comics_v2"
COMICS_V3 = "/content/comics_v3"
COMICS_V4 = "/content/comics_v4"
for d in [KF_DIR_V4, COMICS_V1, COMICS_V2, COMICS_V3, COMICS_V4]:
    os.makedirs(d, exist_ok=True)

pipeline_start = time.time()
timing         = {}
all_memories   = {}
all_timing     = {}

# ══════════════════════════════════════════════════════
# Stage 2 — 下載影片 + 音訊
# ══════════════════════════════════════════════════════
print("\n[Stage 2] 下載影片 & 音訊")
t0 = time.time()
video_path = download_video(label, url)
timing["t_download"] = round(time.time() - t0, 2)

t0 = time.time()
audio_path = extract_audio(label, video_path)
timing["t_audio_extract"] = round(time.time() - t0, 2)

video_duration_s = get_video_duration(video_path)
timing["video_duration_s"] = video_duration_s
print(f"  影片長度    : {video_duration_s} 秒")

# ══════════════════════════════════════════════════════
# Stage 3a — S2 冷啟動探針
# ══════════════════════════════════════════════════════
print("\n[Stage 3a] S2 冷啟動探針（前 100 幀）")
t0 = time.time()
probe_emb, probe_ts = encode_video_probe(video_path, label)
timing["t_probe_clip"] = round(time.time() - t0, 2)
print(f"  探針 CLIP 耗時 : {timing['t_probe_clip']} 秒（{len(probe_emb)} 幀）")

# ── Stage 3b — 探針統計量 ──
t0 = time.time()
probe_stats = compute_s2_probe(probe_emb, probe_ts, k=K)
timing["t_probe_stats"] = round(time.time() - t0, 2)
print(f"  μ={probe_stats['mu']:.4f}  σ={probe_stats['sigma']:.4f}  autocorr={probe_stats['autocorr']:.4f}")
print(f"  V1/V2 threshold = {probe_stats['threshold_v1']:.4f}")
print(f"  V4  r₀          = {probe_stats['r0']:.4f}")

# ══════════════════════════════════════════════════════
# Stage 3c — 全片 CLIP 編碼
# ══════════════════════════════════════════════════════
print("\n[Stage 3c] 全片 CLIP 編碼")
t0 = time.time()
embeddings, timestamps = encode_video_full(video_path, label)
timing["t_full_clip"] = round(time.time() - t0, 2)
print(f"  全片 CLIP 耗時 : {timing['t_full_clip']} 秒（{len(embeddings)} 幀）")

# ══════════════════════════════════════════════════════
# Stage 3d — 關鍵幀選取 + 截圖
# ══════════════════════════════════════════════════════
print("\n[Stage 3d] 關鍵幀選取 + 截圖")
keyframe_times = {}

if "v1" in methods or "v2" in methods:
    kf_v1 = select_keyframes_v1(embeddings, timestamps,
                                 threshold=probe_stats["threshold_v1"])
    keyframe_times["v1"] = kf_v1
    save_screenshots(video_path, label, kf_v1, KF_DIR_V1)
    print(f"  V1：{len(kf_v1)} 個關鍵幀")

if "v2" in methods:
    kf_v2 = select_keyframes_v2(embeddings, timestamps,
                                 threshold=probe_stats["threshold_v1"])
    keyframe_times["v2"] = kf_v2
    save_screenshots(video_path, label, kf_v2, KF_DIR_V2)
    print(f"  V2：{len(kf_v2)} 個關鍵幀")

if "v4" in methods:
    kf_v4 = select_keyframes_v4(embeddings, timestamps,
                                  probe_stats=probe_stats,
                                  k=K, beta=BETA,
                                  video_duration_s=video_duration_s)
    keyframe_times["v4"] = kf_v4
    save_screenshots(video_path, label, kf_v4, KF_DIR_V4)
    print(f"  V4：{len(kf_v4)} 個關鍵幀")

# ══════════════════════════════════════════════════════
# Stage 4 — Whisper 轉錄
# ══════════════════════════════════════════════════════
print("\n[Stage 4] Whisper 轉錄 + Video Profile")
t0 = time.time()
segments, lang, conf, audio_type = transcribe(label, audio_path)
timing["t_whisper"] = round(time.time() - t0, 2)
profile = build_video_profile(label, lang, conf, audio_type,
                               video_id=video_id, url=url)
print(f"  語言：{lang}（信心 {conf:.2f}）  audio_type：{audio_type}")
print(f"  Whisper 耗時：{timing['t_whisper']} 秒")

if audio_type == "silent":
    print("  ⚠ 靜音影片：逐字稿為空，V3 將無內容")

# ══════════════════════════════════════════════════════
# Stage 5 — 漫畫格組裝
# ══════════════════════════════════════════════════════
print("\n[Stage 5] 漫畫格組裝")
panels_by_method = {}
kf_dirs = {"v1": KF_DIR_V1, "v2": KF_DIR_V2, "v3": None, "v4": KF_DIR_V4}
comics_dirs = {"v1": COMICS_V1, "v2": COMICS_V2, "v3": COMICS_V3, "v4": COMICS_V4}

for method in methods:
    if method == "v3":
        p = build_v3_panels(segments)
        write_index(p, f"{COMICS_V3}/{label}", label, "v3",
                    has_image=False, video_id=video_id, url=url)
        panels_by_method["v3"] = p
        print(f"  V3：{len(p)} 個 10 秒段落（純逐字稿）")
    elif method in keyframe_times:
        kf_dir = kf_dirs[method]
        comics_dir = comics_dirs[method]
        p = build_panels(label, keyframe_times[method], segments)
        p = apply_temporal_padding(p, fill_interval=FILL_INTERVAL,
                                   video_path=video_path,
                                   label=label, kf_dir=kf_dir)
        write_index(p, f"{comics_dir}/{label}", label, method,
                    video_id=video_id, url=url)
        panels_by_method[method] = p
        real    = sum(1 for x in p if not x.get("is_fill"))
        dynamic = sum(1 for x in p if x.get("is_dynamic"))
        print(f"  {method.upper()}：{real} 個關鍵幀 + {dynamic} 個動態補幀（共 {len(p)} 個）")

# ══════════════════════════════════════════════════════
# Stage 6 — Phi-3 Vision
# ══════════════════════════════════════════════════════
if args.skip_phi3:
    print("\n[Stage 6] 已跳過 Phi-3（--skip-phi3）")
else:
    print("\n[Stage 6] Phi-3 Vision 逐格理解")
    for method in methods:
        if method not in panels_by_method or not panels_by_method[method]:
            print(f"  [{method}] 無 panels，跳過")
            continue
        print(f"\n  ── Method {method.upper()} ──")
        t0 = time.time()
        memories, timing_log = run_phi3_batch(
            label, method, panels_by_method[method], kf_dirs[method]
        )
        timing[f"t_phi3_{method}"] = round(time.time() - t0, 2)
        all_memories[method] = memories
        all_timing[method]   = timing_log

        save_memory_txt(label, method, memories, url=url, video_id=video_id)
        save_memory_json(label, method, memories, url=url,
                         video_id=video_id, kf_dir=kf_dirs[method])
        print(f"  [{method}] {len(memories)} 條記憶，耗時 {timing[f't_phi3_{method}']} 秒")

        if args.gemini_key:
            build_long_term_memory(
                label, method,
                panels_by_method[method],
                {"video_id": video_id, "url": url},
                gemini_api_key=args.gemini_key
            )

# ══════════════════════════════════════════════════════
# Stage 7 — 輸出摘要
# ══════════════════════════════════════════════════════
print("\n[Stage 7] 輸出摘要")
elapsed = time.time() - pipeline_start

summary = {
    "label":       label,
    "video_id":    video_id,
    "url":         url,
    "methods":     methods,
    "profile":     profile,
    "phi3_ran":    not args.skip_phi3,
    "elapsed_sec": round(elapsed, 1),
    "probe_stats": {
        "mu":       round(probe_stats["mu"], 4),
        "sigma":    round(probe_stats["sigma"], 4),
        "autocorr": round(probe_stats["autocorr"], 4),
    },
    "keyframe_counts": {
        method: len(kf)
        for method, kf in keyframe_times.items()
    },
    "memory_counts": {
        method: len(mems)
        for method, mems in all_memories.items()
    },
    "timing": timing,
    "notes": [],
}

if audio_type == "silent":
    summary["notes"].append("silent video: no whisper segments, V3 empty")

summary_path = f"{RESULTS_DIR}/{label}_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 60}")
print(f"Pipeline 完成！總耗時：{elapsed:.0f} 秒")
print(f"  Video profile  : /content/video_profiles/{label}.json")
for method in methods:
    if method in all_memories:
        print(f"  Memory [{method.upper()}] txt  : /content/memories/{method}_{label}.txt")
        print(f"  Memory [{method.upper()}] json : /content/memories/{method}_{label}.json")
if args.gemini_key:
    print(f"  Long-term      : /content/long_term/{video_id if video_id else label}.json")
print(f"  Summary        : {summary_path}")
print(f"\n  language={lang} (conf={conf:.2f})  audio_type={audio_type}")
print(f"  S2 probe: μ={probe_stats['mu']:.4f} σ={probe_stats['sigma']:.4f}")
print("=" * 60)
