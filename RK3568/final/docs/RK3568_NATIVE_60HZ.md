# RK3568 Native 1080p60 Pipeline

## Production path

```text
UVC MJPEG
  -> one copy into a persistent MPP packet buffer
  -> MPP NV12 frame pool (4 DMA-BUFs, token + generation + refcount)
     -> DRM Atomic NV12 plane (plane 103)
     -> RGA letterbox directly into RKNN input DMA-BUF
     -> RGA aligned small BGR ROI for plate recognition

C/FreeType premultiplied ARGB double buffer
  -> DRM Atomic ARGB plane (plane 57)
```

The display queue and detector queue are both latest-wins. Page flips never
wait for detection or recognition. HDMI运行固定使用NV12 DMA-BUF与DRM Atomic，
不生成全帧BGR，也不提供软件显示回退。

The UVC driver supports `VIDIOC_EXPBUF`, but direct compressed DMA-BUF import
into this MPP/JPEGD combination produced decoder IRQ failures and reduced
steady FPS to about 58.4. It is therefore production-disabled. The tested
one-copy packet pool is the default. `RKHDMI_ENABLE_EXPBUF=1` is only an
experimental diagnostic switch.

## Build and run

```bash
cd /home/cat/RK3568/final
make -C core/native all

python3 fpga_ai.py \
  --draw-mode fast \
  --result-log-interval 100000
```

Atomic初始化诊断发现NV12/ARGB Plane不可用时，程序会明确报错并退出。

`--result-log-interval` only controls terminal result messages. It does not
change C-side overlays or video timing. `--font-path` can override the default
font. This board resolves:

```text
/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

## Reproducible verification

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile fpga_ai.py core/*.py plate_det/*.py
make -B -C core/native all

# RKNN DMA-BUF input replay must report replay_max_abs_diff=0,0,0
python3 tools/compare_rknn_zero_copy.py

# 20 complete resource acquire/release cycles
python3 tools/stress_start_stop.py --iterations 20

# 10-second warm-up and 60-second measurement
python3 benchmark_pipeline.py \
  --scenario no-plate --warmup 10 --duration 60 \
  --output /tmp/rk3568-native-no-plate-60s.json \
  -- --debug --result-log-interval 100000

# Force a UTF-8 Chinese full annotation update on every video frame
python3 tools/overlay_stress.py \
  --warmup 10 --duration 60 \
  --output /tmp/rk3568-overlay-fast-60s.json

# Probe only; does not enable the experimental path
cc -O2 -Wall -o /tmp/probe_v4l2_expbuf tools/probe_v4l2_expbuf.c
/tmp/probe_v4l2_expbuf /dev/video9
```

Runtime CPU/NPU inspection:

```bash
pidstat -p "$(pgrep -n -f fpga_ai.py)" 1
mpstat -P ALL 1
watch -n 1 cat /sys/kernel/debug/rknpu/load
```

## Measured results

Board: RK3568, Linux 4.19, 1920x1080 MJPEG 60 Hz. Results below use a
10-second warm-up followed by 60 seconds of measurement.

| Metric | Baseline | Native result |
|---|---:|---:|
| Capture FPS, no plate | 58-59 | 59.385 average, 59.2 minimum |
| Page Flip FPS, no plate | 58-59 | 59.378 average, 59.2 minimum |
| V4L2/MPP/DRM drops | 0 | 0 |
| Detector | 30-50 ms | 15.0 ms average |
| Full-frame RGA | about 8 ms | 0 ms |
| Total processing | 43-85 ms | 19.325 ms average, 19.7 ms P95 |
| Process CPU | about 104% | 43.29% average |
| NPU load | about 30% | 38.46% average |

Forced per-frame C overlay stress:

| Metric | Result |
|---|---:|
| Page Flip FPS | 59.380 |
| C overlay average | 1.395 ms |
| C overlay P99 | 1.906 ms |
| Atomic commit errors | 0 |
| Font / planes | Noto Sans CJK / NV12 103 / ARGB 57 |

The 20-cycle test passed 20/20, averaged 10.03 seconds per full start/stop,
and reported a controller FD delta of 0. The startup retry also handled UVC
USB re-enumeration without changing the video backend.

## Acceptance status and remaining physical test

The no-plate 60-second run, forced Chinese overlay run, 20-cycle resource
test, RKNN raw-output replay, builds, and unit tests pass.

Two planned migration items are intentionally not presented as accepted:

- RKNN execution and input memory are native C, but YOLO/NMS and recognition
  softmax/CTC result decoding still run in Python. They need a fixed real-plate
  regression corpus before removal of the compatibility implementation.
- The board lacks Fontconfig development metadata, so the native overlay uses
  deterministic Noto/WQY/Droid font-path fallback plus `--font-path`, rather
  than linking Fontconfig.

A real plate was not continuously present during the final 60-second window.
Consequently the following must still be repeated with a physical plate in
view before creating the release tag:

- 60-second `--scenario plate` run;
- recognition result and color correctness;
- result-age P95 under 80 ms;
- rapid entry/exit, two plates, edge plates, and new-energy Chinese plates.

Do not create `rk3568-native-60hz-v1` until those physical tests pass. The
command after successful validation is:

```bash
cd /home/cat/RK3568
git tag -a rk3568-native-60hz-v1 -m "RK3568 native 1080p60 accepted"
```

## Git rollback

Each stage is independently revertible. For example:

```bash
cd /home/cat/RK3568
git revert 8245c8a  # aligned small ROI
git revert 53c6791  # EXPBUF probe/fallback
git revert a8fe7e3  # overlay P99 acceptance tool
git revert dc897ca  # RKNN input DMA-BUF
git revert 9307a7e  # one-copy compressed packet pool
```

Do not use a hard reset. Generated `.so` files, `/tmp` JSON reports, logs, and
temporary images are intentionally excluded from Git.
