# RK3568 上位机发布链路

三个实时入口均支持：

```text
--publish URL
--publish-token TOKEN
--publish-stream STREAM_ID
--publish-fps 30
--publish-quality 50
```

默认不启用推流。Native HDMI模式启用后，`rkhdmi_capture.c`在V4L2采集线程中按目标帧率保存最新原始MJPEG，Publisher线程读取该快照并发送，不执行NV12/BGR/JPEG重编码。

协议：

```text
[4字节大端JSON长度][version=1 JSON UTF-8][JPEG]
```

模式映射：

- `fpga_serial.py` → `plate_serial`
- `fpga_ai.py` → `plate_ai`
- `people_redlight.py` → `people_redlight`

Publisher使用latest-wins、持久HTTP连接和0.5至10秒指数退避。网络错误只更新统计并限频打印，不会设置视频或推理线程的停止事件。

构建与测试：

```bash
cd /home/cat/RK3568
make -C final/core/native librkhdmi_capture.so
cd final
python3 -m unittest tests.test_publisher tests.test_entrypoint_defaults -v
```

`librkhdmi_capture.so`是构建产物，不进入Git。
