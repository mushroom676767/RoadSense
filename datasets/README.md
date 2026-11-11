# 数据集下载

数据集作为 [GitHub Release 附件](https://github.com/mushroom676767/RoadSense/releases/tag/datasets-v1) 发布，版本为 `datasets-v1`。压缩包不进入 Git 提交历史。

三个原始压缩包合计 **32,906,449,741 字节（32.91 GB / 30.65 GiB）**。GitHub 单个附件必须小于 2 GiB，因此两个大文件按每卷最多 1,900,000,000 字节进行二进制分卷；还原后与原文件逐字节一致。

## 文件清单

| 原始压缩包 | 字节数 | Release 附件数 | 内容 |
| --- | --- | --- | --- |
| `CRPD.zip` | 19,416,634,858 | 11 | 单车牌、双层车牌及多车牌图片与文本标注 |
| `plate_pose.7z` | 13,128,068,242 | 7 | 整帧车牌 Pose 数据及数据集配置 |
| `plate_pose_roi.7z` | 361,746,641 | 1 | ROI 车牌 Pose 数据及配置 |

`CRPD.zip` 内有 33,757 张 JPG 和对应的 33,757 份 TXT，分布在 `CRPD_single`、`CRPD_double`、`CRPD_multi` 中。

`plate_pose.7z` 中 `images/train` 为 21,011 张，`images/val` 为 4,690 张，另有 4,690 张按尺寸组织的验证图片。`plate_pose_roi.7z` 中训练图片为 25,688 张，验证图片为 5,568 张。完整压缩包还包含配置、统计或缓存等辅助文件；文件名和内容按原包保留。

## 自动下载与还原

需要 Python 3.11 或更新版本，不需要第三方 Python 依赖。可以克隆仓库后在根目录执行以下命令；也可以仅从 Release 下载 `download.py` 与 `manifest.json` 并放在同一目录，再运行 `python download.py`。

在仓库根目录执行：

```bash
# 下载全部三个数据集，验证分卷后还原原始压缩包
python datasets/download.py

# 只下载较小的 ROI 数据集
python datasets/download.py --dataset plate_pose_roi.7z

# 指定有足够空间的保存目录
python datasets/download.py --output /path/to/dataset-storage
```

默认输出目录为 `datasets/downloads/`，已被 Git 忽略。脚本支持已完成文件复用及下载中断后的 Range 续传；每个附件和最终压缩包均校验大小与 SHA-256。还原成功后默认删除分卷文件，仅保留原始压缩包；使用 `--keep-parts` 可以保留分卷。

全部下载并还原时，建议预留至少 **50 GB 空闲空间**，解压后的数据需要额外空间。中断后重新执行相同命令即可。校验失败的下载保存在 `.failed` 文件中，脚本不会将其当作有效压缩包。

仅校验现有压缩包：

```bash
python datasets/download.py --verify-only
```

## 手动下载

在 Release 页面下载所需附件，保持文件名不变：

- `CRPD.zip.part001` ～ `CRPD.zip.part011`：按编号拼接为 `CRPD.zip`。
- `plate_pose.7z.part001` ～ `plate_pose.7z.part007`：按编号拼接为 `plate_pose.7z`。
- `plate_pose_roi.7z`：直接使用，不需要拼接。

分卷是原压缩包的连续字节片段，需先还原完整文件，再使用 ZIP/7-Zip 解压工具打开；它们不是 7-Zip 的 `.001` 分卷格式。

在 Windows 的命令提示符（cmd.exe）中，可在附件所在目录执行：

```bat
copy /b CRPD.zip.part001+CRPD.zip.part002+CRPD.zip.part003+CRPD.zip.part004+CRPD.zip.part005+CRPD.zip.part006+CRPD.zip.part007+CRPD.zip.part008+CRPD.zip.part009+CRPD.zip.part010+CRPD.zip.part011 CRPD.zip
copy /b plate_pose.7z.part001+plate_pose.7z.part002+plate_pose.7z.part003+plate_pose.7z.part004+plate_pose.7z.part005+plate_pose.7z.part006+plate_pose.7z.part007 plate_pose.7z
```

在 Linux/macOS 中：

```bash
cat CRPD.zip.part??? > CRPD.zip
cat plate_pose.7z.part??? > plate_pose.7z
sha256sum CRPD.zip plate_pose.7z plate_pose_roi.7z
```

## 校验清单

[`SHA256SUMS.txt`](SHA256SUMS.txt) 提供全部附件的校验值；[`manifest.json`](manifest.json) 记录每个分卷的顺序、字节偏移、大小、SHA-256 和下载地址。原始完整文件的 SHA-256：

```text
09a79865d5dc9a24d2e1e729d15c88cee6688268edbe7e49f646211c535e8fb3  CRPD.zip
9c1415ac1a450451d4ce8375c354f4a2408df85ab5e7e7857e03eb352611524f  plate_pose.7z
c0d38330dd7e7129e2a6a320ec733ad8c66cff2c19132d13ddef5a3041080c9f  plate_pose_roi.7z
```

## 使用说明

解压后请检查各数据集的 `data.yaml`，按自己的训练环境调整路径。此处发布数据包，不附带训练脚本或模型权重。

源码仓库的 MIT 许可证适用于项目自有源码，不重新授权数据集。当前压缩包未附独立数据许可证，来源与使用条件应结合原始数据集的授权说明确认。

## 下载工具测试

在仓库根目录执行：

```bash
python -m unittest discover -s datasets/tests -v
```

测试使用本地临时数据检查分卷还原、断点续传、坏文件拒绝、已有文件复用和输出路径校验，不会下载真实数据集。
