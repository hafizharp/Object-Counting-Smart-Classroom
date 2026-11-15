# Tugas-Akhir Maulana Hafizh

# Object Counting Smart Classroom — YOLOv8n vs EfficientDet-D0

Repositori eksperimen skripsi **“Perbandingan Model YOLOv8 dan EfficientDet untuk Object Counting pada Smart Classroom”**.

Empat notebook Google Colab ini melatih dan mengevaluasi dua model pada dataset ruang kelas yang sama, masing-masing dalam dua versi:

- **Baseline** — tanpa optimasi (hyperparameter default/tetap).
- **Optimasi** — ditambah hyperparameter tuning dan LR warmup.

Tujuannya adalah membandingkan performa deteksi dan penghitungan objek (`person` dan `table`) secara adil, sehingga selisih hasil antara versi baseline dan optimasi bisa dijadikan bukti kuantitatif di bab hasil/pembahasan.

---

## Daftar Isi

- [Isi Repositori](#isi-repositori)
- [Hasil Ringkas](#hasil-ringkas)
- [Dataset](#dataset)
- [Cara Menjalankan](#cara-menjalankan)
- [Perbedaan Baseline vs Optimasi](#perbedaan-baseline-vs-optimasi)
- [Metrik yang Diukur](#metrik-yang-diukur)
- [Output yang Dihasilkan](#output-yang-dihasilkan)
- [Catatan Penting & Troubleshooting](#catatan-penting--troubleshooting)
- [Referensi](#referensi)

---

## Isi Repositori

| Notebook | Model | Versi | Library utama |
|---|---|---|---|
| `YOLOv8n_Baseline_.ipynb` | YOLOv8n | Baseline | `ultralytics` |
| `YOLOv8n_Optimasi_.ipynb` | YOLOv8n | Optimasi | `ultralytics` |
| `EfficientDet_D0_Baseline.ipynb` | EfficientDet-D0 (`tf_efficientdet_d0`) | Baseline | `effdet`, `timm`, `pycocotools` |
| `EfficientDet_D0_Optimasi.ipynb` | EfficientDet-D0 (`tf_efficientdet_d0`) | Optimasi | `effdet`, `timm`, `pycocotools` |

Tidak ada `requirements.txt`, package Python, atau test suite. Semua dependensi di-install langsung di dalam notebook, dan seluruh pekerjaan berjalan di Google Colab (mount Google Drive).

> Notebook sudah berisi **output hasil eksekusi** (gambar, tabel, dan JSON). Ukurannya 1,8–3,6 MB per file. Jangan menghapus output jika ingin mempertahankan bukti hasil eksperimen.

---

## Hasil Ringkas

Dievaluasi pada **test set 50 gambar**, environment **NVIDIA A100-SXM4-80GB**, dengan data terekam di output notebook.

| Model | Versi | mAP@50 | mAP@50–95 | MAE `person` | MAE `table` | FPS* | Parameter | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | Baseline | 0,847 | 0,590 | 1,56 | 4,40 | 73,27 | 3,01 M | 8,09 |
| YOLOv8n | Optimasi | 0,851 | 0,583 | 1,22 | 2,82 | 71,23 | 3,01 M | 8,09 |
| EfficientDet-D0 | Baseline | 0,641 | 0,346 | 5,86 | 5,66 | 33,62 | 3,76 M | 6,68 |
| EfficientDet-D0 | Optimasi | 0,684 | 0,376 | 5,82 | 6,06 | 33,14 | 3,76 M | 6,68 |

\* **FPS tidak sepenuhnya apple-to-apple.** YOLO diukur dari pipeline `predict()` penuh pada 30 gambar berbeda (1 warm-up). EfficientDet diukur dari forward pass model pada satu gambar yang sama (10 warm-up, 100 iterasi) tanpa I/O gambar.

**Catatan hasil:**

- mAP YOLO berasal dari `best_model.val()` Ultralytics; mAP EfficientDet dari `pycocotools.COCOeval`.
- Precision/Recall EfficientDet dihitung kustom (rata-rata nilai valid dari array `precision`/`recall` COCOeval), jadi **jangan** dibandingkan langsung dengan Precision/Recall YOLO.
- Optimasi tidak selalu menaikkan semua metrik: mAP@50-95 YOLOv8n sedikit turun (0,590 → 0,583), sedangkan MAE membaik.
- MAE keseluruhan EfficientDet: baseline 5,76 dan optimasi 5,94.
- Waktu training final tercatat: EfficientDet baseline **38,3 menit**, EfficientDet optimasi **52,9 menit** (belum termasuk waktu pencarian hyperparameter).

---

## Dataset

Dataset tidak disimpan di repositori ini, melainkan di Google Drive. Komposisinya:

| Split | Jumlah gambar |
|---|---:|
| train | 1.050 |
| valid | 100 |
| test | 50 |

2 kelas: `person` dan `table`. Gambar berasal dari export **Roboflow** yang sudah di-resize ke **640×640** dan diaugmentasi (flip horizontal, rotasi ±5°, brightness/exposure ±10%).

Ada dua format folder di Drive:

**1. Format YOLO** (dipakai notebook YOLOv8n)

```
Ruang Kelas YOLO/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml          # dibuat/diperbarui otomatis oleh notebook
```

**2. Format COCO** (dipakai notebook EfficientDet)

```
Ruang Kelas COCO/              # dipakai EfficientDet Baseline
├── train/_annotations.coco.json
├── valid/_annotations.coco.json
└── test/_annotations.coco.json

Ruang Kelas/                   # dipakai EfficientDet Optimasi
├── train/_annotations.coco.json
├── valid/_annotations.coco.json
└── test/_annotations.coco.json
```

> Perhatikan: kedua notebook EfficientDet memakai nama root yang **berbeda** (`Ruang Kelas COCO` vs `Ruang Kelas`). Samakan path terlebih dahulu sebelum menjalankan ulang.

Karena dataset sudah diaugmentasi di Roboflow, augmentasi bawaan YOLOv8 **dimatikan** (`DISABLED_AUG`) di kedua notebook YOLO untuk mencegah augmentasi ganda. Ini bukan bagian dari optimasi.

---

## Cara Menjalankan

1. Buka notebook di **Google Colab**.
2. Set runtime ke GPU: **Runtime → Change runtime type → GPU**. Eksperimen ini memakai A100; jika memori GPU lebih kecil, turunkan `batch`/`BATCH_SIZE`.
3. Jalankan sel **mount Google Drive** (`drive.mount('/content/drive')`).
4. Sesuaikan path dataset di sel konfigurasi:
   - YOLO: `DATASET_DIR` (bagian 4).
   - EfficientDet: `DATASET_ROOT` (Step 1).
5. Jalankan **Runtime → Run all**.

**Urutan sel penting (YOLO):** `CONF_THRESHOLD` dan `IOU_THRESHOLD` baru didefinisikan di sel FPS. Jika sel MAE dijalankan lebih dulu, akan muncul `NameError`.

**Untuk eksperimen cepat**, nonaktifkan proses mahal terlebih dahulu:

```python
# YOLOv8n_Optimasi_.ipynb
RUN_HYPERPARAMETER_TUNING = False   # default di notebook: True

# EfficientDet_D0_Optimasi.ipynb
N_SEARCH_TRIALS = 1                 # default: 8
NUM_EPOCHS = 5                      # default: 100
```

Tanpa perubahan itu, `Run all` akan menjalankan tuning 30 iterasi × 20 epoch (YOLO) atau 8 trial × 5 epoch lalu 100 epoch (EfficientDet).

---

## Perbedaan Baseline vs Optimasi

| Aspek | YOLOv8n Baseline | YOLOv8n Optimasi | EfficientDet Baseline | EfficientDet Optimasi |
|---|---|---|---|---|
| Hyperparameter | Default Ultralytics | Hasil `model.tune()` | LR `1e-4`, WD `0` (tetap) | Hasil random search |
| Pencarian HP | — | 30 iterasi × 20 epoch | — | 8 trial dari 27 kombinasi, 5 epoch/trial |
| Optimizer | Default | AdamW + `cos_lr=True` | AdamW | AdamW |
| LR warmup | Default | `warmup_epochs=5.0` | — | `warmup_epochs=3`, start factor `0.1` |
| Early stopping | `patience=50` | `patience=50` | `PATIENCE=50` | `PATIENCE=50` |
| Epoch maksimum | 100 | 100 | 100 | 100 |
| Augmentasi | Dimatikan | Dimatikan | Hanya padding | Hanya padding |

Hyperparameter terpilih:

- **YOLOv8n Optimasi:** `lr0=0.00766`, `momentum=0.919`, `weight_decay=0.00098`; training final memakai `cos_lr=True`.
- **EfficientDet-D0 Optimasi:** `lr=3e-4`, `batch_size=4`, `weight_decay=1e-4`.

---

## Metrik yang Diukur

- **mAP@50** dan **mAP@50–95** — akurasi deteksi pada test set.
- **Precision / Recall** — dihitung berbeda antar framework (lihat catatan hasil).
- **MAE counting** — rata-rata selisih absolut jumlah objek prediksi vs ground truth per gambar, per kelas.
- **FPS** — kecepatan inferensi, diukur setelah warm-up.
- **Jumlah parameter** dan **GFLOPs** — ukuran dan biaya komputasi model.
- **Confusion matrix** — termasuk kategori `background` untuk false positive/false negative pada EfficientDet.

Threshold yang benar-benar dipakai di kode: **conf = 0,3** dan **IoU = 0,5**. Beberapa teks markdown di notebook menyebut nilai default Ultralytics (0,25/0,7), tetapi kode menggunakan 0,3/0,5.

---

## Output yang Dihasilkan

Semua output disimpan ke Google Drive, bukan ke repositori.

**YOLOv8n** — di dalam folder run:

```
runs_yolov8n_baselinev2/   atau   runs_yolov8n_optimaze/
├── results.png
├── confusion_matrix.png
├── counting_results_*_per_image.csv
├── counting_results_*_mae_summary.csv
└── weights/best.pt
```

Model final juga disalin ke `saved_models/`:

- `yolov8n_baseline_best.pt`
- `yolov8n_v2_hptuning_lrwarmup_best.pt`

**EfficientDet** — di dalam `efficientdet_d0_baseline_outputs/` atau `efficientdet_d0_outputsv2/`:

```
results_summary.json        # ringkasan semua metrik
hp_search_results.json      # hasil pencarian hyperparameter
loss_curve.png
confusion_matrix.png / .csv
fps_results.json
counting_metrics.json
model_complexity.json
test_predictions_coco.json
counting_simulation.png
*_best.pth
*_final.pth
```

`results_summary.json` adalah artefak utama untuk membandingkan kedua versi EfficientDet.

---

## Catatan Penting & Troubleshooting

- **`FileNotFoundError`** — nama folder di Google Drive bersifat case-sensitive. Cocokkan `DATASET_DIR`/`DATASET_ROOT` persis dengan struktur Drive.
- **`CUDA out of memory`** — turunkan batch size. EfficientDet: hapus nilai besar dari `BATCH_SIZE_CANDIDATES`. YOLO: turunkan `batch`.
- **Tuning berjalan tanpa disadari** — `RUN_HYPERPARAMETER_TUNING = True` di notebook YOLO optimasi, meskipun komentar di sebelahnya menulis “ubah ke True untuk menjalankan tuning”.
- **Label EfficientDet berbasis 1** (bukan 0). Mapping kelas dicetak otomatis saat dataset dimuat; jangan mengasumsikan urutannya sama dengan YOLO.
- **EfficientDet tidak melakukan resize** — gambar ditempel ke kanvas 640×640 di pojok kiri-atas, sehingga input harus sudah 640×640.
- **Jangan rename file notebook** — nama dengan garis bawah/tanpa garis bawah (`Baseline_`, `Optimasi_`) sudah dipakai konsisten di laporan.
- **File notebook besar** — setiap `.ipynb` adalah satu baris JSON raksasa berisi output base64. Edit lewat JSON/`nbformat`, jangan manual.
- Untuk panduan teknis bagi kontributor/agen, lihat [`AGENTS.md`](AGENTS.md).

---

## Referensi

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [effdet (rwightman)](https://github.com/rwightman/efficientdet-pytorch)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [pycocotools](https://github.com/cocodataset/cocoapi)
- [Roboflow](https://roboflow.com/)

