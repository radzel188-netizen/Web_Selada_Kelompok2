"""
=============================================================================
SISTEM CERDAS SELADA KERITING - DATASET COLLECTOR (ESP32-CAM)
Kelompok 2 - Pengumpul Dataset Citra Tanaman untuk Pelatihan Model CNN
-----------------------------------------------------------------------------
Fungsi:
1. Menghubungkan laptop ke ESP32-CAM secara lokal (melalui IP WiFi / Hotspot).
2. Mengambil gambar (snapshot) berkala setiap N menit (misal tiap 10 menit)
   atau secara manual dengan menekan tombol Enter/Spasi.
3. Menyimpan file gambar (.jpg) otomatis dengan timestamp ke folder:
   - dataset/raw/         (Semua foto otomatis)
   - dataset/siap_panen/   (Foto tanaman kondisi panen)
   - dataset/belum_siap/   (Foto tanaman masih vegetatif/muda)
=============================================================================
"""

import os
import sys
import time
import argparse
import datetime
from io import BytesIO
import requests
from PIL import Image

# Pastikan console Windows mendukung UTF-8
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

DEFAULT_CAM_IP = "192.168.4.2"  # IP default ESP32-CAM hotspot portal atau ganti dengan IP router
BASE_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

def buat_folder_dataset():
    """Membuat direktori penyimpanan dataset terstruktur."""
    folders = [
        os.path.join(BASE_DATASET_DIR, "raw"),
        os.path.join(BASE_DATASET_DIR, "siap_panen"),
        os.path.join(BASE_DATASET_DIR, "belum_siap")
    ]
    for f in folders:
        os.makedirs(f, exist_ok=True)
    return folders

def ambil_foto_esp32_cam(cam_ip):
    """Mengambil 1 frame foto dari endpoint /capture ESP32-CAM."""
    url = f"http://{cam_ip}/capture"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200 and len(res.content) > 1000:
            img = Image.open(BytesIO(res.content))
            return img, res.content
        else:
            print(f"[COLLECTOR] ⚠️ Respon kamera tidak valid (Status: {res.status_code}, Bytes: {len(res.content)})")
            return None, None
    except Exception as err:
        print(f"[COLLECTOR] ❌ Gagal terhubung ke ESP32-CAM ({url}): {err}")
        return None, None

def simpan_foto(img_bytes, kategori="raw"):
    """Menyimpan file foto ke folder dataset dengan penamaan standar."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nama_file = f"selada_{kategori}_{timestamp}.jpg"
    folder_tujuan = os.path.join(BASE_DATASET_DIR, kategori)
    path_lengkap = os.path.join(folder_tujuan, nama_file)

    with open(path_lengkap, "wb") as f:
        f.write(img_bytes)

    print(f"[COLLECTOR] 💾 Berhasil menyimpan: {os.path.relpath(path_lengkap)}")
    return path_lengkap

def mode_otomatis(cam_ip, interval_menit):
    """Menjalankan pengumpulan dataset berkala otomatis."""
    print("\n" + "="*65)
    print(f"📸 MODE OTOMATIS: Pengumpul Dataset Aktif (Tiap {interval_menit} Menit)")
    print(f"🎯 Target ESP32-CAM : http://{cam_ip}/capture")
    print(f"📁 Lokasi Folder    : {BASE_DATASET_DIR}")
    print("="*65)
    print("Tekan Ctrl+C untuk menghentikan pengumpulan data.\n")

    counter = 1
    try:
        while True:
            waktu = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{waktu}] [#{counter}] Mengambil foto dari ESP32-CAM...")
            img, bytes_data = ambil_foto_esp32_cam(cam_ip)

            if bytes_data:
                simpan_foto(bytes_data, "raw")
                counter += 1
            else:
                print(f"[{waktu}] Lewati siklus ini karena kamera belum merespons.")

            print(f"Menunggu {interval_menit} menit untuk foto berikutnya...\n")
            time.sleep(interval_menit * 60)
    except KeyboardInterrupt:
        print("\n[COLLECTOR] Pengumpulan otomatis dihentikan.")

def mode_interaktif(cam_ip):
    """Mode manual: tekan tombol untuk menjepret dan langsung melabeli foto."""
    print("\n" + "="*65)
    print("📸 MODE INTERAKTIF / MANUAL LABELING:")
    print("Tekan [1] + Enter: Jepret & Simpan sebagai 'SIAP PANEN'")
    print("Tekan [2] + Enter: Jepret & Simpan sebagai 'BELUM SIAP'")
    print("Tekan [3] + Enter: Jepret & Simpan ke folder 'RAW' (Umum)")
    print("Tekan [Q] + Enter: Keluar dari program")
    print("="*65 + "\n")

    while True:
        pilihan = input("👉 Pilih aksi [1/2/3/Q]: ").strip().upper()
        if pilihan == 'Q':
            break
        elif pilihan in ['1', '2', '3']:
            kategori_map = {'1': 'siap_panen', '2': 'belum_siap', '3': 'raw'}
            kategori = kategori_map[pilihan]
            print(f"Menjepret dari http://{cam_ip}/capture ...")
            img, bytes_data = ambil_foto_esp32_cam(cam_ip)
            if bytes_data:
                simpan_foto(bytes_data, kategori)
            else:
                print("Gagal mengambil foto. Periksa koneksi WiFi dan IP kamera.")
        else:
            print("Pilihan tidak dikenal. Ketik 1, 2, 3, atau Q.")

def main():
    parser = argparse.ArgumentParser(description="Pengumpul Dataset Citra Selada dari ESP32-CAM")
    parser.add_argument("--ip", type=str, default=DEFAULT_CAM_IP, help="Alamat IP ESP32-CAM (Cth: 192.168.0.131)")
    parser.add_argument("--interval", type=int, default=10, help="Interval potret otomatis dalam menit (Default: 10 menit)")
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto", help="Mode operasi: 'auto' (tiap N menit) atau 'manual' (jepret interaktif)")

    args = parser.parse_args()
    buat_folder_dataset()

    if args.mode == "manual":
        mode_interaktif(args.ip)
    else:
        mode_otomatis(args.ip, args.interval)

if __name__ == "__main__":
    main()
