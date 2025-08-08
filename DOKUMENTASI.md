# Dokumentasi API FaceFusion
Berikut adalah panduan lengkap untuk menggunakan endpoint /swap/ beserta semua parameter dan nilai yang tersedia.

## Endpoint Utama
``POST /swap/``
Endpoint ini adalah inti dari API. Ia menerima file gambar/video dan serangkaian parameter opsional untuk melakukan berbagai jenis manipulasi wajah dan video.

*Body:* ``form-data``

## Parameter Wajib
Parameter ini harus selalu disertakan dalam setiap permintaan.

| Key | Tipe | Deskripsi |
| source_file | File | File gambar yang berisi wajah sumber yang ingin digunakan.
| target_file | File | File video target di mana wajah akan diganti atau dimodifikasi.

## Parameter Opsional

### Kontrol Prosesor Utama

| Key | Tipe | Deskripsi & Nilai yang Mungkin |
| processors | Text | Sangat Penting. String berisi nama-nama prosesor yang ingin diaktifkan, dipisahkan koma. Contoh: face_swapper,face_enhancer,age_modifier

### Pengaturan Face Swapper & Deep Swapper

| Key | Tipe | Deskripsi & Nilai yang Mungkin |
| deep_swapper_model | Text | (Hanya aktif jika deep_swapper ada di processors). Memilih model wajah selebriti. Contoh: druuzil/henry_cavill_448, iperov/emma_watson_224 (Lihat daftar lengkap di FaceFusion). |
| deep_swapper_morph | Angka | (0-100) Mengatur seberapa kuat efek "morphing" ke wajah selebriti. Default: 80. |

### Pengaturan Peningkatan Kualitas (Enhancement)
| Key | Tipe | Deskripsi & Nilai yang Mungkin |
| face_enhancer_model | Text | (Hanya aktif jika face_enhancer ada di processors). Memilih model untuk mempertajam wajah. Pilihan: gfpgan_1.4 (paling umum), codeformer, gpen_bfr_512, dll. |
| face_enhancer_blend | Angka | (0-100) Kekuatan efek peningkat wajah. Default: 80.|
| frame_enhancer_model | Text | (Hanya aktif jika frame_enhancer ada di processors). Memilih model untuk mempertajam seluruh frame. Pilihan: real_esrgan_x4 (umum), real_esrgan_x2, ultra_sharp_x4, dll.|
| frame_enhancer_blend | Angka | (0-100) Kekuatan efek peningkat frame. Default: 80.

### Pengaturan Modifikasi & Restorasi
| Key | Tipe | Deskripsi & Nilai yang Mungkin |
| age_modifier_direction | Angka | (Hanya aktif jika age_modifier ada di processors). Mengubah usia. Range: -100 (lebih muda) hingga 100 (lebih tua). |
| expression_restorer_model | Text | (Hanya aktif jika expression_restorer ada di processors). Model untuk memperbaiki ekspresi. Pilihan: live_portrait. | 
| expression_restorer_factor | Angka | (0-100) Kekuatan efek perbaikan ekspresi. Default: 50. |

### Pengaturan Editor Wajah (Face Editor)
Catatan: Semua parameter di bawah ini hanya aktif jika face_editor ada di dalam processors.

| Key | Tipe | Deskripsi & Nilai yang Mungkin |
| face_editor_mouth_smile | Angka Desimal | Mengatur senyum. Range: -1.0 (cemberut) hingga 1.0 (tersenyum lebar).
| face_editor_eye_open_ratio | Angka Desimal | Mengatur seberapa terbuka mata. Range: -1.0 (tertutup) hingga 1.0 (terbuka lebar).
| face_editor_... | Angka Desimal | Parameter lain seperti eye_gaze_horizontal, head_pitch, dll. umumnya memiliki Range: -1.0 hingga 1.0.

### Pengaturan Audio & Warna

| Key | Tipe | Deskripsi & Nilai yang Mungkin | 
| audio_file | File | (Opsional) File audio (.mp3, .wav) yang akan digunakan untuk lip_syncer. | 
| lip_syncer_model | Text | (Hanya aktif jika lip_syncer ada di processors). Memilih model lip-sync. Pilihan: wav2lip_gan, edtalk. |
| lip_syncer_weight | Angka Desimal | (0.0 - 1.0) Kekuatan efek lip-sync. Default: 1.0. |
| frame_colorizer_model | Text | (Hanya aktif jika frame_colorizer ada di processors). Model untuk mewarnai video hitam-putih. Pilihan: ddcolor, deoldify. | 
| frame_colorizer_blend | Angka | (0-100) Kekuatan efek pewarnaan. Default: 80.

### Pengaturan Debugging

| Key | Tipe | Deskripsi & Nilai yang Mungkin |
| face_debugger_items | Text | (Hanya aktif jika face_debugger ada di processors). Menampilkan informasi debug. Pilihan (dipisah koma): bounding-box, face-landmark-68, face-mask, score. |