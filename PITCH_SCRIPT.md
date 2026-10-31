# Skrip pitch — PitchIQ — Expected Goals Intelligence (2–3 menit)

Video screen-record aplikasi, bahasa lisan Indonesia campur istilah teknis, ~350 kata.

---

**(0:00–0:20) Masalah — halaman Shot Map**

"Di lembar skor, semua tembakan dicatat sama: 'satu shot'. Padahal
tembakan dari 40 yard bisa punya peluang yang sangat berbeda dengan tap-in
satu sentuhan. PitchIQ mengubah angka mentah itu menjadi
keputusan yang bisa dilakukan pelatih."

**(0:20–0:50) Data — halaman Model Evaluation**

"Kami memakai 12 ribu tembakan nyata dari StatsBomb Open Data: Premier League
2015/16, La Liga 2019/20, dan Piala Dunia 2022 — total 479 pertandingan. Setiap
tembakan punya posisi, bagian tubuh, dan konteks tekanan. Pada mayoritas shot,
freeze-frame juga mencatat posisi bek dan kiper saat tembakan dilakukan. Yang
tidak kami pakai sebagai input adalah informasi yang muncul setelah tembakan:
hasil akhir, end location, atau angka xG bawaan dari StatsBomb. Split kami
dikelompokkan per pertandingan supaya tidak ada kebocoran antara training dan
testing."

**(0:50–1:25) Model — kurva kalibrasi**

"Model deployment kami adalah regresi logistik yang transparan. Dashboard
membandingkan ROC-AUC, log loss, Brier score, dan kalibrasi dengan eksperimen
model pohon serta benchmark xG StatsBomb. Kami tidak mengklaim mengalahkan
benchmark. Nilai PitchIQ adalah proses yang bisa diperiksa: data sebelum shot,
fitur yang jelas, dan analisis yang membantu menentukan pertanyaan lanjutan
untuk pelatih. Angka di demo ini berasal dari laporan post-audit yang
dihasilkan ulang (Okt 2026), bukan dari arsip lama."

**(1:25–1:45) Demo inti — halaman xG Estimator**

"Lihat Estimator ini: geser posisi tembakan dari tengah kotak ke arah sayap.
Probabilitas gol turun secara real-time, dan segitiga sudut gawang ikut
menyempit. Panel 'Why this xG?' lalu menjelaskan kontribusi tiap faktor
satu per satu — jarak, sudut, bagian tubuh — jadi angkanya tidak kotak-katik."

**(1:45–2:10) Innovation — halaman Match Timeline + replay**

"Ini fitur favorit kami: 479 pertandingan bisa diputar ulang seperti siaran
bola. Kurva xG kumulatif kedua tim dibangun menit demi menit, benchmark
StatsBomb diganjal sebagai garis putus-putus, dan mode replay memainkan
seluruh pertandingan otomatis dengan feed kejadian langsung."

**(2:10–2:30) Gamification — halaman Beat the Model**

"Pengunjung juga bisa menantang modelnya: lima tembakan dari holdout yang
tidak pernah dilatih, kamu tebak peluangnya, lalu log loss menentukan siapa
yang lebih akurat — kamu atau model. Setiap angka di sini berasal dari
split test yang aman, bukan data latih."

**(2:30–2:50) Insight — halaman Team & Player Insights**

"Insight Engine memeriksa bagian tembakan bernilai rendah, play pattern, dan
zona paling produktif — pakai head-to-head radar antar tim dan scout card
yang bisa diunduh. Skenario penggantian peluang rendah-xG tetap kami labeli
sebagai asumsi, bukan jaminan tambahan gol. Angka spesifik diambil dari
laporan post-audit yang baru dihasilkan ulang, bukan arsip lama."

**(2:50–3:00) Penutup**

"PitchIQ — xG transparan, evaluasi yang jujur, dan rekomendasi berbasis angka.
Semua itu dibangun dari open data StatsBomb dengan kredit penuh, dan kodenya
kami rilis dengan lisensi MIT. Terima kasih!"

---

## Angka utama (regenerated post-audit, 29 Okt 2026 — sinkron dengan app)

- LogReg test holdout (seed 20261029, 96 match / 2.459 shot non-penalti):
  AUC **0.796**, log loss **0.257**, Brier **0.0732**; benchmark StatsBomb
  0.803 / 0.255 / 0.0724 — tidak diklaim menang.
- Ablation: geometri 0.756 → +konteks shot 0.781 → +freeze-frame bek/kiper
  **0.796** (+0.040 dibanding geometri saja).
- Aston Villa: **54%** tembakan < 0.05 xG; skenario penggantian sepertiga
  dengan peluang ~0.10 xG ≈ **+5.2 xG** (asumsi eksplisit, bukan causal
  forecast).
- Stoke City: 59 tembakan (14%) dari 30+ unit pitch: xG/shot 0.017, 0 gol.
- Counter-attack: mean xG **0.146 vs 0.097** regular play (558 shot) —
  asosiasi deskriptif, bukan efek kausal tempo.
- Zona terbaik (110+ / dalam channel, band lateral di-mirror): mean xG
  **0.178**, 27.5% dari semua tembakan.
- Messi (Barcelona, La Liga): 27 gol vs 19.9 xG (**+7.06**, 190 tembakan);
  Özil (Arsenal): 142 key-pass-linked shots (14.5 xG).


## Wajib sebelum merekam demo / submit

- Jalankan pipeline lengkap hingga semua tes selesai; jika pengunduhan data gagal, selesaikan kegagalan tersebut dan jangan gunakan cache parsial.
- Buka app setelah pipeline selesai; pastikan warning artefak pra-audit sudah hilang.
- Perbarui angka di pitch dari laporan baru, isi URL live demo di README/Devpost, dan pastikan repository GitHub publik.
