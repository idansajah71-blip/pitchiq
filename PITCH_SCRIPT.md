# Skrip pitch — PitchIQ — Expected Goals Intelligence (2–3 menit)

Video screen-record aplikasi, bahasa lisan Indonesia campur istilah teknis, ~350 kata.

---

**(0:00–0:20) Masalah — halaman Shot Map**

"Di lembar skor, semua tembakan dicatat sama: 'satu shot'. Padahal
tembakan dari 40 yard bisa punya peluang yang sangat berbeda dengan tap-in
satu sentuhan. PitchIQ mengubah angka mentah itu menjadi
keputusan yang bisa dilakukan pelatih."

**(0:20–0:55) Data — halaman Evaluasi Model**

"Kami memakai 12 ribu tembakan nyata dari StatsBomb Open Data: Premier League
2015/16, La Liga 2019/20, dan Piala Dunia 2022 — total 479 pertandingan. Setiap
tembakan punya posisi, bagian tubuh, dan konteks tekanan. Pada mayoritas shot,
freeze-frame juga mencatat posisi bek dan kiper saat tembakan dilakukan. Yang
tidak kami pakai sebagai input adalah informasi yang muncul setelah tembakan:
hasil akhir, end location, atau angka xG bawaan dari StatsBomb. Split kami
dikelompokkan per pertandingan supaya tidak ada kebocoran antara training dan
testing."

**(0:55–1:30) Model — kurva kalibrasi**

"Model deployment kami adalah regresi logistik yang transparan. Dashboard
membandingkan ROC-AUC, log loss, Brier score, dan kalibrasi dengan eksperimen
model pohon serta benchmark xG StatsBomb. Kami tidak mengklaim mengalahkan
benchmark. Nilai PitchIQ adalah proses yang bisa diperiksa: data sebelum shot,
fitur yang jelas, dan analisis yang membantu menentukan pertanyaan lanjutan
untuk pelatih. Angka di demo ini berasal dari laporan post-audit yang
dihasilkan ulang (Okt 2026), bukan dari arsip lama."

**(1:30–1:50) Demo inti — halaman xG Estimator**

"Lihat Estimator ini: geser posisi tembakan dari tengah kotak ke arah sayap.
Probabilitas gol turun secara real-time, dan segitiga sudut gawang ikut
menyempit. Semua orang langsung paham alasannya: jarak, sudut, dan bek di
jalur tembakan."

**(1:50–2:35) Insight — halaman Insight Pemain & Tim**

"Insight Engine memeriksa bagian tembakan bernilai rendah, sampel tembakan
jarak jauh, play pattern, dan zona yang paling produktif. Skenario penggantian
peluang rendah-xG adalah kalkulasi berbasis asumsi, bukan jaminan tambahan gol.
Perbedaan xG antar-play-pattern kami sajikan sebagai asosiasi deskriptif, bukan
bukti sebab-akibat. Zona juga dibandingkan dengan memasangkan band lateral yang
simetris di kedua sisi pusat gawang. Angka spesifik di bagian ini diambil dari
laporan yang baru dihasilkan ulang, bukan arsip lama."

**(2:35–2:50) Penutup**

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
