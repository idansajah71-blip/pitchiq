# Skrip pitch — PitchIQ — Expected Goals Intelligence (2–3 menit)

Video screen-record aplikasi, bahasa lisan Indonesia campur istilah teknis, ~350 kata.

---

**(0:00–0:20) Masalah — halaman Shot Map**

"Di lembar skor, semua tembakan dicatat sama: 'satu shot'. Padahal
tembakan dari 40 yard betul-betul berbeda dengan tap-in satu sentuhan — dan
selisihnya puluhan kali lipat. PitchIQ mengubah angka mentah itu menjadi
keputusan yang bisa dilakukan pelatih."

**(0:20–0:55) Data — halaman Evaluasi Model**

"Kami memakai 12 ribu tembakan nyata dari StatsBomb Open Data: Premier League
2015/16, La Liga 2019/20, dan Piala Dunia 2022 — total 479 pertandingan. Setiap
tembakan punya posisi, bagian tubuh, tekanan lawan, dan yang paling penting:
freeze-frame, yaitu posisi bek dan kiper persis sebelum bola ditendang. Yang
tidak kami pakai sebagai input adalah informasi yang muncul setelah tembakan:
hasil akhir, end location, atau angka xG bawaan dari StatsBomb. Split kami
dikelompokkan per pertandingan supaya tidak ada kebocoran antara training dan
testing."

**(0:55–1:30) Model — kurva kalibrasi**

"Model pilihan kami regresi logistik yang transparan: ROC-AUC 0.772, hanya
sedikit di bawah XGBoost yang 0.781 — jadi kemampuan membedakan tembakan bagus
dan buruk kira-kira dua dari tiga kali. Kami memilih versi transparan karena
pelatih perlu tahu alasan di balik angkanya. Yang kami pamerkan bukan akurasi,
melainkan kalibrasi: tembakan yang model nilai 25 persen memang masuk sekitar
seperempat waktu. Kalibrasi yang jujur lebih berharga
daripada angka yang terlihat bagus tapi tidak bisa dipakai."

**(1:30–1:50) Demo inti — halaman xG Estimator**

"Lihat Estimator ini: geser posisi tembakan dari tengah kotak ke arah sayap.
Probabilitas gol turun secara real-time, dan segitiga sudut gawang ikut
menyempit. Semua orang langsung paham alasannya: jarak, sudut, dan bek di
jalur tembakan."

**(1:50–2:35) Insight — halaman Insight Pemain & Tim**

"Temuan yang bisa langsung dipakai. Aston Villa mengambil 55 persen tembakannya
di bawah 0.05 xG; kalau sepertiga tembakan semacam itu diganti peluang
berkualitas sedang, timnya dapat sekitar tambahan 5.7 xG per musim. Stoke City
menembak dari jarak jauh 59 kali — nol gol. Counter-attack menghasilkan 0.170
xG per tembakan, hampir dua kali lipat regular play, sehingga naik tempo adalah
upgrade termurah. Dan koridor tepat di depan gawang, zona dengan xG terbaik
0.184, hanya menerima 14 persen dari semua tembakan."

**(2:35–2:50) Penutup**

"PitchIQ — xG transparan, evaluasi yang jujur, dan rekomendasi berbasis angka.
Semua itu dibangun dari open data StatsBomb dengan kredit penuh, dan kodenya
kami rilis dengan lisensi MIT. Terima kasih!"

---

## Angka utama (sinkron dengan aplikasi & Devpost)

- LogReg test: AUC 0.772, log loss 0.263, Brier 0.0741; XGBoost: 0.781.
- Aston Villa: 55% tembakan < 0.05 xG; estimasi +5.7 xG per musim bila
  sepertiganya diganti peluang medium (~0.10 xG).
- Stoke City: 59 tembakan dari 30+ unit, xG/shot 0.016, 0 gol.
- 'From Counter': mean xG 0.170 (conversion 14.7%) vs regular play 0.104.
- Koridor 110+/y20-40: mean xG 0.184, hanya 14.1% dari semua tembakan.
