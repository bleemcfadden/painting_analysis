# Graded Challenge 7

_Graded Challenge ini dibuat guna mengevaluasi pembelajaran pada Hacktiv8 Data Science Fulltime Program khususnya pada konsep Artificial Neural Network._

_version: GAIAv2.2_

_update date: 20250330_

---

## Objectives

*Graded Challenge 7* ini dibuat dengan tujuan sebagai berikut :

- Mampu memahami konsep Computer Vision/NLP.

- Mampu mempersiapkan data untuk digunakan dalam model Computer Vision/NLP.

- Mampu mengimplementasikan Artificial Neural Network dengan data yang dipilih.

- Mampu menganalisis dan menjelaskan performansi dari arsitektur Artificial Neural Network yang dibuat.

---

## Problem

Student dipersilakan memilih topik antara **Computer Vision atau Natural Language Processing** menggunakan dataset sendiri (scraping oleh student) atau menggunakan open dataset.

---

## Dataset

### Ketentuan Dataset
1. **Konsultasikan terlebih dahulu dataset yang hendak digunakan ke buddy masing-masing student. Jika disetujui, maka silakan dikerjakan. Jika tidak disetujui, maka cari dataset yang lain dan konsultasikan lagi mengenai dataset yang baru ini.**

2. Student tidak boleh menggunakan dataset yang sudah dipakai dalam sesi pembelajaran saat dikelas bersama instruktur. Carilah dataset yang baru untuk tugas Graded Challenge ini.

3. **Student diizikan untuk melakukan scraping dataset**. Sertakan code mengenai scraping ini ke dalam repository.

4. Dataset tidak perlu diupload ke dalam repository. Sertakan link dataset pada file `url.txt`. Jika melakukan scraping, maka upload dataset ke dalam Google Drive dan share link mengenai dataset pada file `url.txt`. Sertakan juga code mengenai proses scraping yang dilakukan ke dalam repository.

5. Ketentuan teknis dataset : 

   1. **Ukuran :** tidak perlu terlalu besar. Maksimal berukuran ratusan MB, jangan sampai GB. Dikhawatirkan nanti proses trainingnya tidak selesai.

   2. **Jumlah data :** wajib gunakan seluruh data. Oleh karena itu, ukuran total dataset perlu diperhatikan. Jangan juga terlalu sedikit agar lebih merepresentasikan problem yang diangkat.

   3. **Jumlah class/label :** usahakan maksimal 5 class/label agar proses analisanya dapat mendalam.

   4. **Bagi yang mengambil Computer Vision, tidak disarankan untuk mengangkat problem mengenai dunia medis.** Hal ini dikarenakan, saat EDA maka penjelasan eksplorasinya akan mengenai karakteristik masing-masing class dari sisi medis. Jika dirasa mampu menguraikan hal ini, silakan saja.

6. **Saat melakukan konsultasi mengenai topik yang akan dikerjakan dengan buddy sertakan :**
   - **URL dataset yang hendak dipakai.**
   - **Problem yang hendak diselesaikan (akan digunakan untuk menyelesaikan kasus apa ketika modelnya selesai dibangun, bukan ke arah akan memprediksi apa)**
   - **Berikan justifikasi bahwa kasus tersebut ada/terjadi dan perlu diselesaikan. Anda dapat memberikan URL suatu sumber yang membahas mengenai hal ini.**
   - **User/pengguna yang akan menggunakan aplikasi.**
      

### Data Sources

Student dapat memilih dataset dari salah satu repository dibawah ini. Popular open data repositories :

- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle datasets](https://www.kaggle.com/datasets)
- [Amazon’s AWS datasets](https://registry.opendata.aws/)

Meta portals :

- [Data Portals](http://dataportals.org/)
- [OpenDataMonitor](https://opendatamonitor.eu/frontend/web/index.php?r=dashboard%2Findex)
- [Quandl](https://www.quandl.com/)
- Sumber lain yang kredibel.

---

## Instructions

*Graded Challenge 7* dikerjakan dalam format ***notebook*** dan ***model deployment*** dengan beberapa *kriteria wajib* di bawah ini:

1. Deep Learning framework yang digunakan adalah *TensorFlow*.

2. Ada penggunaan library visualisasi, seperti *matplotlib*, *seaborn*, atau yang lain.

3. Isi *notebook* harus mengikuti *outline* di bawah ini:
   1. Perkenalan
      > Bab pengenalan harus diisi dengan identitas, gambaran besar dataset yang digunakan, dan *objective* yang ingin dicapai.
   
   2. Import Libraries
      > *Cell* pertama pada *notebook* **harus berisi dan hanya berisi** semua *library* yang digunakan dalam *project*.
   
   3. Data Loading
      > Bagian ini berisi proses penyiapan data sebelum dilakukan eksplorasi data lebih lanjut. Proses Data Loading dapat berupa memberi nama baru untuk setiap kolom, mengecek ukuran dataset, dll.
   
   4. Exploratory Data Analysis (EDA)
      > Bagian ini berisi explorasi data pada dataset diatas dengan menggunakan query, grouping, visualisasi sederhana, dan lain sebagainya.

   5. Feature Engineering
      > Bagian ini berisi proses penyiapan data untuk proses pelatihan model, seperti pembagian data menjadi train-val-test, preprocessing, dan proses-proses lain yang dibutuhkan.   
   
   6. ANN Training (Sequential API/Functional API)
      
      vi.1. Model Definition
      > Bagian ini berisi cell untuk mendefinisikan model. Jelaskan alasan menggunakan suatu algoritma/model, hyperparameter yang dipakai, jenis penggunaan metrics yang dipakai, dan hal lain yang terkait dengan model.

      vi.2. Model Training
      > Cell pada bagian ini hanya berisi code untuk melatih model dan output yang dihasilkan. Lakukan beberapa kali proses training dengan hyperparameter yang berbeda untuk melihat hasil yang didapatkan. Analisis dan narasikan hasil ini pada bagian Model Evaluation.
   
      vi.3. Model Evaluation
      > Pada bagian ini, dilakukan evaluasi model yang harus menunjukkan bagaimana performa model berdasarkan metrics yang dipilih. Hal ini harus dibuktikan dengan visualisasi tren performa dan/atau tingkat kesalahan model. **Lakukan analisis secara detail terkait dengan hasil pada model dan tuliskan hasil analisisnya**.

   7. ANN Improvement (Sequential API/Functional API)
      
      vii.1. Model Definition
      > Bagian ini berisi cell untuk mendefinisikan model. Jelaskan alasan menggunakan suatu algoritma/model, hyperparameter yang dipakai, jenis penggunaan metrics yang dipakai, dan hal lain yang terkait dengan model.

      vii.2. Model Training
      > Cell pada bagian ini hanya berisi code untuk melatih model dan output yang dihasilkan. Lakukan beberapa kali proses training dengan hyperparameter yang berbeda untuk melihat hasil yang didapatkan. Analisis dan narasikan hasil ini pada bagian Model Evaluation.
   
      vii.3. Model Evaluation
      > Pada bagian ini, dilakukan evaluasi model yang harus menunjukkan bagaimana performa model berdasarkan metrics yang dipilih. Hal ini harus dibuktikan dengan visualisasi tren performa dan/atau tingkat kesalahan model. **Lakukan analisis secara detail terkait dengan hasil pada model dan tuliskan hasil analisisnya**.
   
   8. Model Saving
      > Pada bagian ini, dilakukan proses penyimpanan model dan file-file lain yang terkait dengan hasil proses pembuatan model. Pilihlah 1 arsitektur ANN yang terbaik berdasarkan hasil evaluasi sebelumnya. **Nyatakan secara jelas arsitektur mana yang akan dipakai sebagai model terbaik (Sequential API atau Functional API).**
   
   9. Model Inference
      > Model yang sudah dilatih akan dicoba pada data yang bukan termasuk ke dalam train-set, val-set, ataupun test-set. Data ini harus dalam format yang asli, bukan data yang sudah di-scaled. **Model Inference harus berada pada notebook yang berbeda dari notebook yang dipakai untuk pembuatan model.**
   
   10. Pengambilan Kesimpulan
       > Pada bagian terakhir ini, **harus berisi** kesimpulan yang mencerminkan hasil yang didapat dengan *objective* yang sudah ditulis di bagian pengenalan.

---

## Submission

- Simpan assignment pada sesi ini dengan nama `P2G7_<nama-student>.ipynb`, misal `P2G7_raka_ardhi.ipynb`.

- **Anda tidak perlu mengupload dataset dan model ke GitHub**, karena dikhawatirkan ukurannya melebihi batas ukuran maksimum file yang diizinkan oleh GitHub.

- Upload model Anda ke Google Drive pribadi dan set permission ke public.

- Push Assigment yang telah Anda buat ke akun Github Classroom Anda masing-masing.

- Untuk Model Deployment :
  * Buat sebuah folder bernama `deployment` dan masukkan semua file yang berkaitan dengan deployment ke folder ini.
  * Buat sebuah file bernama `url.txt` yang berisi URL dataset, URL model, dan URL deployment.
  * Contoh bentuk isi repository dengan Model Deployment.
    ```
    P2-GC7-Set-1/raka-ardhi
    |
    ├── deployment/
    │   ├── app.py
    │   ├── eda.py
    │   ├── prediction.py
    │   ├── requirements.txt
    ├── P2G7_raka_ardhi.ipynb
    ├── P2G7_raka_ardhi_inference.ipynb
    ├── url.txt
    └── README.md
    ```

---

## Rubrics

### Code Review

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Feature Engineering | Mampu melakukan proses Feature Engineering sebelum melakukan proses modeling. Sertakan narasi mengenai alasan Feature Engineering tersebut dilakukan | 20 pts |
```
Penilaian terhadap code Feature Engineering : 10 points
Penilaian terhadap narasi dan keterhubungannya dengan domain yang dihadapi : 10 points
```

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| ANN Training | Membuat model Artificial Neural Network untuk menyelesaikan masalah Computer Vision atau NLP | 20 pts  |
| ANN Improvement | Membuat improvisasi model Artificial Neural Network | 20 pts  |
| Model Inference | Mencoba model yang telah dibuat dengan data baru | 10 pts |
| Runs Perfectly | Kode berjalan tanpa ada error. Seluruh kode berfungsi dan dibuat dengan benar. | 10 pts |

```
Catatan mengenai ANN Training : 

1. Narasikan mengenai layer yang Anda buat beserta hyperparameter yang Anda tulis.
2. Sebaiknya lakukan training dengan lebih dari 20 epochs agar dapat terlihat performa model dengan lebih jelas.
3. Anda dipersilakan membuat arsitektur model dengan Sequential API/Functional API/keduanya.
4. Pada bagian ini, buatlah arsitektur ANN Anda sendiri. Anda tidak diperkenankan menggunakan transfer learning pada bagian ANN Training ini.

Catatan mengenai ANN Improvement :
1. Narasikan mengenai apa yang Anda lakukan untuk memperbaiki performansi dari model-model yang tercipta pada ANN Training.
2. Sama seperti ANN Training, Anda dipersilakan membuat arsitektur model dengan Sequential API/Functional API/keduanya.
3. Arsitektur yang dibuat untuk rubric ANN Improvement akan dibandingkan dengan aristektur dari rubric ANN Training dan harus memiliki performansi yang lebih baik.
4. Contoh definisi performansi yang lebih baik :
   - memiliki nilai accuracy atau metrics lain yang lebih baik,
   - proses training lebih cepat meski nilai accuracy sama,
   - nilai accuracy sama namun ukuran model lebih kecil, dll.
5. Model yang dipilih sebagai model terbaik haruslah model dari ANN Improvement, tidak boleh berasal dari ANN Training. Model terbaik ini akan digunakan pada Model Inference dan Model Deployment.
6. Anda diperkenankan menggunakan transfer learning. Ujilah beberapa arsitektur buatan Anda sendiri.
7. Anda dipersilakan menggunakan TensorFlow Callback.

Bagi yang mengambil topik mengenai Computer Vision :
- Semua arsitektur ANN yang dibuat haruslah dilatih dengan menggunakan Image Augmentation.
- Artinya, data yang dipakai untuk melatih semua model berasal dari data yang sama (dengan sudah dilakukan Image Augmentation).
- Tidak diizinkan membuat 2 buah model berbeda dengan data yang tidak sama. Contoh :
   + Model 1 : ANN dengan 5 layer yang dibuat dengan tanpa adanya Image Augmentation.
   + Model 2 : Arsietktur ANN yang sama dengan Model 1 dengan adanya Image Augmentation.
- Yang membedakan antara ANN Training dan ANN Improvement adalah arsitektur ANN yang dibuat, bukan asal datanya.
```

### Readability

| Criteria | Meet Expectations | Points|
| --- | --- | --- |
| Tertata Dengan Baik | Semua baris kode terdokumentasi dengan baik dengan menggunakan Markdown untuk penjelasan kode. | 10 pts |

```
Kriteria tertata dengan baik diantaranya adalah : 

1. Terdapat section Perkenalan yang jelas
2. Tidak menyalin markdown dari tugas lain.
3. Import library rapih (terdapat dalam 1 cell dan tidak ada unused libs).
4. Pemakaian fungsi markdown yang optimal (Heading, text formating, dll). 
5. Terdapat komentar pada setiap baris kode.
6. Adanya pemisah yang jelas antar section, dll.
7. Tidak adanya typo.
```

### Analysis

| Criteria | Meet Expectations | Points|
| --- | --- | --- |
| Model Analysis | Menganalisa informasi dari model yang telah dibuat | 30 pts |
| Overall Analysis | Menarik informasi/kesimpulan dari keseluruhan kegiatan yang dilakukan | 20 pts |

```
Contoh kriteria analisa yang baik diantaranya adalah: 

1. Terdapat penjelasan macam-macam hasil metric evaluasi dan interpretasinya terhadap kasus yang diselesaikan.
2. Dapat menjelaskan kelemahan/kekurangan dan kelebihan dari model yang dibuat.
3. Dapat memberikan statement untuk improvement selanjutnya dari model yang dibuat. 
4. Sebutkan insight yang dapat diambil setelah proses EDA, dll.
```

### Deployment

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Model Deployment | Membuat webapps terhadap project yang telah dibuat. | 10 pts |

```
Catatan mengenai Model Deployment : 

1. Ketiadaan URL deployment ataupun source code deployment di repository, akan tetap diperhitungkan untuk menilai bagian Model Deployment. 
2. Tidak diperkenankan adanya informasi tambahan/informasi susulan seperti lupa memberikan URL deployment atau lupa mengupload source code via apapun (DM buddy, email, atau yang lain).
3. Student akan dianggap tidak melakukan Model Deployment jika tidak ada URL deployment dan source code deployment di repository.
4. Untuk hasil prediksi, ubah kembali menjadi label aslinya sesuai dengan dataset asalnya.
   - Contoh : saat melakukan modeling, Anda melakukan konversi target seperti 
     `car` menjadi `2`
     `bus` menjadi `1`
     `truck` menjadi `0`
   - Saat menampilkan hasil prediction, ubahlah label target ke dalam bentuk string (`car`/`bus`/`truck`).
```

---

```
Total Points : 150
```

---

## Notes

* Deadline : 
  - Default : Phase 2 Week 2 Day 2 pukul 18:00 WIB.
  - Deadline dapat berubah jika terdapat kondisi khusus seperti hari libur, mendapatkan reward EE, atau hal lain.
  - Silakan cek kembali mengenai deadline tugas ini pada repository Anda atau silakan tanyakan ke instruktur/buddy Anda.

* Commit terakhir di GitHub yang akan dinilai. Pastikan tidak ada commit yang dibuat setelah deadline.

* Jika commit terakhir GitHub telah melewati deadline, meskipun ada commit sebelumnya yang belum melewati deadline, akan mengakibatkan tugas Graded Challenge 7 dinyatakan terlambat dan Graded Challenge 7 mendapat nilai 0.
