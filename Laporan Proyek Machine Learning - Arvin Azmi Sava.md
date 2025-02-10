# Laporan Proyek Machine Learning - Arvin Azmi Sava
## Domain Proyek
Peramalan harga saham merupakan salah satu tantangan dalam dunia keuangan yang memiliki dampak signifikan bagi investor dan institusi keuangan. Dengan volatilitas yang tinggi, kemampuan untuk memprediksi harga saham di masa depan dapat memberikan keuntungan kompetitif dalam pengambilan keputusan investasi.

Peramalan harga saham penting karena membantu investor dan institusi keuangan dalam mengambil keputusan yang lebih baik [3]. Dengan prediksi yang akurat, risiko kerugian dapat dikurangi, dan peluang keuntungan bisa dimaksimalkan. Selain itu, model prediksi yang baik dapat mempercepat dan meningkatkan efisiensi perdagangan saham. Perkembangan teknologi memungkinkan penggunaan machine learning untuk memahami pola dalam data saham yang sulit dilihat secara langsung. Oleh karena itu, pengembangan model yang lebih akurat sangat diperlukan agar keputusan investasi lebih optimal.

## Business Understanding
### Problem Statements
1. Bagaimana cara memprediksi harga saham berdasarkan data historis?
2. Bagaimana efektivitas model BiLSTM dan BiGRU dalam meningkatkan akurasi prediksi harga saham?
3. Bagaimana pengaruh berbagai fitur (seperti harga pembukaan, harga tertinggi, volume) terhadap prediksi harga saham?
### Goals
1. Mengembangkan model machine learning untuk memprediksi harga saham.
2. Membandingkan performa BiLSTM dan BiGRU menggunakan metrik evaluasi yang sesuai.
3. Mengidentifikasi fitur yang paling berpengaruh dalam peramalan harga saham.

### Solution Statements
1. Menggunakan Bidirectional Long Short-Term Memory (BiLSTM) dan Bidirectional Gated Recurrent Unit (BiGRU) sebagai model deep learning untuk menangkap pola dalam data time series harga saham.
2. Melakukan hyperparameter tuning untuk meningkatkan akurasi model dengan menyesuaikan jumlah unit, dropout rate, dan optimizer.


## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan dataset harga saham harian Tesla yang mencakup periode tertentu dari periode 06/29/2010 hingga 03/17/2017. Dataset ini diperoleh dari Kaggle melalui tautan berikut: https://www.kaggle.com/datasets/rpaguirre/tesla-stock-price/data

### Variabel dalam dataset:
- Date - Tanggal pencatatan harga saham.
- Open - Harga saham pada awal perdagangan.
- High - Harga tertinggi yang dicapai dalam satu hari perdagangan.
- Low - Harga terendah dalam satu hari perdagangan.
- Close - Harga penutupan saham pada akhir perdagangan.
- Adj Close - Harga penutupan yang disesuaikan.
- Volume - Jumlah saham yang diperdagangkan dalam satu hari.
### Exploratory Data Analysis (EDA)
- Plot harga saham historis untuk melihat tren.

![image](https://github.com/user-attachments/assets/71545ac0-de14-46c2-932b-1aa6f8c5b51c) ![image](https://github.com/user-attachments/assets/5652cc40-36da-40ec-9c3b-9b171efaddc2)


- Uji korelasi antara variabel-variabel dalam dataset.

![image](https://github.com/user-attachments/assets/8d3e50ee-3bec-4966-9c03-7ba572c6a86d)

Berdasarkan uji korelasi yang ditunjukkan pada gambar di atas. Terdapat beberapa variabel yaitu variabel Open, High, dan Low yang memiliki korelasi sangat kuat dengan variabel target yaitu harga penutupan (Close), sedangkan Volume memiliki korelasi yang lebih rendah dibandingkan variabel lainnya.Meskipun korelasi Volume dengan Close lebih rendah, fitur ini tetap digunakan dalam pemodelan karena volume perdagangan dapat memberikan informasi tambahan mengenai pergerakan harga saham. Hasil uji korelasi ini menunjukkan bahwa fitur Open, High, dan Low berpengaruh besar terhadap prediksi harga saham, sementara Volume memiliki pengaruh yang lebih rendah. Dengan mempertimbangkan semua fitur yang tersedia, model BiLSTM dan BiGRU akan dievaluasi untuk menentukan efektivitasnya dalam meningkatkan akurasi prediksi harga saham.
## Data Preparation
- Deteksi Outlier
  ![image](https://github.com/user-attachments/assets/be12abff-9cc9-4441-b461-94ef576e0954)

  Berdasarkan visualisasi boxplot, terdapat outlier pada variabel Volume. Oleh karena itu, akan dilakukan pengecekan outlier tersebut. Kemudian, berikutnya outlier tersebut akan dihapus / didrop karena jumlahnya tidak begitu banyak sehingga tidak berpengaruh signifikan terhadap dataset yang ada

  ![image](https://github.com/user-attachments/assets/71a5201c-7e43-48c2-b3b4-86e70eec2b5a)

  Penghapusan outliers dilakukan menggunakan metode IQR. Outlier berjumlah 80. Setelah dilakukan penghapusan baris data yang mengandung outliers, saat ini total baris data berjumlah 1612 dari sebelumnya berjumlah 1692 baris.


- Pembagian Data Latih dan Uji

Pada tahap ini dilakukan pembagian data menjadi menjadi dua bagian yaitu data latih dan data uji. Pembagian data latih dan data uji ini menggunakan  skenario 80:20. Skenario ini dirancang untuk menguji seberapa baik kemampuan model dalam memprediksi data yang belum pernah dilihat sebelumnya. Setelah pembagian data, dilakukan penentuan variabel fitur dan target. Variabel fitur terdiri dari kolom Open, High, Low, dan Volume, sedangkan variabel targetnya yaitu Close. 
- Normalisasi

Normalisasi adalah proses mengubah nilai-nilai dari suatu dataset ke dalam rentang nilai tertentu. Tujuan utama normalisasi adalah untuk menghasilkan data yang konsisten sehingga setiap variabel memiliki pangaruh yang seimbang terhadap model yang dibangun [2]. Selain itu, normalisasi juga akan mengurangi bias yang mungkin terjadi akibat perbedaan skala antar variabel. Dengan demikian, normalisasi sangat penting dalam pembuatan model karena dapat menghasilkan model yang lebih stabil dan akurat. Hasil normalisasi pada variabel fitur ditunjukkan pada gambar di bawah ini

![image](https://github.com/user-attachments/assets/a18ac943-5bc1-494e-99c6-af4881f5eddc)

Sementara itu, untuk hasil normalisasi variabel target ditunjukkan gambar di bawah ini

![image](https://github.com/user-attachments/assets/b553097d-c425-49df-8ae9-cc2d1b7a8806)


- Pembuatan Urutan Data Baru

Setelah melakukan normalisasi data, tahap berikutnya adalah pembuatan urutan data baru menjadi ukuran 3 dimensi (samples, timesteps, jumlah fitur) agar sesuai dengan input yang diperlukan oleh model biLSTM. Timesteps digunakan untuk menentukan jumlah data masa lalu yang diperhitungkan dalam memprediksi satu nilai di masa depan. Pada tahap ini timesteps yang digunakan adalah 9 sehingga setiap prediksi di masa depan mempertimbangkan 9data di masa lalu. Pada Gambar di bawah ini ditunjukkan bahwa ukuran data latih dan data uji sudah berubah menjadi ukuran 3 dimensi.

![image](https://github.com/user-attachments/assets/a50c404a-4a97-4c35-bb27-fcc4e3c3b660)

## Modeling
Pada proyek ini, digunakan dua model deep learning, yaitu Bidirectional LSTM (BiLSTM) dan Bidirectional Gated Recurrent Unit (BiGRU), untuk memprediksi harga saham Tesla berdasarkan data historis. Kedua model ini dikembangkan dengan hypertuning untuk mencari kombinasi parameter terbaik yang menghasilkan performa optimal.

Proses hypertuning dilakukan menggunakan Keras Tuner dengan metode Hyperband. Metode Hyperband dipilih karena mampu mengalokasikan sumber daya secara efisien dalam eksplorasi hyperparameter, dengan prinsip early stopping untuk mengeliminasi konfigurasi yang kurang menjanjikan lebih awal. Keuntungan utama metode ini adalah:

- Efisiensi waktu: Mengurangi jumlah eksperimen yang harus dilakukan dibandingkan pencarian grid search atau random search.
- Seleksi adaptif: Model dengan performa buruk dieliminasi lebih awal, sehingga sumber daya lebih banyak dialokasikan ke model yang lebih menjanjikan.
- Optimasi yang lebih cepat: Memungkinkan eksplorasi berbagai kombinasi hyperparameter tanpa perlu melatih semua model hingga selesai.
  
Parameter terbaik yang diperoleh dari hasil tuning untuk model BiLSTM adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/b4e071d3-9606-4aad-8c4f-3556de773dd8)


Sementara itu, parameter terbaik hasil tuning untuk model BiGRU adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/6950cbe0-dc58-4a90-9dae-7a0367907035)

Setelah mendapatkan parameter terbaik dari hasil tuning, model final kemudian dibangun dan dilatih menggunakan parameter tersebut.
### Bidirectional Long Short-Term Memory (BILSTM)
Model Long Short-Term Memory (LSTM) merupakan pembaruan dari model Recurrent 
Neural Network (RNN) yang pertama kali diperkenalkan oleh Hochreiter dan Schmidhuber 
pada tahun 1997.  LSTM didesain untuk menangani persoalan yang muncul pada RNN berupa 
long term dependency problem [29]. Masalah tersebut menyebabkan RNN akan kehilangan 
informasi penting yang didapatkan sebelumnya di awal jika urutannya cukup panjang pada saat 
forward propagation. Bidirectional LSTM adalah varian dari LSTM yang memungkinkan informasi diproses dalam dua arah yaitu arah maju dan mundur. Model ini menggunakan dua lapisan LSTM untuk menangkap konteks sebelumnya dan berikutnya dalam urutan data sehingga meningkatkan pemahaman konteks data [1].

### Bidirectional Gated Recurrent Unit (BIGRU)
GRU diperkenalkan oleh Cho et al. pada tahun 2014 sebagai alternatif yang lebih sederhana dibandingkan Long Short-Term Memory (LSTM) dengan tetap mempertahankan kemampuan menangani dependensi jangka panjang.
GRU memiliki dua mekanisme utama, yaitu reset gate dan update gate. Reset gate mengontrol seberapa banyak informasi dari langkah sebelumnya yang perlu dilupakan, sedangkan update gate menentukan informasi baru yang akan disimpan dalam memori. Dengan struktur ini, GRU dapat menyesuaikan aliran informasi tanpa menggunakan memori sel seperti pada LSTM sehingga BiGRU sering kali lebih efisien secara komputasi dengan performa yang tetap kompetitif dalam berbagai aplikasi. Bidirectional GRU (BiGRU) adalah varian dari GRU yang memproses informasi dalam dua arah, yaitu maju (forward) dan mundur (backward). BiGRU menggunakan dua lapisan GRU yang berjalan secara paralel untuk menangkap informasi dari konteks sebelumnya maupun berikutnya dalam urutan data. Dengan pendekatan ini, BiGRU lebih efektif dalam memahami hubungan dalam data sekuensial dibandingkan GRU biasa, terutama dalam aplikasi seperti pemrosesan bahasa alami dan analisis deret waktu.

## Evaluation
Setelah memperoleh hasil prediksi dari model yang telah dibangun. Perlu dilakukan 
evaluasi kinerja model. Hal ini bertujuan untuk menillai seberapa akurat model yang telah 
dibuat dalam memprediksi data. Berikut ini beberapa metrik evaluasi kinerja yang digunakan 
untuk menilai seberapa baik peramalan yang dihasilkan:
1. Mean Square Error (MSE)

Mean Square Error (MSE) merupakan salah satu metrik evaluasi yang umum 
digunakan untuk mengukur seberapa baik model memprediksi nilai tertentu. MSE didapatkan 
dengan cara mengukur hasil akar dari rata-rata perbedaan kuadrat antara nilai aktual (y) dan 
nilai hasil prediksi (y). Rumus MSE dapat dinyatakan sebagai berikut:

   $$
   MSE = \frac{1}{n} \sum_{i=1}^{n} (x_i - f_i)^2
   $$

Di mana

n = Jumlah data

xi= Nilai prediksi pada periode ke-i 

fi = Nilai actual indeks pada periode ke-i

2. Mean Absolute Percentage Error (SMAPE)

Mean Absolute Percentage Error (MAPE) merupakan salah satu metrik evaluasi yang umum digunakan untuk mengukur seberapa baik model memprediksi nilai tertentu.  
MAPE didapatkan dengan cara menghitung rata-rata persentase kesalahan absolut antara nilai aktual (x) dan nilai hasil prediksi (f).  
Rumus MAPE dapat dinyatakan sebagai berikut:

$$
MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{x_i - f_i}{x_i} \right| \times 100\%
$$

Di mana

n = jumlah data

xi = Nilai Aktual Indeks pada periode ke-i 

fi= Nilai Prediksi Indeks pada periode ke-i


3. Mean Absolute Error (MAE)
   
Mean Absolute Error (MAE) merupakan rata-rata perbedaan antara nilai data aktual dan nilai prediksi dari model. Secara umum rumus matematis MAE dapat dinyatakan sebagai berikut:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |x_i - f_i|
$$

Di mana

n = jumlah data

xi = Nilai Aktual Indeks pada periode ke-i 

fi= Nilai Prediksi Indeks pada periode ke-i

4. R-Squared (R²)

R-Squared (R²) atau koefisien determinasi merupakan metrik evaluasi yang digunakan untuk mengukur seberapa baik model dapat menjelaskan variabilitas data aktual. Nilai R² berkisar antara 0 hingga 1, di mana semakin mendekati 1, maka model semakin baik dalam menjelaskan variasi data. Semakin tinggi nilai R², semakin baik model dalam menjelaskan variasi data aktual.

### Hasil Evaluasi Model BiLSTM dan BiGRU

| Metrik      | BiLSTM  | BiGRU  |
|------------|--------|--------|
| MSE        | 76.1575 | 69.4211 |
| MAPE (%)   | 3.2016  | 3.0134  |
| MAE        | 6.7463  | 6.3869  |
| R-Squared  | 0.8776  | 0.8884  |

Analisis Hasil Evaluasi:

- MSE (Mean Squared Error): BiGRU memiliki nilai MSE yang lebih rendah dibandingkan BiLSTM, menunjukkan bahwa model ini menghasilkan error yang lebih kecil.

- MAPE (Mean Absolute Percentage Error): BiGRU memiliki MAPE yang lebih rendah, menunjukkan bahwa kesalahan relatif model ini lebih kecil dibandingkan BiLSTM.

- MAE (Mean Absolute Error): BiGRU memiliki MAE yang lebih kecil, yang berarti model ini menghasilkan prediksi yang lebih akurat secara absolut.

- R-Squared (R²): BiGRU memiliki nilai R² lebih tinggi, menunjukkan bahwa model ini lebih mampu menjelaskan variasi dalam data dibandingkan BiLSTM.

### Dampak terhadap Business Understanding

Model prediksi harga saham yang dikembangkan memiliki dampak signifikan terhadap pemahaman bisnis, terutama dalam pengambilan keputusan investasi. Berikut adalah beberapa aspek utama dampaknya:

1. Menjawab Problem Statement dan Mencapai Goals

- Model yang dikembangkan mampu memprediksi harga saham berdasarkan data historis dengan tingkat akurasi yang baik, membantu investor dalam membuat keputusan lebih tepat.

- Penggunaan BiLSTM dan BiGRU menunjukkan bahwa deep learning efektif dalam memprediksi harga saham, metrik evaluasi menunjukkan bahwa BiGRU lebih optimal dengan error yang lebih rendah dan nilai R² yang lebih tinggi
  
- Variabel fitur Open, High, dan Low memiliki pengaruh besar terhadap variabel target Close (mendekati 1) dalam prediksi harga saham. Sementara itu, variabel fitur Volume menunjukkan korelasi yang rendah dengan variabel Close (0,40) yang menunjukkan bahwa perubahan harga tidak selalu berkaitan dengan jumlah volume perdagangan.

2. Dampak dari Solusi Statement

- Penggunaan BiLSTM dan BiGRU membuktikan bahwa deep learning dapat menangkap pola harga saham dengan baik, memberikan metode yang lebih canggih dibandingkan pendekatan konvensional.

- Hasil tuning hyperparameter yang dilakukan berhasil meningkatkan kinerja model secara signifikan, menjadikan model lebih akurat dan andal.

- Model yang dikembangkan dapat menjadi acuan bagi investor dan analis dalam pengambilan keputusan bisnis terkait investasi saham, sehingga dapat meminimalkan risiko dan memaksimalkan keuntungan.
## Kesimpulan
Berdasarkan hasil evaluasi, model algoritma BiGRU dipilih sebagai model terbaik untuk memprediksi harga saham. Hal ini didasarkan pada metrik evaluasi yang menunjukkan bahwa BiGRU memiliki error yang lebih rendah (MSE, MAE, dan MAPE) serta nilai R² yang lebih tinggi dibandingkan BiLSTM. Selain itu, BiGRU lebih efisien dalam proses pelatihan dibandingkan BiLSTM, sehingga lebih optimal untuk digunakan dalam peramalan harga saham.


## Refrensi



[1] A. Graves and J. Schmidhuber, “Framewise phoneme classification with bidirectional LSTM and other neural network architectures,” in Neural Networks, Jul. 2005, pp. 602–610. doi: 10.1016/j.neunet.2005.06.042.

[2] J. Han, M. Kamber, and J. Pei, “Data Mining. Concepts and Techniques, 3rd Edition (The Morgan Kaufmann Series in Data Management Systems),” 2011.

[3] Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). "Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques." Expert Systems with Applications, 42(1), 259-268. https://doi.org/10.1016/j.eswa.2014.08.002
