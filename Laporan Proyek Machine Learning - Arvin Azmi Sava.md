# Laporan Proyek Machine Learning - Arvin Azmi Sava
## Domain Proyek
Peramalan harga saham merupakan salah satu tantangan dalam dunia keuangan yang memiliki dampak signifikan bagi investor dan institusi keuangan. Dengan volatilitas yang tinggi, kemampuan untuk memprediksi harga saham di masa depan dapat memberikan keuntungan kompetitif dalam pengambilan keputusan investasi.

Peramalan harga saham penting karena membantu investor dan institusi keuangan dalam mengambil keputusan yang lebih baik. Dengan prediksi yang akurat, risiko kerugian dapat dikurangi, dan peluang keuntungan bisa dimaksimalkan. Selain itu, model prediksi yang baik dapat mempercepat dan meningkatkan efisiensi perdagangan saham. Perkembangan teknologi memungkinkan penggunaan machine learning untuk memahami pola dalam data saham yang sulit dilihat secara langsung. Oleh karena itu, pengembangan model yang lebih akurat sangat diperlukan agar keputusan investasi lebih optimal.

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


## Data Understanding
Dataset yang digunakan dalam proyek ini diperoleh dari Kaggle dengan url https://www.kaggle.com/datasets/rpaguirre/tesla-stock-price/data

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
- Uji korelasi antara variabel-variabel dalam dataset.

## Data Preparation
- Deteksi Outlier
- Pembagian Data Latih dan Uji
- Normalisasi
- Pembuatan Urutan Data Baru

## Modeling
Pada proyek ini menggunakan dua model deep learning yaitu Bidirectional LSTM dan Bidirectional Gater Reccurrent Unit.
### Bidirectional Long Short-Term Memory (BILSTM)
Model Long Short-Term Memory (LSTM) merupakan pembaruan dari model Recurrent 
Neural Network (RNN) yang pertama kali diperkenalkan oleh Hochreiter dan Schmidhuber 
pada tahun 1997.  LSTM didesain untuk menangani persoalan yang muncul pada RNN berupa 
long term dependency problem [29]. Masalah tersebut menyebabkan RNN akan kehilangan 
informasi penting yang didapatkan sebelumnya di awal jika urutannya cukup panjang pada saat 
forward propagation. Bidirectional LSTM adalah varian dari LSTM yang memungkinkan informasi diproses dalam dua arah yaitu arah maju dan mundur. Model ini menggunakan dua lapisan LSTM untuk menangkap konteks sebelumnya dan berikutnya dalam urutan data sehingga meningkatkan pemahaman konteks data [30].

### Bidirectional Gated Recurrent Unit (BIGRU)
```sh
cd dillinger
npm i
node app
```

For production environments...

```sh
npm install --production
NODE_ENV=production node app
```

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
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - f_i)^2}
$$


2. Mean Absolute Percentage Error (SMAPE)
3. Mean Absolute Error (MAE)
4. 




| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantaneously see your updates!

Open your favorite Terminal and run these commands.

First Tab:

```sh
node app
```

Second Tab:

```sh
gulp watch
```

(optional) Third:

```sh
karma test
```

#### Building for source

For production release:

```sh
gulp build --prod
```

Generating pre-built zip archives for distribution:

```sh
gulp build dist --prod
```

## Docker

Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the
Dockerfile if necessary. When ready, simply use the Dockerfile to
build the image.

```sh
cd dillinger
docker build -t <youruser>/dillinger:${package.json.version} .
```

This will create the dillinger image and pull in the necessary dependencies.
Be sure to swap out `${package.json.version}` with the actual
version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on
your host. In this example, we simply map port 8000 of the host to
port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart=always --cap-add=SYS_ADMIN --name=dillinger <youruser>/dillinger:${package.json.version}
```

> Note: `--capt-add=SYS-ADMIN` is required for PDF rendering.

Verify the deployment by navigating to your server address in
your preferred browser.

```sh
127.0.0.1:8000
```

## License

MIT

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
