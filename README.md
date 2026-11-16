# Praktikum Machine Learning: Support Vector Machine (SVM)

## Deskripsi

Repository ini berisi hasil praktikum mata kuliah Machine Learning menggunakan algoritma **Support Vector Machine (SVM)** dengan bahasa pemrograman Python.

Praktikum ini bertujuan untuk memahami penerapan algoritma SVM dalam proses klasifikasi, regresi, serta visualisasi garis pemisah antar kelas (*Decision Boundary*). Terdapat tiga praktikum yang dikerjakan menggunakan dataset dan metode yang berbeda.

## Tujuan Praktikum

- Memahami konsep dasar algoritma Support Vector Machine.
- Menerapkan SVM untuk melakukan klasifikasi data.
- Menerapkan Support Vector Regression untuk melakukan prediksi.
- Memahami proses pembagian data, standarisasi, dan pelatihan model.
- Menampilkan visualisasi hasil pemodelan menggunakan Matplotlib.

## Daftar Isi

- [1. Klasifikasi Diabetes Menggunakan SVM](#1-klasifikasi-diabetes-menggunakan-svm)
- [2. Prediksi Gaji Menggunakan Support Vector Regression (SVR)](#2-prediksi-gaji-menggunakan-support-vector-regression-svr)
- [3. Visualisasi SVM Decision Boundary](#3-visualisasi-svm-decision-boundary)
- [Tools dan Library](#tools-dan-library)
- [Struktur Repository](#struktur-repository)
- [Cara Menjalankan Program](#cara-menjalankan-program)
- [Kesimpulan](#kesimpulan)

---

## 1. Klasifikasi Diabetes Menggunakan SVM

### Deskripsi

Praktikum pertama menggunakan algoritma **Support Vector Classifier (SVC)** untuk melakukan klasifikasi data diabetes. Model digunakan untuk memprediksi kategori diabetes berdasarkan beberapa atribut yang terdapat dalam dataset.

### Dataset: `diabetes.csv`

Dataset memiliki **768 baris** data dan **9 kolom**, yang terdiri dari:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome`

### Tahapan Pengerjaan

1. Mengimpor library Pandas untuk membaca dan mengolah dataset.
2. Membaca file `diabetes.csv` menggunakan fungsi `read_csv()`.
3. Menampilkan beberapa data awal menggunakan `head()`.
4. Memeriksa informasi dataset menggunakan `info()`.
5. Memisahkan atribut atau fitur (`X`) dan label (`y`).
6. Melakukan standarisasi data menggunakan `StandardScaler()`.
7. Membagi dataset menjadi data training dan testing menggunakan `train_test_split()` dengan `test_size=0.33` dan `random_state=42`.
8. Membuat model klasifikasi menggunakan `SVC()`.
9. Melatih model menggunakan fungsi `fit()`.
10. Mengukur performa model menggunakan fungsi `score()`.

### Hasil Praktikum

Berdasarkan hasil pengujian, model SVC memperoleh nilai akurasi sebesar **75,59%** pada data testing. Hasil tersebut menunjukkan bahwa model mampu melakukan klasifikasi terhadap data diabetes dengan tingkat akurasi sebesar 75,59%.

---

## 2. Prediksi Gaji Menggunakan Support Vector Regression (SVR)

### Deskripsi

Praktikum kedua menggunakan metode **Support Vector Regression (SVR)** untuk memprediksi besaran gaji berdasarkan pengalaman kerja seseorang.

### Dataset: `Salary_Data.csv`

Dataset memiliki **30 baris** data dengan **2 kolom**, yaitu:

| Kolom | Keterangan |
|---|---|
| `YearsExperience` | Lama pengalaman kerja |
| `Salary` | Besaran gaji |

### Tahapan Pengerjaan

1. Mengimpor library Pandas untuk membaca dataset.
2. Membaca file `Salary_Data.csv` menggunakan `read_csv()`.
3. Memeriksa informasi dataset menggunakan `info()`.
4. Menampilkan beberapa data awal menggunakan `head()`.
5. Memisahkan atribut `YearsExperience` sebagai variabel `X` dan `Salary` sebagai variabel `y`.
6. Mengubah bentuk data `X` menggunakan NumPy agar sesuai dengan format input model.
7. Mengimpor algoritma SVR dari Scikit-learn.
8. Membuat model SVR menggunakan kernel RBF dengan parameter `C = 1000` dan `gamma = 0.05`.
9. Melatih model menggunakan fungsi `fit()`.
10. Menampilkan visualisasi data aktual dan hasil prediksi menggunakan Matplotlib.

### Hasil Praktikum

Hasil praktikum menampilkan hubungan antara pengalaman kerja dan besaran gaji melalui grafik *scatter plot*. Garis hasil prediksi model SVR ditampilkan untuk memperlihatkan pola hubungan antara kedua variabel tersebut.

---

## 3. Visualisasi SVM Decision Boundary

### Deskripsi

Praktikum ketiga membahas penerapan algoritma SVM dengan **kernel linear** untuk menentukan garis pemisah antara dua kelompok data.

Dataset dibuat secara sederhana menggunakan NumPy dengan dua atribut:

| Atribut | Keterangan |
|---|---|
| `Study Hours` | Jumlah jam belajar |
| `Sleep Hours` | Jumlah jam tidur |

Data dibagi menjadi dua kelas:

| Kelas | Keterangan |
|---|---|
| `Passed (1)` | Kelompok yang dinyatakan lulus |
| `Failed (-1)` | Kelompok yang dinyatakan tidak lulus |

### Tahapan Pengerjaan

1. Mengimpor library NumPy dan Matplotlib.
2. Mengimpor modul SVM dari Scikit-learn.
3. Membuat dataset menggunakan array NumPy.
4. Menentukan atribut `Study Hours` dan `Sleep Hours`.
5. Menentukan label untuk masing-masing kelompok data.
6. Menggabungkan kedua atribut menggunakan `np.column_stack()`.
7. Membuat model SVM menggunakan kernel linear.
8. Melatih model menggunakan fungsi `fit()`.
9. Mengambil nilai koefisien dan intercept dari model.
10. Menghitung garis pemisah atau *Decision Boundary*.
11. Menampilkan data dalam bentuk *scatter plot*.
12. Menampilkan garis *Decision Boundary* menggunakan Matplotlib.

### Hasil Praktikum

Hasil visualisasi menampilkan dua kelompok data dengan warna yang berbeda. Garis berwarna hijau menunjukkan *Decision Boundary* yang dihasilkan oleh model SVM.

Visualisasi tersebut membantu memahami bagaimana algoritma SVM memisahkan dua kelompok data berdasarkan atribut yang digunakan.

---

## Tools dan Library

| Tools / Library | Kegunaan |
|---|---|
| Python | Bahasa pemrograman |
| Jupyter Notebook | Menjalankan program |
| Pandas | Membaca dan mengolah dataset |
| NumPy | Pengolahan data numerik |
| Matplotlib | Visualisasi data |
| Scikit-learn | Pembuatan dan pelatihan model Machine Learning |

## Struktur Repository

```
praktikum-machine-learning-svm/
│
├── SVM KLASIFIKASI.ipynb
├── diabetes.csv
│
├── salary.ipynb
├── Salary_Data.csv
│
├── SVM NOT OK.ipynb
│
└── README.md
```

## Cara Menjalankan Program

1. Clone repository dari GitHub:

   ```bash
   git clone https://github.com/USERNAME/praktikum-machine-learning-svm.git
   ```

2. Masuk ke direktori repository:

   ```bash
   cd praktikum-machine-learning-svm
   ```

3. Install library yang dibutuhkan:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

4. Buka folder menggunakan Visual Studio Code atau Jupyter Notebook.
5. Buka file notebook sesuai praktikum yang ingin dijalankan.
6. Pastikan dataset berada dalam folder yang sama dengan file notebook terkait.
7. Jalankan setiap cell secara berurutan untuk melihat proses dan hasil praktikum.

## Kesimpulan

Berdasarkan ketiga praktikum yang telah dilakukan, algoritma Support Vector Machine dapat diterapkan untuk menyelesaikan beberapa permasalahan dalam Machine Learning.

- **Praktikum 1:** metode SVC digunakan untuk klasifikasi diabetes dan memperoleh akurasi sebesar 75,59%.
- **Praktikum 2:** metode SVR digunakan untuk memprediksi gaji berdasarkan pengalaman kerja dengan kernel RBF.
- **Praktikum 3:** SVM dengan kernel linear digunakan untuk memahami proses pemisahan dua kelompok data melalui *Decision Boundary*.

Dari seluruh praktikum tersebut, dapat dipahami bahwa SVM tidak hanya digunakan untuk klasifikasi, tetapi juga dapat diterapkan pada regresi serta visualisasi pemisahan data.
