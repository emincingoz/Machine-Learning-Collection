## PCA (Principal Component Analysis)

* PCA'in temel amacı verilerdeki kalıpları/modelleri belirlemektir. PCA veriler arasındaki korelasyonu tespit etmeyi amaçlar.
* Boyut indirgeme yalnızca veriler arasında güçlü bir korelasyon mevcut olduğunda anlamlıdır.
* PCA'in olayı, çok boyutlu verilerde maksimum varyans yönlerini bulmak ve bilgilerin olabildiğince çoğunu koruyarak onu daha küçük boyutlu alt uzaya yansıtmaktır.

### Summary of PCA

1. Veriler standartlaştırılır
2. Kovaryans matrisinden veya korelasyon matrisinden öz değerler (eigen value) ve öz vektörler (eigen vector) elde edilir. Veya tekil değer ayrıştırması (Singular Value Decomposition/SVD) gerçekleştirilir.
3. Öz değerler büyükten küçüğe sıralanır ve en büyük k tanesi seçilir. (k: Oluşturulacak alt uzayın boyut sayısı)
4. Seçilen k sayıda eigen vektörden W izdüşüm matrisi oluşturulur.
5. Yeni k boyutlu özellik alt uzayı Y elde etmek için orjinal veri kümesi W (izdüşüm) matrisi ile dönüştürülür.

## LDA (Linear Discriminant Analysis)

* Genel LDA yaklaşımı, PCA ile oldukça benzerdir. Verilerin varyansını maksimuma çıkaran eksenleri bulmaya ek olarak birden fazla sınıf arasındaki ayrımı en üst düzeye çıkaran eksenlerin bulunması amaçlanır.

## PCA vs LDA

* LDA ve PCA, boyut indirgeme için kullanılan iki lineer dönüşüm tekniğidir.
* PCA, gözetimsiz (unsupervised) bir algoritma yöntemi (sınıf etiketleri gözardı edilir.) iken LDA, gözetimli (supervised) bir algoritmadır (sınıf etiketleri önemlidir.).
* LDA’de amaç sınıfları birbirinden ayıran en iyi boyutu bulmaktır. PCA’de ise amaç verileri birbirinden ayrıştıran en iyi boyutu bulmaktır.

 ![image](https://user-images.githubusercontent.com/49842813/153505867-632ee8f4-ffe7-454f-aa22-ce5beb624a57.png)
