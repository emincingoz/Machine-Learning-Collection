# Association Rule Mining/Learning

Olayların birlikte gerçekleşme durumlarını tespit edip, veriler arasında ilişkiler kuran bir makine öğrenmesi yöntemidir.

Kategorik verilerle başarılı bir şekilde çalışabilirler.

#### Causation vs Correlation 
![image](https://user-images.githubusercontent.com/49842813/162502513-60b9f492-d9cb-4108-af83-c28822fae658.png)

**Support** (Destek): Bir varlığı içeren eylem sayısının, toplam eylem sayısınıa oranıdır. (ex: 100 kişiye soruldu, ...)
![image](https://user-images.githubusercontent.com/49842813/162508370-5cb43208-9946-44be-9ffc-2dacec6a5dcb.png)

**Confidence** (Güven): İki varlığı içeren eylem sayısının birisine oranıdır. (ex: a ürününü alan 100 kişiye soruldu, ...)
![image](https://user-images.githubusercontent.com/49842813/162509648-2cdc8086-6deb-4f2b-8bdb-6f050ef4898b.png)

**Lift**: Bir eylemin diğer eyleme ne kadar etki ettiğini gösterir. (ex: a ürününün alınması b ürününün alınmasını nasıl etkiledi)

* Lift(a->b) < 1 ise A eyleminin yapılması, B eylemini olumsuz etkiler.
* Lift(a->b) > 1 ise A eyleminin yapılması, B eylemini olumlu etkiler.

![image](https://user-images.githubusercontent.com/49842813/162510722-0618ccb1-98c6-4c96-af82-9e979bbf8890.png)

---


![image](https://user-images.githubusercontent.com/49842813/162512011-a77c3ade-8221-42cf-9d08-e0f89c3661d4.png)



