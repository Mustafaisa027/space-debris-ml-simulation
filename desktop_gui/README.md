# Orbital Sentinel Desktop GUI

PySide6 tabanli Mission Control masaustu arayuzu, `gui/data/simulation.json`
dosyasindaki sabit replay ve makale agregasini goruntuler. Uygulama alti sahne sunar:
acilis brifingi, 3B yorunge simulasyonu, konjonksiyon detayi, model karsilastirma,
veri kalitesi/seffaflik ve kapanis ozeti.

Bu paket, eski `gui/` web prototipinin yerini alan birincil GUI'dir. Web prototipi
referans ve geriye donuk kullanim icin depoda korunur.

## Kurulum

Proje kokunde:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

OpenCV gerekmez. Video kayit katmani bilerek dahil edilmemistir; kayit harici ekran
kaydedici ile alinmalidir.

## Calistirma

```powershell
.\.venv\Scripts\python.exe -m desktop_gui.main
```

Farkli bir veri dosyasi kullanmak icin:

```powershell
.\.venv\Scripts\python.exe -m desktop_gui.main --data path\to\simulation.json
```

PyVista/VTK olusturmadan guvenli kipte baslatmak icin:

```powershell
.\.venv\Scripts\python.exe -m desktop_gui.main --no-3d
```

Normal baslangicta 3B varsayilandir. Uygulama once ayri ve zaman sinirli bir alt
surecte Qt/VTK/OpenGL yetenek probu calistirir. Prob basarisiz olur veya zaman asimina
ugrarsa diger alti sayfa calismaya devam eder ve 3B alanda guvenli kip bildirimi
gosterilir. Renderer tanilamasi `output/desktop_gui_renderer.log` dosyasina yazilir.

3B sahne, JSON'un `replay.tle_source` alaninda isaret edilen arsivlenmis 75-nesne
TLE katalogunu SGP4 ile snapshot zamanindan itibaren ilerletir. Dunya, atmosfer ve
nesne boyutlari sunum amacli olup olcege gore degildir; sayisal konjonksiyon degerleri
yalnizca `simulation.json` kaynaklidir. Harici ag veya canli veri servisi kullanilmaz.
`assets/earth_daymap.png`, 3B kure icin yerel 2:1 equirectangular gunduz dokusudur.
Bu varlik yerlesik ImageGen araci ile proje icin uretilmistir; harici bir harita
dosyasi veya atif gerektiren ucuncu taraf varligi kullanilmaz.

SGP4 konumlari ham TEME cercevesinde ve kilometre biriminde gosterilir. Kose yonelim
gostergesi bu nedenle `FRAME: TEME` olarak etiketlidir; Earth-fixed donusum iddiasi
yapilmaz. Secili ciftte cyan PRIMARY, amber SECONDARY, kirmizi TCA/minimum ayrim
geometrisini gosterir. TCA oncesi yerel yol daha guclu, TCA sonrasi yol daha soluktur.

3B arac cubugundaki `ALL`, `SELECTED` ve `TCA` kipleri ayni sahne aktorlerinin gorunur
hiyerarsisini degistirir. `ALL`, sekiz encounter'in E01-E08 konumlarini her event'in
kendi TCA zamaninda gosterir; bu marker'lar eszamanli uydu konumlari degildir.
`SELECTED`, diger event'leri baglam olarak korurken secili cifti vurgular. `TCA`, secili
miss geometrisine fiziksel konumlari degistirmeden yaklasir.

Kanonik canli mesafe, JSON'daki screened `distance_series` verisinin gercek simulasyon
dakikasindaki dogrusal interpolasyonudur ve arayuzde `[SERIES]` olarak belirtilir. 3B
konumlar ayni arsivlenmis TLE'lerden ham SGP4 ile uretilir. Regresyon denetimi sekiz
senaryoda saklanan seri ile SGP4 geometrisinin 0.001 km icinde uyumlu olmasini ister.
TCA miss cizgisi iki gercek SGP4 konumu arasindadir; etiketteki miss degeri kaynak
`min_distance_km` alanidir.

Model panosunda PR-AUC, F1 ve recall sabit makale agregasindan gelir. Yalnizca
`(yerel)` etiketli ROC-AUC, precision ve false-alarm-rate alanlari JSON icindeki ayri
75-nesne gelistirme raporundan okunur ve makale sonucu olarak sunulmaz.

## Test

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest tests\test_desktop_gui_smoke.py -q -p no:cacheprovider
```

Smoke testi veri sozlesmesini, alti sahnenin kurulmasini, sahne gecislerini ve temel
zaman cizelgesi davranisini 3B renderer acmadan dogrular.
