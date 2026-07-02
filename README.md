# Space Debris Conjunction Simulation

Bu proje, PDF'teki ozet hedefe gore LEO objeleri icin basit ama duzenli bir
conjunction analizi altyapisi kurar:

- TLE verisini okur.
- Skyfield/SGP4 ile objeleri zaman icinde ilerletir.
- Uydu ciftleri icin minimum yaklasma mesafesi, TCA, bagil hiz ve LEO irtifa
  ozelliklerini hesaplar.
- Potansiyel yakin gecisleri `identified_conjunctions.csv` dosyasina ayirir.
- Dataset uretir ve sabit mesafe esigi ile ML modellerini karsilastirir.

## Kurulum

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`xgboost` abstract'taki model karsilastirmasi icin requirements'a dahildir.

## Tek seferlik calistirma

```powershell
.\.venv\Scripts\python.exe src\run_pipeline.py
```

Guncel TLE cekmek icin:

```powershell
.\.venv\Scripts\python.exe src\fetch_tles.py
.\.venv\Scripts\python.exe src\run_pipeline.py --tle data\live_tles_YYYYMMDD_HHMMSS.txt
```

CelesTrak group verisi cekmek icin:

```powershell
.\.venv\Scripts\python.exe src\fetch_tles.py --group STATIONS WEATHER --max-objects 75 --output data\leo_group_sample.txt
.\.venv\Scripts\python.exe src\run_pipeline.py --tle data\leo_group_sample.txt
```

## 60 gunluk deney

CelesTrak GP verisi 2 saatten sik cekilmemelidir. Collector bunu alt sinir
olarak uygular.

Tek deneme:

```powershell
.\.venv\Scripts\python.exe src\collect_observations.py --once
```

60 gun calistirma:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_collector_60_days.ps1
```

Terminal acik kalmadan Windows gorevi olarak calistirma:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_collector_task.ps1
```

Gorevi kaldirma:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\remove_collector_task.ps1
```

Tarihsel veri uzerinden zaman bazli model karsilastirmasi:

```powershell
.\.venv\Scripts\python.exe src\train_from_history.py
```

Ciktilar:

- `outputs/pipeline/conjunction_dataset.csv`
- `outputs/pipeline/identified_conjunctions.csv`
- `outputs/pipeline/model_comparison.csv`
- `outputs/pipeline/top_pair_distance_timeseries.csv`
- `outputs/pipeline/top_pair_distance.png`
- `outputs/pipeline/risk_feature_space.png`
- `outputs/pipeline/tca_distance_scatter.png`
- `outputs/pipeline/altitude_distance_geometry.png`
- `outputs/pipeline/risk_ranking.png`
- `outputs/pipeline/model_metrics.png`
- `outputs/pipeline/top_pair_eci_trajectory.png`
- `outputs/pipeline/top_pair_altitude_evolution.png`
- `outputs/pipeline/top_pair_encounter_plane.png`
- `outputs/pipeline/top_pair_relative_ric.png`

Mevcut ciktilardan grafikleri yeniden uretmek icin:

```powershell
.\.venv\Scripts\python.exe src\generate_plots.py --outputs outputs\verification_pipeline
```

## PDF'e gore proje durumu

Bu repo artik PDF'teki cekirdek akisa hizalanmis durumda: LEO filtresi,
SGP4 tabanli zaman simulasyonu, TCA/minimum mesafe/bagil hiz ciktilari,
identified conjunction dataset'i, sabit esik karsilastirmasi ve ML raporu var.

Henuz gercek Pc (collision probability) hesaplanmiyor; bunun icin kovaryans,
nesne boyutu, hata elipsoidi veya daha ileri fizik tabanli model gerekir.

IAC uygunluk degerlendirmesi icin:

- `docs/iac_gap_analysis.md`
- `config/experiment_60_days.json`
