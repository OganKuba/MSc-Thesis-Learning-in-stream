# Reprodukcja projektu na nowym komputerze

Ten plik opisuje, co trzeba zainstalowac, pobrac i gdzie polozyc dane, zeby po sklonowaniu repo odtworzyc preprocessing, eksperymenty i analize wynikow.

Zakladana struktura po sklonowaniu:

```text
MSc-Thesis-Learning-in-stream/
├── analysis/
├── preproccessing/
└── stream/
```

## 1. Wymagania systemowe

Zainstaluj:

- Git
- Python 3.12 lub nowszy
- Java JDK 17
- Maven 3.x

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip openjdk-17-jdk maven
```

Sprawdzenie wersji:

```bash
git --version
python3 --version
java -version
mvn -version
```

## 2. Klonowanie repo

```bash
git clone <URL_DO_REPO> MSc-Thesis-Learning-in-stream
cd MSc-Thesis-Learning-in-stream
```

Po wejscu do katalogu powinienes widziec m.in.:

```bash
ls
```

```text
analysis  preproccessing  stream
```

## 3. Srodowisko Pythona dla preprocessingu

```bash
cd preproccessing
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

## 4. Srodowisko Pythona dla analizy wynikow

```bash
cd analysis
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

## 5. Pobranie danych wejsciowych

Duzych danych nie trzymamy w git. Trzeba je odtworzyc lokalnie.

### 5.1 Yahoo Finance

Skrypt pobiera dane przez `yfinance` dla tickerow z `preproccessing/config.py`.

```bash
cd preproccessing
source .venv/bin/activate
python -m yahoo_finance.download
```

Wynik powinien trafic do:

```text
preproccessing/data/raw/yahoo/
```

Najwazniejsze pliki:

```text
preproccessing/data/raw/yahoo/all_tickers.csv
preproccessing/data/raw/yahoo/AAPL.csv
...
```

### 5.2 NYC Taxi

Dane sa pobierane z oficjalnego NYC TLC trip data:

```text
https://d37ci6vzurychx.cloudfront.net/trip-data
```

Skrypt pobiera pliki `yellow_tripdata_YYYY-MM.parquet` dla lat 2022-2024.

```bash
cd preproccessing
source .venv/bin/activate
python -m nyc_taxi.download
```

Wynik powinien trafic do:

```text
preproccessing/data/raw/nyc_taxi/
```

Przykladowe pliki:

```text
preproccessing/data/raw/nyc_taxi/yellow_tripdata_2022-01.parquet
preproccessing/data/raw/nyc_taxi/yellow_tripdata_2024-12.parquet
```

### 5.3 NHTS

Dane NHTS trzeba pobrac recznie ze strony National Household Travel Survey:

```text
https://nhts.ornl.gov/
```

Potrzebne edycje:

- 2009
- 2017
- 2022

Po pobraniu rozpakuj pliki tak, zeby powstala taka struktura:

```text
preproccessing/data/raw/nhts/2009/
preproccessing/data/raw/nhts/2017/
preproccessing/data/raw/nhts/2022/
```

Minimalnie wymagane pliki:

```text
preproccessing/data/raw/nhts/2009/DAYV2PUB.csv
preproccessing/data/raw/nhts/2009/PERV2PUB.csv
preproccessing/data/raw/nhts/2009/HHV2PUB.csv

preproccessing/data/raw/nhts/2017/trippub.csv
preproccessing/data/raw/nhts/2017/perpub.csv
preproccessing/data/raw/nhts/2017/hhpub.csv

preproccessing/data/raw/nhts/2022/tripv2pub.csv
preproccessing/data/raw/nhts/2022/perv2pub.csv
preproccessing/data/raw/nhts/2022/hhv2pub.csv
```

Wielkosc liter w nazwach plikow nie powinna przeszkadzac, bo loader szuka nazw case-insensitive.

## 6. Preprocessing i generowanie ARFF

Uruchom z katalogu `preproccessing`.

```bash
cd preproccessing
source .venv/bin/activate
```

### 6.1 Yahoo Finance

```bash
python -m yahoo_finance.features
python -m yahoo_finance.build_stream
```

Powinny powstac:

```text
preproccessing/data/processed/yahoo_features.csv
preproccessing/data/arff/yahoo_finance.arff
```

### 6.2 NYC Taxi

```bash
python -m nyc_taxi.aggregate
python -m nyc_taxi.features
python -m nyc_taxi.build_stream
```

Powinny powstac:

```text
preproccessing/data/processed/nyc_taxi_aggregated.csv
preproccessing/data/processed/nyc_taxi_features.csv
preproccessing/data/arff/nyc_taxi.arff
```

### 6.3 NHTS

```bash
python -m nhts.load
python -m nhts.features
python -m nhts.build_stream
```

Powinny powstac:

```text
preproccessing/data/processed/nhts_joined.csv
preproccessing/data/processed/nhts_features.csv
preproccessing/data/arff/nhts.arff
```

Wroc do katalogu glownego repo:

```bash
cd ..
```

## 7. Zaleznosci Javy/Maven

```bash
cd stream
mvn dependency:resolve
mvn dependency:get -Dartifact=com.github.fracpete:sizeofag:1.1.0
```

Projekt uzywa:

- Java 17
- MOA `2024.07.0`
- Jackson
- Commons Math
- Lombok
- `sizeofag` jako `-javaagent` do pomiaru rozmiaru modeli

## 8. Poprawienie sciezek do ARFF po sklonowaniu

Plik:

```text
stream/src/main/java/thesis/experiments/master_experiments.json
```

zawiera sciezki do plikow ARFF. Po sklonowaniu na innym komputerze ustaw je na lokalne sciezki.

Najprosciej z katalogu glownego repo:

```bash
ROOT="$(pwd)"
python3 - <<PY
from pathlib import Path

root = Path("$ROOT").resolve()
cfg = root / "stream/src/main/java/thesis/experiments/master_experiments.json"
text = cfg.read_text()

text = text.replace(
    "/home/kubog/MSc-Thesis-Learning-in-stream/preproccessing/yahoo_finance/data/arff/yahoo_finance.arff",
    str(root / "preproccessing/data/arff/yahoo_finance.arff"),
)
text = text.replace(
    "/home/kubog/MSc-Thesis-Learning-in-stream/preproccessing/nyc_taxi/data/arff/nyc_taxi.arff",
    str(root / "preproccessing/data/arff/nyc_taxi.arff"),
)
text = text.replace(
    "/home/kubog/MSc-Thesis-Learning-in-stream/preproccessing/data/arff/nhts.arff",
    str(root / "preproccessing/data/arff/nhts.arff"),
)

cfg.write_text(text)
print("Updated", cfg)
PY
```

Sprawdzenie:

```bash
rg "path" stream/src/main/java/thesis/experiments/master_experiments.json
```

## 9. Uruchomienie eksperymentow

Z katalogu `stream`:

```bash
cd stream
```

Skrypt `run_experiments.sh` domyslnie mial lokalna sciezke do JDK z jednego komputera. Na nowym komputerze ustaw `JAVA_HOME` na swoj JDK 17.

Ubuntu/Debian najczesciej:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

Potem:

```bash
bash run_experiments.sh
```

Wyniki powinny powstac w:

```text
stream/results/
```

Najwazniejsze pliki:

```text
stream/results/runs_raw.csv
stream/results/master_summary.csv
stream/results/E1/
stream/results/E2/
stream/results/E3/
stream/results/E4/
stream/results/E5/
```

Uwaga: pelny run moze trwac dlugo i nadpisuje `stream/results/`.

## 10. Generowanie tabel i wykresow

Po eksperymentach uruchom pipeline analityczny:

```bash
cd ..
source analysis/.venv/bin/activate
python -m analysis
```

Wyniki analizy trafia do:

```text
stream/results/figures/
stream/results/tables/
```

## 11. Szybka kontrola po wszystkim

Z katalogu glownego repo:

```bash
test -f preproccessing/data/arff/yahoo_finance.arff && echo "Yahoo ARFF OK"
test -f preproccessing/data/arff/nyc_taxi.arff && echo "NYC ARFF OK"
test -f preproccessing/data/arff/nhts.arff && echo "NHTS ARFF OK"
test -f stream/results/master_summary.csv && echo "Experiments OK"
test -d stream/results/figures && echo "Figures OK"
test -d stream/results/tables && echo "Tables OK"
```

## 12. Czego nie commitowac

Nie commituj:

- `stream/results/`
- `stream/target/`
- `preproccessing/**/data/`
- `analysis/.venv/`
- `preproccessing/.venv/`
- `__pycache__/`
- `.idea/`

Te rzeczy sa generowane lokalnie albo sa za duze do zwyklego gita.
