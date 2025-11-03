#/bin/bash
set -euo pipefail

DATA_DIR="raw_dataset"
mkdir -p "${DATA_DIR}"

wget -O "${DATA_DIR}/names.zip" \
  "https://www.ssa.gov/oact/babynames/names.zip"

unzip -jqo "${DATA_DIR}/names.zip" "yob2010.txt" -d "${DATA_DIR}"

wget -q --show-progress -O "${DATA_DIR}/2010_surname.xlsx" \
  "https://www2.census.gov/topics/genealogy/2010surnames/Names_2010Census_Top1000.xlsx"

wget -q --show-progress -O "${DATA_DIR}/country-list.zip" \
  "https://download.geonames.org/export/dump/cities15000.zip"
unzip -jqo "${DATA_DIR}/country-list.zip" -d "${DATA_DIR}"

wget -q --show-progress -O "${DATA_DIR}/cwts_leiden_ranking_2024.xlsx" \
  "https://zenodo.org/records/12606083/files/CWTS%20Leiden%20Ranking%202024.xlsx?download=1"

wget -q --show-progress -O "${DATA_DIR}/cip2020.csv" \
  "https://nces.ed.gov/ipeds/cipcode/Files/CIPCode2020.csv"

wget -q --show-progress -O "${DATA_DIR}/largest_companies.zip" \
  "https://www.kaggle.com/api/v1/datasets/download/shiivvvaam/largest-companies-by-market-cap"
unzip -jqo "${DATA_DIR}/largest_companies.zip" "companiesmarketcap.csv" -d "${DATA_DIR}"