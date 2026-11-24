# Dataset setup

This project uses the real **US Accidents** dataset by Sobhan Moosavi and colleagues:

<https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents>

The project does not fabricate data or automatically bypass Kaggle authentication.
The current working copy contains `accidents.csv` and its cached 100,000-row sample.
The large source CSV is normally kept local rather than included when sharing the
project because it is large and distributed through Kaggle.

## Manual download (recommended for beginners)

1. Sign in to Kaggle and open the dataset page above.
2. Select **Download** and extract the archive.
3. Copy the CSV (commonly named `US_Accidents_March23.csv`) into this directory.
4. Rename it to `accidents.csv`.

The expected location is `data/accidents.csv`. The loader also recognizes the common
`US_Accidents_March23.csv` name. The CSV should contain a `Severity` column.

## Optional Kaggle API download

This requires a Kaggle account and API credentials:

1. In Kaggle, open **Account → API → Create New Token** to download `kaggle.json`.
2. Put the credentials at `%USERPROFILE%\.kaggle\kaggle.json`. Keep the file private;
   never commit or share it.
3. In the project virtual environment, install the optional CLI:

   ```powershell
   python -m pip install kaggle
   ```

4. From the project root, download and extract:

   ```powershell
   kaggle datasets download -d sobhanmoosavi/us-accidents -p data --unzip
   ```

5. Rename the extracted CSV to `accidents.csv` if needed.

The optional `kaggle` package is not required by the project itself. Follow Kaggle's
current terms and dataset license when using or redistributing the data.

## Memory-friendly loading

The analysis and dashboard default to a reproducible, uniformly sampled subset of
up to 100,000 rows. The CSV is read in chunks to avoid loading the entire dataset
into memory. To explicitly load all rows in your own script, call:

```python
from src.data_loader import load_dataset

df = load_dataset(max_rows=None)
```

Full-data loading may require several gigabytes of memory. The actual available
columns are inspected at runtime; optional fields such as speed limit are not
assumed to exist.
