import subprocess
from pathlib import Path
from .utils import timerPrint, get_file_path_str

def import_csv(csv_file: str | Path, sqlite_db_file: str | Path, table_name: str):
  # import a csv file using cli
  # expects first line to contain header which will be skipped
  # https://stackoverflow.com/a/59671652/359001
  cleaned_csv_file_str = get_file_path_str(csv_file).replace('\\','\\\\')
  # timerPrint(f'Importing: "{cleaned_csv_file}" ...')
  # `skip 1` to skip header line in CSV file
  result = subprocess.run(['sqlite3',
                         get_file_path_str(sqlite_db_file),
                         '-cmd',
                         '.mode csv',
                         f'.import --skip 1 "{cleaned_csv_file_str}" {table_name}'],
                        capture_output=True)

  if result.returncode != 0:
    timerPrint(f'Got non-zero code for this file: {cleaned_csv_file_str}')

