import csv
from pathlib import Path


class ExperimentLogger:

    def __init__(self, log_path):

        self.log_path = Path(log_path)

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def log(self, row):

        file_exists = (
            self.log_path.exists()
            and self.log_path.stat().st_size > 0
        )

        with open(
            self.log_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(row.keys())
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)