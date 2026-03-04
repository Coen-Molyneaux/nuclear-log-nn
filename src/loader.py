import csv
import numpy as np

class DocumentLoader:
    def __init__(self, csvfile: str):
        self.csvfile = csvfile

    def load(self):
        X = []
        y = []

        with open(self.csvfile, newline='') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    features = [
                        float(row["GR"]),
                        float(row["SP"]),
                        float(row["RILD"]),
                        float(row["RLL3"]),
                        float(row["MI"]),
                        float(row["MN"]),
                        float(row["CNLS"]),
                        float(row["MCAL"]),
                        float(row["DCAL"]),
                    ]

                    target = float(row["RHOB"])

                    X.append(features)
                    y.append(target)

                except (ValueError, TypeError):
                    continue

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)