import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def run_drift(reference_path='monitoring/reference.csv',
              current_path='monitoring/current.csv',
              output_path='drift_report.html'):
    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(output_path)
    print(f"Drift report written to {output_path}")

if __name__ == "__main__":
    run_drift()
