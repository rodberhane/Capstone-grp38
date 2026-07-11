"""Run the whole pipeline in order and report pass/fail. See 00_START_HERE.md.

    python run_all.py

All six scripts are config-driven: they read data paths from config.py (set DATA_DIR there)
and write outputs under artifacts/. Clone the repo, download the data into data/ per
docs/01_data_sources.md, and this runs unchanged on any machine.
"""
import subprocess, sys, pathlib, time

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable
STEPS = [
    ("src/harmonisation.py",        "DATA: harmonise BER + build the reference-curve family"),
    ("src/vulnerability_model.py",  "VULNERABILITY: trained depth-damage model (Ch.3 staged architecture)"),
    ("src/exposure_classifier.py",  "EXPOSURE: dwelling-type classifier, method comparison"),
    ("src/hazard_coastal.py",       "HAZARD: coastal flood from terrain, method comparison"),
    ("src/hazard_fluvial.py",       "HAZARD: fluvial flood + distance-to-river (needs internet)"),
    ("src/deployment_dublin.py",    "DEPLOY: apply curve to Dublin, euro band"),
]
results = []
for i, (script, purpose) in enumerate(STEPS, 1):
    print("\n" + "=" * 78 + f"\n[{i}/{len(STEPS)}] {script}\n      {purpose}\n" + "=" * 78)
    t0 = time.time()
    rc = subprocess.run([PY, str(HERE / script)]).returncode
    results.append((script, "OK" if rc == 0 else f"FAILED ({rc})", time.time() - t0))
print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
for s, st, dt in results:
    print(f"  {st:12} {s:32} {dt:5.1f}s")
print(f"\n{sum(1 for _,st,_ in results if st=='OK')}/{len(results)} passed. Outputs in artifacts/.")
