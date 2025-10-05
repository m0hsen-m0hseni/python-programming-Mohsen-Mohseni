# Lab 3 – Linjär klassificering
#
# Steg:
# 1) Läs in unlabelled_data.csv (två kolumner x,y utan rubrik)
# 2) Välj en linje y = kx + m (default: k=0, m = median(y - kx))
# 3) Märk punkter: label = 0 om y < kx+m, annars 1
# 4) Skriv labelled_data.csv och rita figuren lab3_plot.png
# 5) (VG) jämför med tre givna linjer f,g,h och skriv ut procentuell överensstämmelse
#
# Körning:
# python lab3.py

import argparse
from pathlib import Path
import csv 
import sys
import numpy as np
import matplotlib.pyplot as plt


def las_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        sys.exit("Hittar inte 'unlabelled_data.csv' i den här mappen.")
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        if data.size != 2:
            sys.exit("Fel format: varje rad måste ha exakt två värden (x,y).")
        data = data.reshape(1, -1)
    X = data[:, 0].astype(float)
    Y = data[:, 1].astype(float)
    return X, Y


def klassificera_punkt(x: float, y: float, k: float, m: float) -> int:
    # 0 = under linjen, 1 = på/över linjen
    return 0 if y < (k * x + m) else 1


def klassificera_alla(X: np.ndarray, Y: np.ndarray, k: float, m: float) -> np.ndarray:
    return np.array([klassificera_punkt(x, y, k, m) for x, y in zip(X, Y)], dtype=int)


def spara_labelled_csv(path: Path, X: np.ndarray, Y: np.ndarray, labels: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "label"])
        for x, y, lbl in zip(X, Y, labels):
            w.writerow([x, y, int(lbl)])


def plotta(X: np.ndarray, Y: np.ndarray, labels: np.ndarray, k: float, m: float, out_png: Path) -> None:
    plt.figure()
    plt.scatter(X[labels == 0], Y[labels == 0], marker="o", label="klass 0 (under)")
    plt.scatter(X[labels == 1], Y[labels == 1], marker="x", label="klass 1 (över)")
    xs = np.linspace(X.min() - 0.5, X.max() + 0.5, 200)
    ys = k * xs + m
    plt.plot(xs, ys, label=f"y = {k:.3f}x + {m:.3f}")
    plt.title("Lab 3 – punkter och linje")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Lab 3 – Linjär klassificering")
    parser.add_argument("--k", type=float, default=None, help="Lutning k för linjen y=kx+m (default 0.0)")
    parser.add_argument("--m", type=float, default=None, help="Intercept m (default median(y - kx))")
    args = parser.parse_args()

    data_file = Path("unlabelled_data.csv")
    X, Y = las_data(data_file)

    # välj k och m
    k = 0.0 if args.k is None else float(args.k)
    m = float(np.median(Y - k * X)) if args.m is None else float(args.m)

    # klassificera punkter
    labels = klassificera_alla(X, Y, k, m)

    # spara utdata
    out_csv = Path("labelled_data.csv")
    out_png = Path("lab3_plot.png")
    spara_labelled_csv(out_csv, X, Y, labels)
    plotta(X, Y, labels, k, m, out_png)

    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    p1 = labels.mean() * 100.0

    print(f"Skrev '{out_csv}' ({len(labels)} rader).")
    print(f"Sparade figur till '{out_png}'.")
    print(f"Din linje: y = {k:.6g}x + {m:.6g}")
    print(f"klass0: {n0} klass1: {n1} andel klass1: {p1:.1f}%")

    # --- VG-del: jämför med tre givna linjer ---
    trio = [
        (0.489, 0.0, "f(x)=0.489x"),
        (-2.0, 0.16, "g(x)=-2x+0.16"),
        (800.0, -120.0, "h(x)=800x-120"),
    ]

    def jamfor_linje(kk: float, mm: float, name: str) -> float:
        lbls = klassificera_alla(X, Y, kk, mm)
        same = np.mean(lbls == labels) * 100.0
        return same

    print("\n— VG-jämförelse mot tre givna linjer —")
    for kk, mm, name in trio:
        same = jamfor_linje(kk, mm, name)
        print(f"{name:<18} överensstämmer {same:5.1f}% med din linje (y={k:.3f}x+{m:.3f}).")


if __name__ == "__main__":
    main()