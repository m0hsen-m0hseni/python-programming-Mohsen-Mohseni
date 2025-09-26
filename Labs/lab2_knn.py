"""
Lab 2 - Pichu vs Pikachu (k-NN)
Mohsen Mohseni

Vad progammet gör:
- Läser in datapoints.txt (bredd, höjd, label: 0=pichu, 1=Pikachu)
- Läser in testpoints.txt (rader som: "1. (25, 32)")
- Plottar träningsdata + testpunkter
- Klassificerar med k=1 och k=10 (k-NN)
- Tar in användarinput (med felhantering)
- Delar upp data i train/test, räknar accuracy och confusion matrix
- Upprepar flera gånger och ritar en graf på accuracy
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple
import math
import re
import random
import matplotlib.pyplot as plt

# ---------- Datastrukturer ----------
@dataclass(frozen=True)
class Point:
    w: float                           # bredd (cm)
    h: float                           # höjd (cm)

@dataclass(frozen=True)
class LabeledPoint(Point):
    label: int

# ---------- Filinläsning ----------
def load_training_data(path: str) -> List[LabeledPoint]:
    """
    Läser in 'datapoints.txt'.
    Första raden är header: width, height, label
    Därefter rader i format: <float>, <float>, <int>
    """
    data: List[LabeledPoint] = []
    with open(path, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            s = line.strip()
            if not s:
                continue
            if header:
                header = False
                continue
            parts = [p.strip() for p in s.split(",")]
            w, h, lbl = float(parts[0]), float(parts[1]), int(parts[2])
            data.append(LabeledPoint(w, h, lbl))
    return data


def load_test_points(path: str) -> List[Point]:
    """
    Läser in 'testpoints.txt'.
    Plockar ut alla par tal inom parentes, t.ex. '(25, 32)' även om före.
    """
    pts: List[Point] = []
    pat = re.compile(r"\(?\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)?")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                pts.append(Point(float(m.group(1)), float(m.group(2))))
    return pts

# ---------- Avstånd ----------
def euclidean(a: Point, b: Point) -> float:
    """Euklidiskt avstånd mellan två punkter."""
    dx = a.w - b.w
    dy = a.h - b.h
    return math.sqrt(dx*dx + dy*dy)

# ---------- k-NN ----------
def knn_classify(x: Point,
                 training: Sequence[LabeledPoint],
                 k: int = 1,
                 debug: bool = False) -> int:
    """Returnerar labeln (0/1) för x med k närmaste grannar."""
    # Lista av (avstånd, punktobjekt):
    dists: List[Tuple[float, LabeledPoint]] = [(euclidean(x, p), p) for p in training]
    dists.sort(key=lambda t: t[0]) # stigande
    k_neigh = dists[:k] # k närmaste
    labels = [p.label for _, p in k_neigh]

    if debug:
        print(f"[Debug k-nearst labels]: {labels}")
        for d, p in k_neigh:
            print(f" d={d:.3f} (w={p.w:.1f}, h={p.h:.1f}) lbl={p.label}")

    # Majoritetsröstning + tie-break med allra närmaste
    count1 = sum(1 for lbl in labels if lbl == 1) # Pikachu
    count0 = k - count1 # Pichu
    if count1 > count0:
        return 1
    if count0 > count1:
        return 0
    return labels[0] # lika: närmaste bestämmer


# ---------- Plotta ----------
def plot_data(training: Sequence[LabeledPoint],
              test: Optional[Sequence[Point]] = None,
              title: str = "Träningsdata + testpunkter") -> None:
    xs0 = [p.w for p in training if p.label == 0]
    ys0 = [p.h for p in training if p.label == 0]
    xs1 = [p.w for p in training if p.label == 1]
    ys1 = [p.h for p in training if p.label == 1]

    plt.figure()
    plt.scatter(xs0, ys0, label="Pichu (0)")
    plt.scatter(xs1, ys1, label="Pikachu (1)")
    if test:
        tx = [p.w for p in test]
        ty = [p.h for p in test]
    plt.scatter(tx, ty, marker="x", s=80, label="Testpunkter")
    plt.title(title)
    plt.xlabel("Bredd (cm)")
    plt.ylabel("Höjd (cm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

# ---------- Användarinmatning ----------
def read_user_point() -> Optional[Point]:
    """Frågar användaren om en punkt. Enter på bredd → avbryt."""
    s = input("\nAnge bredd (cm) eller tryck Enter för att avbryta: ").strip()
    if not s:
        return None
    try:
        w = float(s)
        h = float(input("Ange höjd (cm): ").strip())
        if w <= 0 or h <= 0:
            print("Fel: bredd/höjd måste vara > 0.")
            return None
        return Point(w, h)
    except ValueError:
        print("Felaktig inmatning: använd siffror, t.ex. 24.5")
        return None

# ---------- Confusion Matrix ----------
@dataclass
class Confusion:
    """Pikachu = positiv klass, Pichu = negativ."""
    TP: int = 0 # Pikachu rätt
    TN: int = 0 # Pichu rätt
    FP: int = 0 # Pichu -> fel som Pikachu
    FN: int = 0 # Pikachu -> fel som Pichu

    def update(self, y_true: int, y_pred: int) -> None:
        if y_true == 1 and y_pred == 1: self.TP += 1
        elif y_true == 1 and y_pred == 0: self.FN += 1
        elif y_true == 0 and y_pred == 1: self.FP += 1
        else: self.TN += 1

    def accuracy(self) -> float:
        tot = self.TP + self.TN + self.FP + self.FN
        return (self.TP + self.TN) / tot if tot else 0.0

    def __str__(self) -> str:
        return ( "Confusion matrix (Pikachu positiv, Pichu negativ)\n"
                 " Pikachu Pichu\n"
                 f"Pred = 1 -> TP={self.TP:3d} FP={self.FP:3d}\n"
                 f"Pred = 0 -> FN={self.FN:3d} TN={self.TN:3d}\n"
                 f"Accuracy: {self.accuracy():.3f}" )

# ---------- Split & utvärdering ----------
def split_train_test(data: Sequence[LabeledPoint],
                     n_train_per_class: int = 50,
                     n_test_per_class: int = 25,
                     seed: Optional[int] = 42) -> Tuple[List[LabeledPoint], List[LabeledPoint]]:
    """Delar data klassvis i train/test med fasta kvoter."""
    if seed is not None:
        random.seed(seed)
    pichu = [p for p in data if p.label == 0]
    pika = [p for p in data if p.label == 1]
    random.shuffle(pichu)
    random.shuffle(pika)

    n_train0 = min(n_train_per_class, len(pichu))
    n_train1 = min(n_train_per_class, len(pika))
    n_test0 = min(n_test_per_class, len(pichu) - n_train0)
    n_test1 = min(n_test_per_class, len(pika) - n_train1)

    train = pichu[:n_train0] + pika[:n_train1]
    test = pichu[n_train0:n_train0+n_test0] + pika[n_train1:n_train1+n_test1]
    random.shuffle(train)
    random.shuffle(test)
    print(f"[SPLIT] Pichu: total={len(pichu)}, train={n_train0}, test={n_test0} | "
          f"Pikachu: total={len(pika)}, train={n_train1}, test={n_test1}")
    return train, test


def evaluate_knn(data: Sequence[LabeledPoint],
                 k: int,
                 repeats: int = 4,
                 seed: int = 42) -> float:
    """
    Kör flera train/test-splitar, skriver ut accuracy + confusion matrix,
    samt ritar en liten graf över accuracy per körning.
    """
    accs: List[float] = []
    for i in range(repeats):
        train, test = split_train_test(data, seed=seed+i)
        cf = Confusion()
        for p in test:
            pred = knn_classify(Point(p.w, p.h), train, k)
            cf.update(p.label, pred)
        print(f"\n[Körning {i+1}] k={k} -> accuracy={cf.accuracy():.3f}")
        print(cf)
        accs.append(cf.accuracy())

    mean_acc = sum(accs)/len(accs)
    
    plt.figure()
    plt.plot(range(1, repeats+1), accs, marker="o")
    plt.title(f"Accuracy över körning (k={k}) - medel={mean_acc:.3f}")
    plt.xlabel("Körning")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show(block=False)
    return mean_acc

# ---------- Huvudprogram ----------
def main() -> None:
    label_names = {0: "Pichu", 1: "Pikachu"}

    # 1) Läs in data
    training_all = load_training_data("datapoints.txt")
    test_points = load_test_points("testpoints.txt")

    # valfri snabb kontroll/debug
    print("\n[DEBUG] check (25, 32) med k=10")
    _ = knn_classify(Point(25, 32), training_all, k=10, debug=True)

    # 2) Rita data
    plot_data(training_all, test_points, title="Träningsdata + testpunkter")

    # 3) Klassificera givna testpunkter med k=1
    print("\n== 1-NN för givna testpunkter ==")
    for tp in test_points:
        lbl = knn_classify(tp, training_all, k=1)
        print(f"Sample med (bredd, höjd): ({tp.w:.1f}, {tp.h:.1f}) klassad som {label_names[lbl]}")

    # 4) Klassificera samma testpunkter med k=10 (krav uppgiftens del b)
    print("\n== 10-NN för givna testpunkter ==")
    for tp in test_points:
        lbl = knn_classify(tp, training_all, k=10)
        print(f"Sample med (bredd, höjd): ({tp.w:.1f}, {tp.h:.1f}) klassad som {label_names[lbl]}")

    # 5) Användarinmatning + felhantering (frivilligt men bra för VG)
    up = read_user_point()
    if up:
        print("\n== Användarinmatning: klassificering för k=1..10 ==")
        for k in range(1, 11):
            lbl = knn_classify(up, training_all, k=k)
            print(f"k={k:2d} -> Din punkt ({up.w:.1f}, {up.h:.1f}) klassad som {label_names[lbl]}")

    # 6) Bonus: upprepa train/test-split och beräkna accuracy + confusion matrix
    print("\n== Bonus: accuracy över flera körningar ==")
    for k in (3,):
        mean_acc = evaluate_knn(training_all, k=k, repeats=4, seed=42)
        print(f"Medelaccuracy för k={k}: {mean_acc:.3f}")

    print("\nKlart. Stäng figurerna för att avsluta.")
    plt.show()


if __name__ == "__main__":
    main()



