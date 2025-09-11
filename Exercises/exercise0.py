#Uppgift 1, a) Bräkna hypotenusan när a=3 och b=4
import math
a = 3 
b = 4
c = math.sqrt(a**2 + b**2)
print("a) Hypotenusan är:", c)

#Uppgift 1, b) Bräkna den andra när c=7 och a=5

import math
c = 7
a = 5
b = math.sqrt(c**2 - a**2)
print("b) Den kateten är:", round(b, 1))

#Uppgift 2, Bräkna accuracy
correct = 300
total = 365
accuracy = correct / total
print("Accuracy: ", round(accuracy, 2))

#Uppgift 3
TP = 2
FP = 2
FN = 11
TN = 985
accuracy = (TP + TN) / (TP + TN + FP + FN)
print("Noggrannhet för modellen ", round(accuracy, 3))
print("Detta är inte en bra modell, eftersome den missar många riktiga bränder")

#Uppgift 4
x1, y1 = 4, 4
x2, y2 = 0, 1
k = (y1 - y2) / (x1 - x2)
m = y1 - k * x1
print(f"Lutning k = {k}")
print(f"Konstantterm m = {m}")
print(f"Linjen är : y = {k}x + {m}")

#Uppgift 5
import math
x1, y1 = 3, 5
x2, y2 = -2, 4
distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
print("Avståndet mellan P1 och P2 är:", round(distance, 1))

#Uppgift 6
import math
x1, y1, z1 = 2, 1, 4
x2, y2, z2 = 3, 1, 0
distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
print("Avståndet mella P1 och P2 är:", round(distance, 2))

