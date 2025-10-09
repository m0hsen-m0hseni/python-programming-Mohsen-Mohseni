# Lab 3 - Linjär klassificering

Detta repo inehåller lösningen till **Lab3 i programmering 

##Programebeskrivning
Programmet ('lab3.py') läser in data från 'unlabelled_data.csv'.
Det ritar en linje enligt formen 'y = kx + m' och klassificerar punkterna:
- **label = 0** om punkten ligger under linjen
- **label = 1** om punkten ligger på/över linje

Resultatet sparas i:
- 'labelled_data.csv' (punkterna med etiketter)
- 'lab3_plot.png' (en figur med punkterna och linje)

## För VG
I filen 'report_lab3.ipynb' finns en kort rapport:
- visning av data och min linje
- jämförelse med de tre givna linjerna
- diskussion kring resultatet

## Hur man kör programmet
1. Se till att du har Pyhton installerat samt biblioteken 'numpy' och 'matplotlib':
   '''bash 
   pip install numpy matplotlib
2. Kör programmet
   python lab3.py
