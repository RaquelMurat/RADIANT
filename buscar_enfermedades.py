import pandas as pd # Para manejar el csv

data = pd.read_csv('PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
data = data[data.Projection == "PA"]    # Me quedo solo con las filas cuya proyeccion sea PA
data = data[['Report', 'Labels', 'ImageID']]
data = data.dropna()

patron = "unchanged"
xounter = 0
for row in data["Labels"]:
    if isinstance(row,str):
        etiquetas = row[1:-1].replace("'","").split(",")
        for enfermedad in etiquetas:
            if len(enfermedad) > 1:
                if enfermedad[0] == " ":
                    enfermedad = enfermedad[1:]
                if enfermedad[-1] == " ":
                    enfermedad = enfermedad[:-1]
                if enfermedad == patron:
                    xounter += 1

print(patron, ": ", xounter)
