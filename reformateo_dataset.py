import pandas as pd # Para manejar el csv
import json # para guardar la lista de enfermedades





#*********************************************************************************
#************************ AGRUPAR ************************************************
data = pd.read_csv('PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
data = data[data.Projection == "PA"]    # Me quedo solo con las filas cuya proyeccion sea PA
data = data[['Report', 'Labels', 'ImageID', "ImageDir"]]
data['File_name'] = data["ImageDir"].astype(str) + "/" + data["ImageID"]
data = data[['Report', 'Labels', 'File_name']]
data = data.dropna()




d_padre_hijos = {"normal":{"normal"}, "unchanged":{"unchanged"}, "chronic changes": {"chronic changes"}, "calcified densities": {"calcified densities", "calcified granuloma", "calcified adenopathy", "calcified mediastinal adenopathy", "calcified pleural thickening", "calcified pleural plaques", "heart valve calcified", "calcified fibroadenoma"}, "granuloma": {"granuloma", "calcified granuloma"}, "nodule":{"nodule", "multiple nodules"}, "pseudonodule":{"pseudonodule", "nipple shadow", "end on vessel"}, "fibrotic band":{"fibrotic band"}, "volume loss":{"volume loss"}, "hypoexpansion":{"hypoexpansion"}, "air trapping":{"air trapping"}, "bronchiectasis":{"bronchiectasis"}, "infiltrates":{"infiltrates", "interstitial pattern", "ground glass pattern", "reticular interstitial pattern", "reticulonodular interstitial pattern", "miliary opacities", "alveolar pattern", "consolidation", "air bronchogram", "air bronchogram"}, "increased density": {"increased density"}, "atelectasis":{"atelectasis", "total atelectasis","lobar atelectasis","segmental atelectasis","laminar atelectasis","round atelectasis","atelectasis basa"}, "pleural thickening":{"pleural thickening", "apical pleural thickening", "calcified pleural thickening"}, "pleural effusion":{"pleural effusion","loculated pleural effusion", "loculated fissural effusion", "hydropneumothorax", "empyema", "hemothorax"}, "costophrenic angle blunting":{"costophrenic angle blunting"}, "hilar enlargement":{"hilar enlargement", "adenopathy", "vascular hilar enlargement", "pulmonary artery enlargement"}, "hilar congestion":{"hilar congestion"}, "cardiomegaly":{"cardiomegaly"}, "aortic atheromatosis":{"aortic atheromatosis"}, "aortic elongation":{"aortic elongation", "descendent aortic elongation", "ascendent aortic elongation", "aortic button enlargement", "supra aortic elongation"}, "mediastinal enlargement":{"mediastinal enlargement", "superior mediastinal enlargement", "goiter", "supra aortic elongation ", "descendent aortic elongation", "ascendent aortic elongation", "aortic aneurysm", "mediastinal mass", "hiatal hernia"}, "mass":{"mass", "mediastinal mass", "breast mas", "pleural mass", "pulmonary mass", "soft tissue mass"}, "thoracic cage deformation":{"thoracic cage deformation", "scoliosis", "kyphosis", "pectum excavatum", "pectum carinatum", "cervical rib"}, "vertebral degenerative changes":{"vertebral degenerative changes", "vertebral compression", "vertebral anterior compression"}, "fracture":{"fracture", "clavicle fracture", "humeral fracture", "vertebral fracture", "rib fracture", "callus rib fracture"}, "hemidiaphragm elevation":{"hemidiaphragm elevation"}, "tracheostomy tube":{"tracheostomy tube"}, "endotracheal tube":{"endotracheal tube"}, "NSG tube":{"NSG tube"}, "catheter":{"catheter", "central venous catheter", "central venous catheter via subclavian vein", "central venous catheter via jugular vein", "reservoir central venous catheter", "central venous catheter via umbilical vein"}, "electrical device":{"electrical device", "dual chamber device", "single chamber device", "pacemaker", "dai"}, "artificial heart valve":{"artificial heart valve", "artificial mitral heart valve", "artificial aortic heart valve"}, "surgery":{"surgery", "metal", "osteosynthesis material", "sternotomy", "suture material", "bone cement", "prosthesis", "humeral prosthesis", "mammary prosthesis", "endoprosthesis", "aortic endoprosthesis", "surgery breast", "mastectomy", "surgery neck", "surgery lung", "surgery heart", "surgery humeral"}, "pneumonia":{"pneumonia", "atypical pneumonia"}, "pulmonary fibrosis":{"pulmonary fibrosis", "post radiotherapy changes", "asbestosis signs"}, "COPD signs":{"COPD signs"}, "heart insufficiency":{"heart insufficiency"}, "pulmonary edema":{"pulmonary edema"}, "obesity":{"obesity"}, "abscess":{"abscess"}, "cyst":{"cyst"}, "cavitation":{"cavitation"}, "bullas":{"bullas"}, "pneumothorax":{"pneumothorax", "hydropneumothorax"}, "pneumoperitoneo":{"pneumoperitoneo"}, "pneumomediastinum":{"pneumomediastinum"}, "subcutaneous emphysema":{"subcutaneous emphysema"}, "hyperinflated lung":{"hyperinflated lung"}, "flattened diaphragm":{"flattened diaphragm"}, "lung vascular paucity":{"lung vascular paucity"},  "bronchovascular markings":{"bronchovascular markings"},  "air fluid level":{"air fluid level"},  "mediastinal shift":{"mediastinal shift"}, "azygos lobe":{"azygos lobe"}, "fissure thickening":{"fissure thickening", "minor fissure thickening", "major fissure thickening", "loculated fissural effusion"}, "pleural plaques":{"pleural plaques", "calcified pleural plaques"}, "vascular redistribution":{"vascular redistribution", "central vascular redistribution"}, "pericardial effusion":{"pericardial effusion"}, "kerley lines":{"kerley lines"}, "dextrocardia":{"dextrocardia"}, "right sided aortic arch":{"right sided aortic arch"}, "tracheal shift":{"tracheal shift"}, "esophagic dilatation":{"esophagic dilatation"}, "azygoesophageal recess shift":{"azygoesophageal recess shift"}, "pericardial effusion":{"pericardial effusion"}, "mediastinic lipomatosis":{"mediastinic lipomatosis"}, "lytic bone lesion":{"lytic bone lesion"}, "sclerotic bone lesion":{"sclerotic bone lesion", "blastic bone lesion"}, "costochondral junction hypertrophy":{"costochondral junction hypertrophy"}, "sternoclavicular junction hypertrophy":{"sternoclavicular junction hypertrophy"}, "axial hyperostosis":{"axial hyperostosis"}, "osteopenia":{"osteopenia"}, "osteoporosis":{"osteoporosis"}, "non axial articular degenerative changes":{"non axial articular degenerative changes"}, "subacromial space narrowing":{"subacromial space narrowing"}, "gynecomastia":{"gynecomastia"}, "Chilaiditi sign":{"Chilaiditi sign"}, "diaphragmatic eventration":{"diaphragmatic eventration"}, "chest drain tube":{"chest drain tube"}, "ventriculoperitoneal drain tube":{"ventriculoperitoneal drain tube"}, "gastrostomy tube":{"gastrostomy tube"}, "nephrostomy tube":{"nephrostomy tube"}, "double J stent":{"double J stent"}, "abnormal foreign body":{"abnormal foreign body"}, "external foreign body":{"external foreign body"}, "tuberculosis":{"tuberculosis", "tuberculosis sequelae"}, "lung metastasis":{"lung metastasis"}, "lymphangitis carcinomatosa":{"lymphangitis carcinomatosa"}, "lepidic adenocarcinoma":{"lepidic adenocarcinoma"}, "emphysema":{"emphysema"}, "respiratory distress":{"respiratory distress"}, "pulmonary hypertension":{"pulmonary hypertension", "pulmonary artery hypertension", "pulmonary venous hypertension"}, "bone metastasis":{"bone metastasis"}}


set_padres = set(d_padre_hijos.keys())


'''#********* Comprobar que no hay hijos como padre en otros grupos *******
for padre in d_padre_hijos.keys():
    if padre in d_padre_hijos[padre]:
       d_padre_hijos[padre].remove(padre)'''


    
set_hijos = set()
for padre in d_padre_hijos.keys():
    set_hijos = set_hijos.union(d_padre_hijos[padre])


for padre in d_padre_hijos.keys():
    d_padre_hijos[padre].add(padre)


lista_hijos = list(set_hijos) # lista auxiliar para crear el diccionario de hijos a set_padres
d_hijo_padres = {}
for hijo in lista_hijos:
    d_hijo_padres[hijo] = set()
    for padre in d_padre_hijos:
        if hijo in d_padre_hijos[padre]:
            d_hijo_padres[hijo].add(padre)
    
    
# print("Labels original: ", row["Labels"], "\n. Labels: " , fila, ". len:", len(fila))    

borrar_indices = []
for index, row in data.iterrows():
    fila = row["Labels"]
    if isinstance(fila,str):
        borrar = True
        etiquetas = fila[1:-1].replace("'","").split(",")
        #nuevo = set()
        for enfermedad in etiquetas:
            if len(enfermedad) > 1:
                #print("Labels: " , fila, "enfermedad:", enfermedad, ". len:", len(enfermedad))
                if enfermedad[0] == " ":
                    enfermedad = enfermedad[1:]
                if enfermedad[-1] == " ":
                    enfermedad = enfermedad[:-1]
                if enfermedad in set_hijos or enfermedad == "normal" or enfermedad == "unchanged":
                    #nuevo = nuevo.union(d_hijo_padres[enfermedad])
                    borrar = False
        if borrar:
            borrar_indices.append(index)
        
    else:
        borrar_indices.append(index)
        
d_frecuencias_5zonas = {"Pulmon": 0, "Mediastino e hilios pulmonares": 0, "Pleura y diafragma y pared abdominal": 0,
"Calcificacion": 0, "Cuerpos extranos": 0, "Patologica": 0, "Unchanged": 0}



for index, row in data.iterrows():
    if row["Pulmon"] == 1:
        d_frecuencias_5zonas["Pulmon"] += 1
    if row["Mediastino e hilios pulmonares"] == 1:
        d_frecuencias_5zonas["Mediastino e hilios pulmonares"] += 1
    if row["Pleura y diafragma y pared abdominal"] == 1:
        d_frecuencias_5zonas["Pleura y diafragma y pared abdominal"] += 1
    if row["Calcificacion"] == 1:
        d_frecuencias_5zonas["Calcificacion"] += 1
    if row["Cuerpos extranos"] == 1:
        d_frecuencias_5zonas["Cuerpos extranos"] += 1
    if row["Patologica"] == 1:
        d_frecuencias_5zonas["Patologica"] += 1
    if row["Unchanged"] == 1:
        d_frecuencias_5zonas["Unchanged"] += 1
    

data.drop(borrar_indices, inplace = True)
                
'''##  ahora vot a ver que etiquetas se han borrado
data_original = pd.read_csv('PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
data_original = data_original[data_original.Projection == "PA"]    # Me quedo solo con las filas cuya proyeccion sea PA
data_original = data_original[['Report', 'Labels', 'ImageID']]
data_original = data_original.dropna()
borrados = {}
for cadena in data_original["Labels"][borrar_indices]:
    etiquetas = cadena[1:-1].replace("'","").split(",")
    for enfermedad in etiquetas:
            if len(enfermedad) > 1:
                if enfermedad[0] == " ":
                    enfermedad = enfermedad[1:]
                if enfermedad[-1] == " ":
                    enfermedad = enfermedad[:-1]
                if enfermedad in borrados.keys():
                    borrados[enfermedad] += 1
                else:
                    borrados[enfermedad] = 1'''
                
d_frecuencias = {}
for row in data["Labels"]:
    for enfermedad in row:
        if enfermedad in d_frecuencias.keys():
            d_frecuencias[enfermedad] += 1
        else:
            d_frecuencias[enfermedad] = 1

d_frecuencias = {k: v for k, v in sorted(d_frecuencias.items(), key=lambda item: item[1])}        


                        
                        
            
            
            
#************************************************************************
#********************* ETIQUETAS BINARIAS *******************************
                        
# Creo una lista con todas las etiquetas
# Ordeno el diccionario de frecuencias por valor
# Creo una nueva columna en el dataset para almacenar 1 y 0 
# Creo una lista para almacenar filas inservibles que hay que borrar
lista_enfermedades = list(d_frecuencias.keys())
lista_enfermedades = lista_enfermedades[65:]
conjunto_enfermedades = set(lista_enfermedades)
num_enfermedades = len(lista_enfermedades)
data["Etiqueta_numerica"] = data["Report"] # creo una nueva columna
borrar_indices = []

# En este bucle voy a guardar las etiquetas en una lista de 0s y 1s y voy 
# a borrar aquellas filas que no tengan ninguna de nuestras 29 enfermedades
for index, row in data.iterrows():
    nuevo = list()
    frase = row["Labels"]
    borrar = True
    for enfermedad in frase:
        if enfermedad in conjunto_enfermedades:
            nuevo.append(lista_enfermedades.index(enfermedad))
            borrar = False
    if borrar:
        borrar_indices.append(index)
    else:
        data.at[index, "Etiqueta_numerica"] = nuevo
            
    
                 
print("Datos antes del borrado: ", len(data))
data.drop(borrar_indices, inplace = True)
print("Datos despues del borrado: ", len(data))

# Finalmente, guardo el dataframe que he obtenido
data.to_csv("/home/murat/padchest_formateado_compacto.csv")

# Tambien necesito guardar la lista de enfermedades para poder traducir de 0 y 1 a etiquetas
json.dump(lista_enfermedades, open("/home/murat/lista_enfermedades.json", 'w'))

# Para guardar el diccionario de padre a hijos, los valores no pueden ser set 
for key in d_padre_hijos.keys():
    d_padre_hijos[key] = list(d_padre_hijos[key])
    
json.dump(d_padre_hijos, open("/home/murat/d_padre_hijos.json", 'w'))




