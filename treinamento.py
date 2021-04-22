import cv2
import os
import numpy as np

eigenFace = cv2.face.EigenFaceRecognizer_create(num_components=50)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    # print(len(caminhos), caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Face", imagemFace)
        # cv2.waitKey(500)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)
    return np.array(ids), faces


ids, faces = getImagemComId()

eigenFace.train(faces, ids)
eigenFace.write('classificadorEigen.yml')

# fisherface.train(faces, ids)
# fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print('GAME OVER')
