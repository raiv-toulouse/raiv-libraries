import numpy as np
import matplotlib.pyplot as plt


for number_file in range(1, 7): # Numéro du fichier

    road = '/common/data_courbes_matplotlib/train/data_model_TRAIN' + str(number_file) + '.txt'
    a_file = open(road, "r")

    #Stockage de chaque ligne du document dans une liste
    list_of_lists = []
    for line in a_file:
        stripped_line = line.strip()
        line_list = stripped_line.split(';')
        list_of_lists.append(line_list)
    del list_of_lists[0]

    # Convertion des string en float sur la grande liste :
    for b in range(len(list_of_lists)):
        for c in range(len(list_of_lists[0])):
            list_of_lists[b][c] = float(list_of_lists[b][c])
    a_file.close()

    # Séparation de la grand liste en plusieurs petites listes epoch, loss, accuracy et f1 score :
    epoch = []
    loss = []
    accuracy = []
    f1_score = []
    for i in range(len(list_of_lists)):
        epoch.append(list_of_lists[i][0])
        loss.append(list_of_lists[i][1])
        accuracy.append(list_of_lists[i][2])
        f1_score.append(list_of_lists[i][3])

    # Affichage des résultats du graphique :
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True,
                                        figsize=(12, 4))
    ax0.set_title('Loss')
    ax0.errorbar(epoch, loss)

    ax1.set_title('Accuracy')
    ax1.errorbar(epoch, accuracy)

    ax2.set_title('F1_Score')
    ax2.errorbar(epoch, f1_score)

    fig.suptitle('Courbe Train ' + str(number_file))
    plt.show()
