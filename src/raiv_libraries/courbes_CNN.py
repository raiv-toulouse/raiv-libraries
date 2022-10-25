import numpy as np
import matplotlib.pyplot as plt

TYPE_IMAGE = "DEPTH"  # écrire "DEPTH" ou "RGB"
CHEMIN = '450im_depth'  # nom
TYPE_COURBE = 'val'  # écrire "val" ou "train"
#lol = open('/common/work/data_courbes_matplotlib/DEPTH/450im_depth/val/data_model_val1.txt')
giga_list = []

for number_file in range(1, 6):  # Numéro du fichier
    road = '/common/work/data_courbes_matplotlib/DEPTH/450im_depth/' + str(TYPE_COURBE) + '/data_model_val' + str(number_file) + '.txt'
    # road = '/common/work/data_courbes_matplotlib/' + str(TYPE_IMAGE) + "/" + str(CHEMIN) + "/" + str(TYPE_COURBE) + '/data_model_' + str(TYPE_COURBE) + str(number_file) + '.txt '
    a_file = open(road, "r")

    # Stockage de chaque ligne du document dans une liste
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
    giga_list.append(list_of_lists)
    # print(giga_list)

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
    plt.xlabel('epoch')

    ax1.set_title('Accuracy')
    ax1.errorbar(epoch, accuracy)

    ax2.set_title('F1_Score')
    ax2.errorbar(epoch, f1_score)

    fig.suptitle('Courbe ' + str(TYPE_COURBE) + ' ' + str(number_file))
    plt.show()

giga_list2 = np.array(giga_list)
print(giga_list2)

# Ecriture de la moyenne des loss, accuracy etc.
mean_list = np.mean(giga_list2, axis=0)

epoch = []
loss = []
accuracy = []
f1_score = []

# Ecriture de l'écart-type :
std_list = np.std(giga_list2, axis=0)

error_loss = []
error_accuracy = []
error_f1score = []

# Ecriture courbe min et max :
max_list = np.max(giga_list2, axis=0)

min_list = np.min(giga_list2, axis=0)

max_loss = []
max_accuracy = []
max_f1score = []
min_loss = []
min_accuracy = []
min_f1score = []

# Ecriture de chaque courbe/itération :

giga_loss = []
for i in range(len(giga_list2)):
    mega_loss = []
    for yolo in range(len(giga_list2[i])):
        mega_loss.append(giga_list2[i][yolo][1])
    giga_loss.append(mega_loss)

gigaloss2 = np.array(giga_loss)
print(gigaloss2)

# Séparation dans toutes les listes respectives :
for n in range(len(std_list)):
    # ecart type
    error_loss.append(std_list[n][1])
    error_accuracy.append(std_list[n][2])
    error_f1score.append(std_list[n][3])

    # moyenne
    epoch.append(mean_list[n][0])
    loss.append(mean_list[n][1])
    accuracy.append(mean_list[n][2])
    f1_score.append(mean_list[n][3])

    # max
    max_loss.append(max_list[n][1])
    max_accuracy.append(max_list[n][2])
    max_f1score.append(max_list[n][3])

    # min
    min_loss.append(min_list[n][1])
    min_accuracy.append(min_list[n][2])
    min_f1score.append(min_list[n][3])

# Affichage des courbes moyenne
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True,
                                    figsize=(12, 4))
ax0.set_title('Loss')
ax0.errorbar(epoch, loss, yerr=error_loss, label='Moyenne + écart-type')
ax0.plot(epoch, max_loss, label='Max')
ax0.plot(epoch, min_loss, label='Min')
# for a in range(len(giga_loss)):
#     ax0.plot(epoch, giga_loss[a])
ax0.set_xlabel('epoch')
ax0.legend()

ax1.set_title('Accuracy')
ax1.errorbar(epoch, accuracy, yerr=error_accuracy)
ax1.plot(epoch, max_accuracy)
ax1.plot(epoch, min_accuracy)
ax1.set_xlabel('epoch')
ax1.legend()

ax2.set_title('F1_Score')
ax2.errorbar(epoch, f1_score, yerr=error_f1score)
ax2.plot(epoch, max_f1score)
ax2.plot(epoch, min_f1score)
ax2.set_xlabel('epoch')

fig.suptitle('Courbe ' + str(TYPE_COURBE))
plt.show()
