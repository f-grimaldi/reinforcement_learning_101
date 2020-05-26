import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


"""
Define a dictionary that map a value to a color
"""
my_col = {5.0:  [1, 1, 0],
          3.0:  [0.2, 0.4, 0.6],
          2.0:  [0.2, 0.6, 0.6],
          1.0:  [0.2, 0.6, 0.6],
          0.0:  [0.6, 0.6, 0.9],
          -1.0: [0, 0.2, 0.3],
          -2.0: [0, 0, 0.15]}


"""
Plot a matrix representing the status of the environment
"""
def plot(mat, env, figsize=(7, 7), reduct = False, close=False):

    fig = plt.figure(figsize=(7, 7))
    img = np.zeros((mat.shape[0], mat.shape[1], 3))
    for key in my_col.keys():
        img[mat == key] = my_col[key]

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):

            # Agent in Sand
            if mat[j, i] == env.value_map['Agent'] and [j, i] in env.sand:
                if reduct == True:
                    plt.text(i, j, 'A\n(in Sand)', ha="center", va="center")
                else:
                    plt.text(i, j, 'Agent (In Sand)', ha="center", va="center")

            # Agent in Goal
            elif mat[j, i] == env.value_map['Agent'] and [j, i] == env.goal:
                if reduct == True:
                    plt.text(i, j, 'A\n(in Goal)', ha="center", va="center")
                else:
                    plt.text(i, j, 'Agent\nin Goal', ha="center", va="center")

            # Path
            elif mat[j, i] == env.value_map['Path']:
                if reduct == True:
                    plt.text(i, j, 'P', ha="center", va="center")
                else:
                    plt.text(i, j, 'Path', ha="center", va="center")

            # Terrain
            elif mat[j, i] == env.value_map['Terrain']:
                if reduct == True:
                    continue
                else:
                    plt.text(i, j, 'Terrain', ha="center", va="center")

            # Obstacle
            elif mat[j, i] == env.value_map['Obstacle']:
                if reduct == True:
                    plt.text(i, j, 'O', ha="center", va="center", color="w")
                else:
                    plt.text(i, j, 'Obstacle', ha="center", va="center", color="w")

            # Path in Sand
            elif mat[j, i] == env.value_map['Path\n(In Sand)']:
                if reduct == True:
                    plt.text(i, j, 'P\n(in Sand)', ha="center", va="center")
                else:
                    plt.text(i, j, 'Path\n(in Sand)', ha="center", va="center")

            # Sand
            elif mat[j, i] == env.value_map['Sand']:
                if reduct == True:
                    plt.text(i, j, 'S', ha="center", va="center")
                else:
                    plt.text(i, j, 'Sand', ha="center", va="center")

            # Agent
            elif mat[j, i] == env.value_map['Agent']:
                if reduct == True:
                    plt.text(i, j, 'A', ha="center", va="center")
                else:
                    plt.text(i, j, 'Agent', ha="center", va="center")

            # Goal
            else:
                if reduct == True:
                    plt.text(i, j, 'G', ha="center", va="center")
                else:
                    plt.text(i, j, 'Goal', ha="center", va="center")


    # Initial
    if reduct == True:
        plt.text(env.start[1], env.start[0], '\n\n(Start)', ha="center", va="center")
    else:
        plt.text(env.start[1], env.start[0], '\n\n(Start)', ha="center", va="center")

    im = plt.imshow(img)
    if close == True:
        plt.close()

    return im

"""
Add a legend to a matrix
"""
def add_patches(im, env):
    values = np.unique(env.matrix.ravel())

    # colormap used by imshow
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=my_col[i], label="{}: {}".format(env.value2char[i][0], env.value2char[i])) for i in values if i != 1]

    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.99), loc=2, borderaxespad=0.2 )
    plt.show()
