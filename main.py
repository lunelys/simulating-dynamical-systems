# Author: Lunelys RUNESHAW <lunelys.runeshaw@etudiant.univ-rennes.fr>
# Master 2, Project for the UE Simulating Dynamical Systems in Biology, last edit 17/12/23
# Script to simulate the patterns and an interruption of it on the zebrafish skin, based on D. Bullara, 2015
# Python 3.10

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Ways to encode the boundaries. Put only one True at a time, only change here.
periodic_b = True  # the initial one (best)
closed_b = False
open_b = False

# parameters for space and time
length = 100  # length => lattice 100x100

t_init = 0
t_end = 1000000  # generally enough without a cut: 1000000; with a cut: 2000000
dt = 0.01  # time is advanced by 1/N0 (N0 = nb of nodes, 100 because lattice = 100x100)
t = np.arange(t_init, t_end, dt)
Nt = len(t)  # with t_end = 100, Nt = 10000

# parameters
dm = 0
dx = 0
bm = 0
bx = 1
sm = 1
sx = 1
lx = 2.5
h = 16

# parameters healing: we can set healing rates that are different for each cellular type
hx = 1
hm = 1
hi = 1
hs = 1  # the cells that are not a chromatophore can also heal... at a different rate too.

# definition of neighbors, to apply correctly periodic boundary conditions
neighb1 = np.zeros((length, length), dtype=int)  # represents the neighbors in one direction
neighb2 = np.zeros((length, length), dtype=int)  # represents the neighbors in the opposite direction
for i in range(length):
    for j in range(length):
        if periodic_b or open_b:  # neighbors calculation for PERIODIC boundaries
            neighb1[i][j] = (j + i) % length
            neighb2[i][j] = (j - i) % length
        elif closed_b:  # neighbors calculation for CLOSED boundaries (right now just stays blocked at the border)
            neighb1[i][j] = min(j + i, length - 1)
            neighb2[i][j] = max(j - i, 0)
        # neighbors calculation for OPEN boundaries : no need, simply change the picking of i,j in the loop


def pick_neighbor(i, j, lattice, cell_type, neighb_type):
    site = np.random.randint(4)  # 4 for all the directions (down, up, right, left)
    if site == 0 and lattice[neighb1[1][i], j] == neighb_type:  # row neighb1[1][i] of L, column j
        lattice[i][j] = cell_type
    elif site == 1 and lattice[neighb2[1][i], j] == neighb_type:
        lattice[i][j] = cell_type
    elif site == 2 and lattice[i, neighb1[1][j]] == neighb_type:  # row i of L, column neighb1[1][j]
        lattice[i][j] = cell_type
    elif site == 3 and lattice[i, neighb2[1][j]] == neighb_type:
        lattice[i][j] = cell_type

# initialize the lattice with all white (0); better biologically speaking to use this one
# L = np.zeros((length, length), dtype=int)  # otherwise by default it is of type float (0.0)
# OR: create the lattice with random initial conditions
L = np.random.randint(0, 3, (length, length), dtype=int)

# IRIDOPHORES. If you don't want iridophores, set band to 0
# cf: "It is known that the initial stage of the pattern in the trunk of the wild type zebrafish consists of a
# single band of iridophores, which inhibits the growth of melanophores on top of them"
band = 2  # width band of iridophores
LI = np.zeros((length, length), dtype=int)  # initialization of the iridophores lattice, below L
if band > 0:  # if there is a band of iridophores, we fill it with 1s at the location i,j
    for i in range(length):
        for j in range(int(length / 2 - band / 2), int(length / 2 + band / 2)):
            LI[i][j] = 1  # Note: the band of iridophore doesn't really appear blue because its color is dulled (alpha = 0.8 of the upper epidermis)
# Example for L = 10, band = 2
#     for i in range(10): # i will go from 0 to 10
#         for j in range(int(L / 2 - band / 2), int(L / 2 + band / 2)): # j will go from int(10/2-2/2), int(10/2 + 2/2) so 4 to 6
#             LI[i][j] = 1  # so all i (line), and j's between 4 and 6 (vertical), making indeed a 2 width band of iri

# add a cut: parameters. If you don't want the cut to happen, set cut_width to 0
# how wide is the cut?
cut_width = 0  # 15 is good
# cut_width = np.random.randint(1, length)  # for later, if we want to randomize the cut width
# when does the cut happen? (2/3: here test quite at the end)
cut_time = int(Nt * (2 / 3))  # how does the pattern react if it happens at the beginning, mid, at the end, etc
# cut_time = int(Nt * np.random.randint(0, 1))  # for later, if we want to randomize the cut time
# where does the cut happen?
cut_location = int(length * (1/4))  # on the vertical, in the first quarter
# cut_location = length - np.random.randint(0, length)  # for later, if we want to randomize the cut location
# Note: for the moment it is only vertical and on all the length. Could make it diagonal and not necessarily along the whole length

# add a defect: parameters. If you don't want a defect, set defect_width to 0
# how wide is the defect?
defect_width = 0
# defect_width = np.random.randint(1, length)  # for later, if we want to randomize the defect width
# where is the defect located?
defect_location = int(length * (1/4))  # on the vertical, in the third quarter
# defect_location = length - np.random.randint(0, length)  # for later, if we want to randomize the defect location
# Same note than for the cut. Could make it diagonal and not necessarily along the whole length
# For the moment, let's code it as a simple small square of 10x10 starting at defect_location
defect_area = range(defect_location, defect_location + defect_width)
if defect_width > 0:  # if there is a defect, we reset both lattices with 0s in the defect location (no chromatophore)
    for i in defect_area:
        for j in defect_area:
            L[i][j] = 0  # the upper epidermis has a defect...
            LI[i][j] = 0  # ... and the lower too
# Now the simulation will begin with a patch of 0 at the defect location.

# parameters for the snapshot function, graph construction etc
divider = 5  # how many evenly spaced snapshots do we want? 5 here to see the evolution in detail
snapshots_time = list(range(0, Nt + 1, Nt // divider))
# https://stackoverflow.com/questions/49761923/plot-a-matrix-in-python-with-custom-colors
colorsLI = 'white blue red'.split()  # the lower epidermis can also be cut -> bloody = color code 2
cmapLI = matplotlib.colors.ListedColormap(colorsLI, name='colors', N=None)
colorsL = 'white yellow black red'.split()  # we add the red for the cut! -> bloody = color code 3
cmapL = matplotlib.colors.ListedColormap(colorsL, name='colors', N=None)
# those parameters do not change in the suptitle. Let's prepare them only once:
if cut_width > 0:
    cut_str = "Cut width = " + str(cut_width)
else:
    cut_str = "No cut"
if defect_width > 0:
    defect_str = ", defect width = " + str(defect_width)
else:
    defect_str = ", no defect"
if band > 0:
    iri_str = ", iridophore band width = " + str(band)
else:
    iri_str = ", no iridophores"


def snapshot(iteration, period):  # to "take a picture" at regular interval (to change the interval, change the divider var)
    fig, ax = plt.subplots()
    if band > 0:  # if we don't use the iridophore layer, there is no need to add the LI matrix
        im1 = ax.imshow(LI, cmap=cmapLI, alpha=1, vmin=0, vmax=len(colorsLI) - 1)  # LI is below: we display it first
        cbar1 = plt.colorbar(im1, ax=ax, ticks=range(len(colorsLI)), format="%d", orientation='vertical',
                             label='Colorbar for LI (lower epidermis)')  # legend
    im2 = ax.imshow(L, cmap=cmapL, alpha=0.7, vmin=0, vmax=len(colorsL) - 1)  # L is up: we display it last
    cbar2 = plt.colorbar(im2, ax=ax, ticks=range(len(colorsL)), format="%d", orientation='vertical',
                         label='Colorbar for L (upper epidermis)', pad=0.1)
    fig.suptitle(cut_str + defect_str + iri_str, y=0.97)  # the y argument is just to lower a bit the title, which is otherwise too high
    ax.set_title("Iteration k = " + str(iteration) + ": " + period)
    plt.show()


# probabilities
total = bx + bm + dx + dm + sx + sm + lx
P_bx = bx / total
P_bm = bm / total
P_dx = dx / total
P_dm = dm / total
P_sx = sx / total
P_sm = sm / total
P_lx = lx / total

total_healing = hx + hm + hi + hs
P_hx = hx / total_healing
P_hm = hm / total_healing
P_hi = hi / total_healing
P_hs = hs / total_healing

# alternative approach:
# https://stackoverflow.com/questions/62075701/how-to-display-values-of-an-array-as-colored-lattice-points
# Here, approach with Monte Carlo simulation: https://www.ibm.com/topics/monte-carlo-simulation
# Monte Carlo Simulation predicts a set of outcomes based on an estimated range of values versus a
# set of fixed input values. In other words, a Monte Carlo Simulation builds a model of possible results by
# leveraging a probability distribution, such as a uniform or normal distribution, for any variable that has inherent
# uncertainty. It, then, recalculates the results over and over, each time using a different set of random numbers
# between the minimum and maximum values. In a typical Monte Carlo experiment, this exercise can be repeated
# thousands of times to produce a large number of likely outcomes.


# algorithm
for k in range(Nt + 1):
    if k in snapshots_time:
        snapshot(k, "")  # take a snapshot at the beginning of the loop
    # Select a random site (i, j)
    i, j = np.random.randint(0, length, 2)  # returns 2 random integers between 0 and length
    if open_b:  # but if we are in the context of open boundaries, we need to reassign i,j to slightly different values
        i, j = np.random.randint(1, length - 1, 2)  # for OPEN boundaries: we stay away from the borders
    # the healing part -----
    if cut_width > 0:  # if there is a cut
        # if i, j is bloody, we do different rules: different process: process_healing
        process_healing = np.random.rand()  # random process value, to which the healing probability will be compared to
        if process_healing <= P_hs:
            if L[i][j] == 3:
                pick_neighbor(i, j, L, 0, 0)
            if LI[i][j] == 2:  # new "if" because we can heal simultaneously the upper and lower epidermis
                if j not in range(int(length / 2 - band / 2), int(length / 2 + band / 2)):
                    pick_neighbor(i, j, LI, 0, 0)
        # In the case of a defect, and after a cut: the skin can heal in the defect area... but the defect will stay: no
        # chromatophore will recolonize that area. So we check each time if we are not in that area in the cases of chromatophore healing
        elif P_hs < process_healing <= P_hs + P_hx:  # if the selected process falls within the xanthophore healing probabilities...
            if defect_width == 0 or (defect_width > 0 and (i not in defect_area or j not in defect_area)):
                if L[i][j] == 3:  # ... if the lattice site (i, j) is bloody... (upper epidermis)
                    pick_neighbor(i, j, L, 1, 1)
        elif P_hs + P_hx < process_healing <= P_hs + P_hx + P_hm:
            if defect_width == 0 or (defect_width > 0 and (i not in defect_area or j not in defect_area)):
                if L[i][j] == 3:
                    pick_neighbor(i, j, L, 2, 2)
        elif process_healing > P_hs + P_hx + P_hm:  # it falls within the iridophore healing probabilities...
            if defect_width == 0 or (defect_width > 0 and (i not in defect_area or j not in defect_area)):
                if LI[i][j] == 2:  # ... and if the lattice site (i, j) is bloody... (lower epidermis)
                    if j in range(int(length / 2 - band / 2), int(length / 2 + band / 2)):  # ... and the lattice site should be an iridophore ... (= in the initial band)
                        pick_neighbor(i, j, LI, 1, 1)
    # the regular article part -------
    # defect: we change nothing if i and j are in the defect patch (so that it stays at 0 whatever happens)
    if defect_width == 0 or (defect_width > 0 and (i not in defect_area or j not in defect_area)):
        # Select a process with the appropriate probability; def: "random samples from a uniform distribution over [0, 1)."
        process = np.random.rand()  # a random process value, to which the different probabilities will be compared to
        if process <= P_bx:  # the selected process falls within the birth probabilities...
            if L[i][j] == 0:  # ... and the lattice site (i, j) is unoccupied...
                L[i][j] = 1  # ... then update that location to yellow (X) (=> for yellow, the smallest!)
        elif P_bx < process <= P_bx + P_bm:  # if the process is between the birth probabilities of a xanthophore and the one of a melanophore...
            if L[i][j] == 0 and LI[i][j] == 0:  # ... and it's unoccupied AND there is no iridophore below...
                L[i][j] = 2  # ... then update that location to black (M) (=> for black birth, a bit more probable!)
        elif P_bx + P_bm < process <= P_bx + P_bm + P_dx:  # etc, each time adding a probability from the precedent
            if L[i][j] == 1:
                L[i][j] = 0  # white (S)
        elif P_bx + P_bm + P_dx < process <= P_bx + P_bm + P_dx + P_dm:
            if L[i][j] == 2:
                L[i][j] = 0
        elif P_bx + P_bm + P_dx + P_dm < process <= P_bx + P_bm + P_dx + P_dm + P_sx:
            if L[i][j] == 2:
                pick_neighbor(i, j, L, 0, 1)
        elif P_bx + P_bm + P_dx + P_dm + P_sx < process <= P_bx + P_bm + P_dx + P_dm + P_sx + P_sm:
            if L[i][j] == 1:
                pick_neighbor(i, j, L, 0, 2)
        elif process > P_bx + P_bm + P_dx + P_dm + P_sx + P_sm:
            if L[i][j] == 0 and LI[i][j] == 0:  # we can't have an iridophore below
                angle = np.random.rand() * 2 * np.pi  # random samples from a uniform distribution over [0, 1)
                cosangle = np.cos(angle)  # angle = input array in radians
                sinangle = np.sin(angle)
                inew = int(i + cosangle * h + 0.5)
                # -----
                jnew = int(j + sinangle * h + 0.5)
                if periodic_b:  # regular way (periodic boundary)
                    if inew > length - 1:
                        inew -= length
                    elif inew < 0:
                        inew += length
                    if jnew > length - 1:
                        jnew -= length
                    elif jnew < 0:
                        jnew += length
                elif closed_b:  # closed boundary (right now it just stays blocked at the border)
                    if inew > length - 1:
                        inew = length - 1
                    elif inew < 0:
                        inew = 0
                    if jnew > length - 1:
                        jnew = length - 1
                    elif jnew < 0:
                        jnew = 0
                elif open_b: # if the boundaries are opened and we are out of the boundaries, we continue to the next iteration
                    if inew > length - 1 or inew < 0:
                        continue
                    if jnew > length - 1 or jnew < 0:
                        continue
                if L[inew][jnew] == 1:  # if at the far away location there is a xanthophore...
                    L[i][j] = 2  # ... then the current position becomes a melanophore.
    # all at the end, we check if the cut is supposed to happen:
    if cut_width > 0 and k == cut_time:
        snapshot(k, "just before cut")  # ... we take a snapshot just before the cut
        for x in range(length):  # to make the cut diagonal
            for y in range(length):  # x and y to not mess the i, j of the loop
                # Calculate the distance from the current point to the line defining the cut
                distance_to_cut = abs((x - cut_location) + (y - length / 2)) / np.sqrt(2)
                # Check if the point is within the cut width and distance
                if distance_to_cut <= cut_width / 2:
                    L[x][y] = 3  # on the upper epidermis -> becomes bloody (red)
                    LI[x][y] = 2  # and on the epidermis below (iridophores) -> becomes bloody (red)
        snapshot(k, "just after cut")  # ... and just after the cut
