import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the 3D hand data from the HDF5 file.
filename = "3Dpoints_h5\Felix070325E1points3d_150lab.h5"  # update with your actual filename
with h5py.File(filename, 'r') as f:
    # 'tracks' is expected to have shape ([frames], 1, 20, 3)
    tracks = f['tracks'][:]  
# Remove the singleton instance dimension to get shape ([frames], 20, 3)
tracks = np.squeeze(tracks, axis=1)
print(np.shape(tracks))
n_frames, n_nodes, _ = tracks.shape
print(n_frames)
print(n_nodes)

#Convert the data such that the 3 dimensional matrix collapses into a 2 dimensional matrix which will be the no. of frames by the values of XYZ for each node.
templist = []
for i in tracks:
    tempy = []
    for j in range(n_nodes):
        tempy.append(i[j][0])
        tempy.append(i[j][1])
        tempy.append(i[j][2])
    templist.append(tempy)

listnp = np.array(templist)
print(np.shape(listnp))
print(len(templist))
# print(listnp[0])

#Now we subtract every value by the previous value to get the motion data (movement in XYZ)
vellist = []
for i in range(1,len(templist)):
    changeVel = listnp[i]-listnp[i-1]
    vellist.append(changeVel)

vellistnp = np.array(vellist)
print('shape of vellistnp: ',np.shape(vellistnp))
# print(listnp[0])
# print(vellistnp[0])

#Now we fuck around with sEMG data in csv file
df = pd.read_csv('sEMGdata_9PCA_csv\FelixRH_FCU_E_070325_with_9PCA.csv', delimiter = ',', header = 'infer' ) #Change CSV file to read
dfnp = np.array(df)
print('Shape of dfnp: ',np.shape(dfnp))

tpoints, colnum = dfnp.shape

#determines how many timepoints in each chunk of sEMG data
# numT = np.floor(tpoints/(n_frames-1))
numT = tpoints/(n_frames-1)
print("no. of time points in each chunk: ",numT)

#Determine the list of indexes in the time points to place the 3D motion label.
Tindlist = [(i*numT)-1 for i in range(1,n_frames)]#the -1 term allows for Python zero indexing to work
print('Length of list containing index to put output label for seq to 1 mapping: ',len(Tindlist))
print('Final element in Tinlist: ',Tindlist[-1])

#Finally make a csv file with the labelled data
node_names = ['W', 
              'F1(0)', 'F1(1)', 'F1(2)', 
              'F2(0)', 'F2(1)', 'F2(2)', 'F2(3)', 
              'F3(0)', 'F3(1)', 'F3(2)', 'F3(3)', 
              'F4(0)', 'F4(1)', 'F4(2)', 'F4(3)', 
              'F5(0)', 'F5(1)', 'F5(2)', 'F5(3)']

for i in range(20):
    df[node_names[i]+'_X'] = np.nan
    df[node_names[i]+'_Y'] = np.nan
    df[node_names[i]+'_Z'] = np.nan

for idx, row_index in enumerate(Tindlist):
    # Get the 60-element vector for this motion interval.
    label_vector = vellistnp[idx]
    # For each of the 20 nodes, assign the X, Y, and Z values.
    for i in range(20):
        df.loc[np.floor(row_index), node_names[i] + '_X'] = label_vector[3 * i]
        df.loc[np.floor(row_index), node_names[i] + '_Y'] = label_vector[3 * i + 1]
        df.loc[np.floor(row_index), node_names[i] + '_Z'] = label_vector[3 * i + 2]

labelled_df = np.array(df)
# print(df.head())
print(np.shape(labelled_df))

output_csv = 'FelixE1_070325Labelled9PCA.csv'
df.to_csv(output_csv, index=False, sep=';')
print(f"Labeled sEMG data saved to {output_csv}")