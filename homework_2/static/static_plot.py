import pandas as pd
import matplotlib.pyplot as plt

csv_filename = 'static.csv'
df = pd.read_csv(csv_filename)

plt.violinplot(df)
plt.xlabel('Joint')
plt.ylabel('Torque')
plt.xticks(ticks=[1, 2, 3, 4], labels=['pitch', 'roll', 'yaw', 'elbow'])

plt.savefig('static.png')

plt.show()
