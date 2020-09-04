from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd

# csv = ["./TestGesture/single/subject1.csv", "./TestGesture/single/subject2.csv","./TestGesture/single/subject3.csv"]
csv = ["./TestGesture/single/subject1_16.csv", "./TestGesture/single/subject2_16.csv","./TestGesture/single/subject3_16.csv"]

t = 0.25
step = 1

# csv = ["./TestGesture/continuous/subject1_"+str(t) + "_"+str(step)+".csv", 
#                         "./TestGesture/continuous/subject2_"+str(t) + "_"+str(step)+".csv", 
#                         "./TestGesture/continuous/subject3_"+str(t) + "_"+str(step)+".csv", ]


df1 = pd.read_csv(csv[0])
df1 = df1.dropna()
df2 = pd.read_csv(csv[1])
df2 = df2.dropna()
df3 = pd.read_csv(csv[2])
df3 = df3.dropna()

print('Accuracy across subjects')
print('Accuracy:', accuracy_score(df1['ground_truth'], df1['result']))
print('Accuracy:', accuracy_score(df2['ground_truth'], df2['result']))
print('Accuracy:', accuracy_score(df3['ground_truth'], df3['result']))
print('Average speed: ', df1['time'].mean())
print('Average speed: ', df2['time'].mean())
print('Average speed: ', df3['time'].mean())


new_df1 = pd.concat([df1[df1['file'].str.contains("s1")], df2[df2['file'].str.contains("s1")], df3[df3['file'].str.contains("s1")]])
new_df2 = pd.concat([df1[df1['file'].str.contains("s2")], df2[df2['file'].str.contains("s2")], df3[df3['file'].str.contains("s2")]])
new_df3 = pd.concat([df1[df1['file'].str.contains("s3")], df2[df2['file'].str.contains("s3")], df3[df3['file'].str.contains("s3")]])
print('Accuracy across scenes')
print('Accuracy:', accuracy_score(new_df1['ground_truth'], new_df1['result']))
print('Accuracy:', accuracy_score(new_df2['ground_truth'], new_df2['result']))
print('Accuracy:', accuracy_score(new_df3['ground_truth'], new_df3['result']))
print('Average speed: ', new_df1['time'].mean())
print('Average speed: ', new_df2['time'].mean())
print('Average speed: ', new_df3['time'].mean())


df = pd.concat([df1, df2, df3])
print('Accuracy across scenes')
print(df['result'].value_counts())
print('Accuracy:', accuracy_score(df['ground_truth'], df['result']))
print('Average speed: ', df['time'].mean())
print('totoal numbers of sample:' + str(df.shape[0]))

# df1['total_frames'].hist(bins=10)
# plt.ylabel('frequency')
# plt.xlabel('duration of samples')
# plt.title('distribution of sample durations of subject1')
# plt.show()

# df2['total_frames'].hist(bins=10)
# plt.ylabel('frequency')
# plt.xlabel('duration of samples')
# plt.title('distribution of sample durations of subject2')
# plt.show()

# df3['total_frames'].hist(bins=10)
# plt.ylabel('frequency')
# plt.xlabel('duration of samples')
# plt.title('distribution of sample durations of subject3')
# plt.show()

print(df['total_frames'].mean())