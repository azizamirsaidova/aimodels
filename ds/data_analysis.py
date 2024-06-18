import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#create empty series
series = pd.Series(dtype="float64")
print("{}\n".format(series))

series = pd.Series([1, 2, 3])
print('{}\n'.format(series))

# adding index to series
series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print('{}\n'.format(series))

#using dictionary for series
series = pd.Series({'a':1, 'b':2, 'c':3})
print('{}\n'.format(series))

#create a dataframe 
df = pd.DataFrame([[5, 6], [1, 3]],
                  index=['r1', 'r2'],
                  columns=['c1', 'c2'])

#dropping the data
df_drop = df.drop(labels='r1')
df_drop = df.drop(labels=['c1', 'c3'], axis=1)
df.drop(index='r2', columns='c2')

#combining dataframes
df1 = pd.DataFrame({'c1':[1,2], 'c2':[3,4]},
                   index=['r1','r2'])
df2 = pd.DataFrame({'c1':[5,6], 'c2':[7,8]},
                   index=['r1','r2'])
df3 = pd.DataFrame({'c1':[5,6], 'c2':[7,8]})

concat = pd.concat([df1, df2], axis=1)

#merging dataframes
mlb_df1 = pd.DataFrame({'name': ['john doe', 'al smith', 'sam black', 'john doe'],
                        'pos': ['1B', 'C', 'P', '2B'],
                        'year': [2000, 2004, 2008, 2003]})
mlb_df2 = pd.DataFrame({'name': ['john doe', 'al smith', 'jack lee'],
                        'year': [2000, 2004, 2012],
                        'rbi': [80, 100, 12]})

mlb_merged = pd.merge(mlb_df1, mlb_df2)


# iloc - access rows based on integer index 
print('{}\n'.format(df.iloc[1]))

# loc - access rows based on row labels 
print('{}\n'.format(df.loc['r2']))

# read json data
df1 = pd.read_json('data.json')

# read excel 
with pd.ExcelWriter('data.xlsx') as writer:
  mlb_df1.to_excel(writer, index=False, sheet_name='NYY')
  mlb_df2.to_excel(writer, index=False, sheet_name='BOS')
  
df_dict = pd.read_excel('data.xlsx', sheet_name=None)
print(df_dict.keys())

#grouping
groups = df.groupby('yearID')

#filtering
str_f1 = df['playerID'].str.startswith('c')

str_f2 = df['teamID'].str.endswith('S')

str_f3 = ~df['playerID'].str.contains('o')


#true if nan is there, false else
isna = df['teamID'].isna()

#opposite of isna
notna = df['teamID'].notna()
print('{}\n'.format(notna))

#feature filtering
hr40_df = df[df['HR'] > 40]
not_hr30_df = df[~(df['HR'] > 30)]

#sorting
sort1 = df.sort_values('yearID')

sort2 = df.sort_values('playerID', ascending=False)

#plotting
df.plot(kind='line',x='yearID',y='HR')
plt.show()

#convert to binary
converted = pd.get_dummies(df)
