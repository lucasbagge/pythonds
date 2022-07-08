#!/usr/bin/env python
# coding: utf-8

# # Pandas
# 
# A work though of the most important concept in python most essentiel data tools, `pandas`.
# 
# ## Series og Dataframes
# 
# En af de første ting som er værd at forstå ved Pandas er at det hovedsaglig er bygget af to data struktur:
# 
# - `Serier`
# - `Dataframes`
# 
# ![](fig1.png)
# 
# *Figur 1: Forskel mellem Serie og dataframes.*
# 
# At forstå at der er en forskel mellem de to struktur vil gerne en i ens brug af Pandas. Da dataframes er et bundt af serier giver det mening at lave et deep dive ind i serier.
# 
# ![](fig2.png)
# 
# *Figur 2: Figuren viser forholdet mellem data strukturen i pandas.*
# 
# Lad os nu lave en panda serie.

# In[1]:


import pandas as pd
songs2 = pd.Series([145, 142, 38, 13], name='counts')
songs2


# In[2]:


import numpy as np

songs3 = np.array([145, 142, 38, 13])

print("Numpy: \n", songs3[1])

print("Pandas: \n", songs2[1])


# Venstre kolonne, som ikke er en del af selve data værdien er vores index. 
# 
# En Pandas serier og numpy array er meget lig hinanden og kan begge lave index operationer.
# 
# ### Kategorisk variabler
# 
# Hvis man ved at en serier har få værdier kan man bruge en `kategorisk` variabel som har fordelene:
# 
# - Bruger mindre hukommelse end strings.
# - Forbedre præsentation. 
# - Ordre
# 
# En fordel ved kategorisk variabler er de kan indeholder dato, tal og bogstaver som hhv. kan konverteres til et andet format.
# 
# En kategori kan laves med `dtype=category` eller med `.astype("category")`.

# In[3]:


s = pd.Series(['xs', 's', 'm', 'l', 'xl'], dtype='category')
s


# In[4]:


s2 = pd.Series(['m', 'l', 'xs', 's', 'xl'])

size_type = (
    pd
    .api
    .types
    .CategoricalDtype(categories = ['s', 'm', 'l'], 
                      ordered = True)
    )

s3 = s2.astype(size_type)   

s3


# In[5]:


s3 > 's'


# Den forrig funktion transformerede vi om til en ordre kategory, men vi kan også gøre det med det samme.

# Vi kan med `CategoricalDtype` lave en kategori med en ordre. 

# In[6]:


s2 = pd.Series(['m', 'l', 'xs', 's', 'xl'])

size_type = (
    pd
    .api
    .types
    .CategoricalDtype(categories = ['s', 'm', 'l'], 
                      ordered = True)
    )

s3 = s2.astype(size_type)   

s3


# ### Dunder operator
# 
# Dunder metoder kendes også under "operator" eller "magic" metoder. Helt kort er de beskrivelser på hvordan Python reagerer til operationer. F.eks. når man bruger + så vil man nedenunder benytte sig af `.__add__` metoden. 

# In[7]:


print("Dunder metoden", 2+2)
f'hvad sker der egentlig: {(2).__add__(4)}'


# ### Dataframe

# ## Aggregation
# 
# 
# I pandas er der flere måde at summer data på. 
# 
# Dette afsnit omhandler nogle af disse metoder. Her vil vi allerede først starte ud med den simple `.agg` og udvide det til de mere omfattende metode; `.pivot_table`, `.groupby` og `.crossbar`.
# 
# Disse funktioner minder meget om hinanden og gør i bund og grund de samme ting.
# 
# 
# | Metode | Beskrivelse |
# | --- | --- |
# |s.agg(func=None, axis=0, *args, **kwargs)|Returns a scalar if func is a single aggregation function. Returns a series if a list of aggregations are passed to func.|
# |s.all(axis=0, bool_only=None, skipna=True, level=None)|Returns True if every value is truthy. Otherwise False|
# |s.any(axis=0, bool_only=None, skipna=True, level=None)|Returns True if at least one value is truthy. Otherwise False|
# |s.autocorr(lag=1)|Returns Pearson correlation between s and shifted s|
# |s.corr(other, method='pearson')|Returns correlation coefficient for 'pearson', 'spearman', 'kendall', or a callable.|
# |s.cov(other, min_periods=None)|Returns covariance.|
# |s.min(axis=None, skipna=None, level=None,numeric_only=None)| Returns maximum value.|
# |s.mean(axis=None, skipna=None,level=None, numeric_only=None)|Returns minumum value.|
# |s.median(axis=None, skipna=None,level=None, numeric_only=None)|Returns mean value.|
# |s.prod(axis=None, skipna=None,level=None, numeric_only=None,min_count=0)|Returns product of s values.|
# |s.quantile(q=.5, interpolation='linear')|Returns 50% quantile by default. Note returns Series if q is a list|
# |s.sem(axis=None, skipna=None, level=None,ddof=1, numeric_only=None)|Returns unbiased standard error of mean.|
# |s.sem(axis=None, skipna=None, level=None,ddof=1, numeric_only=None)|Returns unbiased standard error of mean.|
# |s.sem(axis=None, skipna=None, level=None,ddof=1, numeric_only=None)|Returns unbiased standard error of mean.|
# |s.std(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)|Returns sample standard deviation.|
# |s.var(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)|Returns unbiased variance.|
# |s.skew(axis=None, skipna=None, level=None, numeric_only=None)|Returns unbiased skew.|
# |s.kurtosis(axis=None, skipna=None, level=None, numeric_only=None)|Returns unbiased kurtosis.|
# |s.nunique(dropna=True)|Returns count of unique items.|
# |s.count(level=None)|Returns count of non-missing items.|
# |s.size|Number of items in series. (Property)|
# |s.is_unique|True if all values are unique|
# 
# 
# 
# | Metode | Beskrivelse |
# | --- | --- |
# |pd.crosstab(index, columns, values=None,rownames=None, colnames=None,aggfunc=None, margins=False,margins_name='All', dropna=True,normalize=False)| Create a cross-tabulation (counts by default) from an index (series or list of series) and columns (series or list of series). Can specify a column (series) to aggregate values along with a function, aggfunc. Using margins=True will add subtotals. Using dropna=False will keep columns that have no values. Can normalize over 'all' values, the rows ('index'), or the 'columns'.|
# |.pivot_table(values=None, index=None,columns=None, aggfunc='mean',fill_value=None, margins=False,margins_name='All', dropna=True,observed=False, sort=True)|Create a pivot table. Use index (series, column name, pd.Grouper, or list of previous) to specify index entries. Use columns (series, column name, pd.Grouper, or list of previous) to specify column entries. The aggfunc (function, list of functions, dictionary (column name to function or list of functions) specifies function to aggregate values. Missing values are replaced with fill_value. Set margins=True to add subtotals/totals. Using dropna=False will keep columns that have no values. Use observed=True to only show values that appeared for categorical groupers.|
# |.groupby(by=None, axis=0, level=None,as_index=True, sort=True,group_keys=True, observed=False,dropna=True)|Return a grouper object, grouped using by (column name, function (accepts each index value, returns group name/id), series, pd.Grouper, or list of column names). Use as_index=False to leave grouping keys as columns. Common plot parameters. Use observed=True to only show values that appeared for categorical groupers. Using dropna=False will keep columns that have no values.|
# |df.resample(rule, axis=0, closed=None,label=None, convention='start',kind=None, on=None, level=None,origin='start_day')|Return a resampled dataframe (with a date in the index, or specify the date column with on). Set closed to 'right' to include the right side of interval (default is 'right' for M/A/Q/BM/BQ/W). Set the label to 'right' to use the right label for bucket. Can specify the timestamp to start origin.|
# 
# 

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('00_data_raw/bikes.xlsx')


# In[9]:


df.head()


# In[10]:


(
    df
    ['price']
    .sum()
)


# ### .aggandAggregationStrings
# 
# 
# `.agg` metoden transformer data. Det er også brugbar hvis man har flere aggregationer.

# In[11]:



(
    df
    ['price']
    .agg([sum, np.mean, np.var])
)


# Du kan også tilføje en dictonary så kan lave udregninger på flere kolonner. 

# In[12]:


df.agg({'model': ['count'], 'price': ['sum', 'max']})


# Til pandas er der en række metoder der går igen for aggreation.

# In[13]:


bikesshop = pd.read_excel("00_data_raw/orderlines.xlsx")
bikesshop.head()


# 

# In[14]:


from string import ascii_uppercase

cols = list(ascii_uppercase[:10])
np.random.seed(42)
data = np.random.randint(1, 100, size=(100_000, 10))
df = pd.DataFrame(data, columns=cols)
df.head()


# ### `pivot_table`

# In[15]:


(
    pd.crosstab(index = df.A, columns = df.B)
)


# ### `crosstab`
# 
# Vi kan også bruge `crosstab`, som er forskellige fra pivot_table  da den tager serier.

# In[16]:


(
    df
    .pivot_table(values='C',
                 index='A',
                 columns='B',
                 aggfunc='count',
                 fill_value=0)
)


# ### `groupby`
# 
# Den sidste metode er `groupby`

# In[17]:


(
    df
    .groupby(['A', 'B'])
    ['C']
    .count()
    .unstack(fill_value=0)
)


# ### Forskel mellem funktionerne
# 
# Vi kan gøre det samme for multiple aggrestioner.

# In[18]:


get_ipython().run_cell_magic('timeit', '', "(\n    df\n    .groupby(['A', 'B'])\n    ['C']\n    .count()\n    .unstack(fill_value=0)\n    )")


# In[19]:


get_ipython().run_cell_magic('timeit', '', "(\n    df\n    .pivot_table(values='C',\n                 index='A',\n                 columns='B',\n                 aggfunc='count',\n                 fill_value=0)\n)")


# In[20]:


get_ipython().run_cell_magic('timeit', '', 'pd.crosstab(index=df.A, columns=df.B)')


# In[21]:


import timeit
from collections import defaultdict


def crosstab(df):
    pd.crosstab(index=df.A, columns=df.B)


def groupby(df):
    (
        df
        .groupby(['A', 'B'])
        ['C']
        .count()
        .unstack(fill_value=0)
    )


def pivot_table(df):
    (
        df
        .pivot_table(values='C',
                     index='A',
                     columns='B',
                     aggfunc='count',
                     fill_value=0)
    )


funcs = [crosstab, groupby, pivot_table]
measurements = []
repetitions = 5

# Use a seed distinct from above to prevent caching
np.random.seed(420)

for size in np.logspace(start=4, stop=7, num=4):
    size = int(size)
    data = np.random.randint(1, 100, size=(size, 10))
    df = pd.DataFrame(data, columns=cols)

    for func in funcs:
        duration = timeit.timeit('func(df)', number=repetitions, globals=globals()) / repetitions
        measurements.append({'Function': func.__name__, 'Row count': size, 'duration': duration})
        

plt.style.use('seaborn-poster')
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(10, 6))
(
    pd.DataFrame(measurements)
    .groupby(['Row count', 'Function'])
    .duration
    .mean()
    .unstack()
    .plot(ax=ax, kind='bar')
)

ax.set_xlabel('DataFrame row count')
ax.set_ylabel('Duration in seconds', labelpad=25, va='top')

fig.suptitle('Runtime Comparison of pandas crosstab, groupby and pivot_table', fontsize=22)


# ### Tidsserier `.resample`
# 

# In[22]:


import pandas_datareader as pdr
# Request data via Yahoo public API
data = pdr.get_data_yahoo('NVDA')
data.head()


# In[23]:


(
    data
    .loc[:, ['Close']]
    .resample('W')
    .sum()
)


# ## Melting, Transposing, and Stacking Data

# Data kan organiseret mange måde, men to centrale måder er **wide** og **long**. Dog kan der være fordele og ulemper ved at have data i en af de respektive formater. For mit eget vedkommende kan jeg bedst lide long format til at lave analyser og plots. 
# 
# 
# | Metode | Beskrivelse |
# | --- | --- |
# | .melt(id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None, ignore_index=True) | Returner en unpivoted dataframe. |
# |.reset_index(level=None, drop=False,col_level=0, col_fill='')|Returner et nyt index level|
# |.pivot_table(values=None, index=None,columns=None, aggfunc='mean',fill_value=None, margins=False,margins_name='All', dropna=True,observed=False, sort=True)|Laver en pivot table. `Index` laver index indgangen. `columns` til kolonne indgangen. `aggfunc` angiver en dictornary til enkelte eller flere aggregertion funktioner.`fill_value` udfylder missing værdier. `margins` til at vise subtotaler.|

# In[24]:


import numpy     as np
import pandas as pd

scores = pd.DataFrame({
'name':['Adam', 'Bob', 'Dave', 'Fred'], 
'age': [15, 16, 16, 15],
'test1': [95, 81, 89, None],
'test2': [80, 82, 84, 88],
'teacher': ['Ashby', 'Ashby', 'Jones', 'Jones']})


# In[25]:


scores.head()


# ### Melting
# 
# Med `Melt` kan vi gå fra wide til long format. 

# In[26]:


(
    scores
    .melt(
        id_vars=['name', 'age'],
        value_vars=['test1', 'test2'],
        var_name='test', 
        value_name='score'
    )
)


# Hvis vi skulle gøre det uden melt funktionen:

# In[27]:


(
scores
.groupby(['name', 'age']) 
.apply(lambda g: pd.concat([
    g[['test1']]
    .rename(columns = {'test1':'val'})
    .assign(var='test1'),
    g[['test2']]
    .rename(columns = {'test2':'val'})
    .assign(var='test2')])) 
.reset_index()
.drop(columns='level_2')
)


# 
# Vores data står i long format men lad os konvertere det lidt frem og tilbage.

# ### Un melting

# In[28]:


melted = (
    scores
    .melt(
        id_vars=['name', 'age', 'teacher'],
        value_vars=['test1', 'test2'],
        var_name='test', 
        value_name='score'
    )
)
melted.head()


# In[29]:


(
    melted
    .pivot_table(
        index = ['name', 'age', 'teacher'],
        columns='test',
        values='score'
    )
    .reset_index()
)


# In[30]:


(
    melted
    .groupby(['name', 'age', 'teacher', 'test'])
    .score
    .mean()
    .unstack()
    .reset_index()
)

