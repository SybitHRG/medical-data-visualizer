import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df[['cholesterol', 'gluc']] = (df[['cholesterol', 'gluc']] > 1).astype(int)
# print(df.head().to_string())


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    # print(df_cat.value_counts())

    # Draw the catplot with 'sns.catplot()'
    catplot = sns.catplot(data=df_cat, x='variable', hue='value', col='cardio',
                          kind='count',
                          order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'],
                          )
    catplot.set_ylabels('total')

    # Get the figure for the output
    fig = catplot.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    mask_invalid_pressure = df['ap_lo'] <= df['ap_hi']
    mask_height_low_quantile = df['height'] >= df['height'].quantile(0.025)
    mask_height_high_quantile = df['height'] <= df['height'].quantile(0.975)
    mask_weight_low_quantile = df['weight'] >= df['weight'].quantile(0.025)
    mask_weight_high_quantile = df['weight'] <= df['weight'].quantile(0.975)
    df_heat = df.loc[mask_invalid_pressure & mask_height_low_quantile & mask_height_high_quantile & mask_weight_low_quantile & mask_weight_high_quantile]
    # print(df_heat.head().to_string())
    # print(df.shape)
    # print(df_heat.shape)

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones(corr.shape, dtype=bool), 0)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(data=corr, ax=ax, mask=mask, annot=True, fmt='.1f', linewidths=.5)


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
