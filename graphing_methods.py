import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

def figure_one(driver_full):    
    #scatter plot
    ax=sns.scatterplot(x='num_rides', y='lifetime_val', data=driver_full,  marker="+")
    
    #ridge regression
    X, y = driver_full[['num_rides']], driver_full['lifetime_val']
    reg = linear_model.Ridge(alpha=.01)
    reg = reg.fit(X, y)
    m, b = reg.coef_, reg.intercept_
    
    #plot regression line and R^2 value
    ax.plot(np.linspace(-100, 1200, 100), np.linspace(-100, 1200, 100)*m+b, c='black')
    ax.set(title='Lifetime Value vs Total Number of Rides')
    ax.set_ylabel('Lifetime Value ($)')
    ax.set_xlabel('Total Number of Rides')
    plt.show()
    print('R^2 value for this model is:', reg.score(X, y))
    
def figure_two(driver_full):
    sns.set(font_scale=1.2)
    #normalizing values
    driver_norm = driver_full.copy()
    driver_norm = driver_norm.transform(lambda x:(x - x.mean())/x.std())
    
    #performing k-means algorithm
    X = driver_norm.values
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    driver_full['group'] = labels
    
    #labeling groups
    experienced = driver_full[driver_full['group']==2]
    average = driver_full[driver_full['group']==0]
    starting = driver_full[driver_full['group']==1]
    groups = {2:'Experienced', 0:'Average', 1:'Starting'}
    driver_full['group'] = driver_full['group'].apply(
        lambda x:groups[x])
    
    #plotting clusters
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(starting['rides_per_day'], starting['pickup_time'], starting['total_fares'], label='Starting')
    ax.scatter3D(average['rides_per_day'], average['pickup_time'], average['total_fares'], label='Average')
    ax.scatter3D(experienced['rides_per_day'], experienced['pickup_time'], experienced['total_fares'], label='Experienced')
    ax.view_init(20, 255)
    ax.legend(loc='center right')
    ax.set(title='K-means Clusters', xlabel='Rides per Day', ylabel='Average Pickup Time (min)', zlabel='Total Fares ($)')
    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    plt.show()
    
    #return table with groups
    return driver_full

def figure_three(driver_full):
    sns.set(font_scale=1.8)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,6))
    
    #plotting num_rides by group
    sns.boxplot(x="group", y="num_rides", data=driver_full, ax=ax1, order=['Starting', 'Average', 'Experienced'])
    ax1.set(title='Total Number of Rides by Group')
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Total Number of Rides')
    
    #plotting rides_per_day by group
    sns.boxplot(x="group", y="rides_per_day", data=driver_full, ax=ax2, order=['Starting', 'Average', 'Experienced'])
    ax2.set(title='Rides per Day by Group')
    ax2.set_xlabel('Group')
    ax2.set_ylabel('Rides per Day')
    
    #plotting lifetime_val by group
    sns.boxplot(x="group", y="lifetime_val", data=driver_full, ax=ax3, order=['Starting', 'Average', 'Experienced'])
    ax3.set(title='Estimated Lifetime Value by Group')
    ax3.set_xlabel('Group')
    ax3.set_ylabel('Lifetime Value ($)')
    plt.show()
    
def figure_four(driver_full):
    #method to add percentages to the top of bars
    def add_percent(ax, total, height_change, c):
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + height_change,
                    '{:1.2f} {}'.format(height/total*100, c),
                    ha="center") 
     
    sns.set(font_scale=1.8)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,6))
    
    #countplot of each group
    sns.countplot(x='group', data=driver_full, order=['Starting', 'Average', 'Experienced'], ax=ax1)
    ax1.set(title='Group Sizes')
    add_percent(ax1, len(driver_full), 3, '%')
    ax1.set_xlabel('Group')
    ax1.set_ylabel('Drivers in Group')
    
    #barplot of total fare by group
    sns.barplot(x='group', y='total_fares', estimator=np.sum, data=driver_full, ci=None,
                      order=['Starting', 'Average', 'Experienced'], ax=ax2)
    ax2.set(title='Fare Contribution by Group')
    add_percent(ax2, np.sum(driver_full['total_fares']), 3, '%')
    ax2.set_xlabel('Group')
    ax2.set_ylabel('Total Fares ($)')
    
    #barplot of median pickup time by group
    sns.barplot(x='group', y='pickup_time', estimator=np.median, data=driver_full, ci=None,
                order=['Starting', 'Average', 'Experienced'], ax=ax3)
    ax3.set(title='Median Pickup Time by Group')
    add_percent(ax3, 100, 0, 'min')
    ax3.set_xlabel('Group')
    ax3.set_ylabel('Median Pickup Time (min)')
    plt.show()   
    