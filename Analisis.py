import pandas as pd     #Base de datos
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos


#   Cargamos La base de datos como "m_df" --> Main Data Frame
#   en el script "main.py" de este Directorio haremos
#   lo necesario para crear dicha base de datos en un xlsx que cargaremos aqui.
m_df = pd.read_excel('MergedDataFrame.xlsx', index_col=0)

#------------------------------
#   Micro script para obtener los valores máximos de la vuelta olímpica
#   Necesita una recorrida limpia de toda la piscina en circular
#t_df = m_df['P_position_x']
#max_x = t_df.max()
#print('Maximo: ',max_x)
#   Como resultado en nuestra versión de Simian obtuvimos que la piscina mide
#   280 unidades originales de radio.
#------------------------------


#Ensayo de gráfico de "paths"

ax = sns.lineplot(x="P_position_x", y="P_position_y",hue="Trial", data=m_df, linewidth=3, alpha = 0.8, legend='full', sort= False)
sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Test-Run')
plt.show()


ax = sns.lineplot(x="P_position_x", y="P_position_y",hue="Trial", data=m_df, linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
#ax.add_artist(rectB)
ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)

ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)
plt.show()