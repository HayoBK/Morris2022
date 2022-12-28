# ---------------------------------------------------------
# Lab ONCE - Diciembre 2022
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------

import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos


m_df = pd.read_csv('AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv', index_col=0)
#p_df = pd.read_csv('AB_SimianMaze_Z2_NaviData_con_posicion.csv', index_col=0)
    # Invocamos en m_df (main Dataframe) la base de datos "corta" con calculo de CSE por Trial
    # Invocamos en p_df (position Dataframe) la base con tutti cuanti - sobre todo datos posicionales

#-------------------------------------------------------------------------------
#  @@@ Aqui primero una preparación adicional de la base de datos

m_df['Main_Block']=m_df['True_Block']  # Aqui vamos a recodificar los Bloques en un "Main Block" más grueso
m_df.loc[(m_df.Main_Block == 'FreeNav'),'Main_Block']='Non_relevant'
m_df.loc[(m_df.Main_Block == 'Training'),'Main_Block']='Non_relevant'
m_df.loc[(m_df.Main_Block == 'VisibleTarget_1'),'Main_Block']='Target_is_Visible'
m_df.loc[(m_df.Main_Block == 'VisibleTarget_2'),'Main_Block']='Target_is_Visible'
m_df.loc[(m_df.Main_Block == 'HiddenTarget_1'),'Main_Block']='Target_is_Hidden'
m_df.loc[(m_df.Main_Block == 'HiddenTarget_2'),'Main_Block']='Target_is_Hidden'
m_df.loc[(m_df.Main_Block == 'HiddenTarget_3'),'Main_Block']='Target_is_Hidden'

r_df = m_df.loc[ m_df['Main_Block']!='Non_relevant'] # Seleccionamos solo los relevantes
rNI_df = r_df.loc[ r_df['Modalidad']=='No Inmersivo'] # rNI_df es la base para lo No Inmersivo

#-------------------------------------------------------------------------------
# Realizaremos estudio de caso a caso.

PXX_List = rNI_df['Sujeto'].unique()  # Obtenemos una lista de todos los sujetos individualizados

for Pxx in PXX_List:     # Vamos ahora sujeto por sujeto.
    print(Pxx)
    show_df = rNI_df.loc[rNI_df['Sujeto']==Pxx]
    ax = sns.countplot(x='True_Block',hue='True_Trial', data= show_df).set(title=('Conteo de Trials en ' + str(Pxx)))
    plt.show()
    print('Next')

#-------------------------------------------------------------------------------
# Ahora intentaremos replicar el estudio original

# Primero nos aseguramos que tengamos todos los trial en este grupo
Trial_Count = rNI_df['Sujeto'].value_counts(ascending = True)
# Y de inmediato tenemos un problema... no todos los sujetos tienen el mismo numero de Trials!
# PORQUE???
# Primero mejor realizaremos el estudio detallado por cada individuo por cada trial



show_df = m_df.loc[ m_df['Main_Block']!='Non_relevant']
g = sns.FacetGrid(show_df, row="Main_Block", col="Modalidad")
g.figure.set_size_inches(17,17)
g.map(sns.lineplot, "True_Trial", "CSE", "Grupo")
g.add_legend()
plt.show()

show_df = m_df.loc[ m_df['Main_Block']=='Target_is_Hidden']
g = sns.FacetGrid(show_df, row="True_Block", col="Modalidad")
g.figure.set_size_inches(17,17)
g.map(sns.lineplot, "True_Trial", "CSE", "Grupo")
g.add_legend()
plt.show()

#show_df = m_df.loc[ m_df['True_Block']=='VisibleTarget_1']
#show_df = m_df.loc[ m_df['Modalidad']=='No Inmersivo']

ax = sns.barplot(x='Sujeto',y='CSE',hue='Grupo', data=m_df)
ax.figure.set_size_inches(17,17)
plt.show()

g = sns.FacetGrid(m_df, row="True_Block", col="Modalidad")
g.figure.set_size_inches(17,17)

g.map(sns.lineplot, "True_Trial", "CSE", "Grupo")
g.add_legend()
plt.show()

show_df = m_df.loc[ m_df['Modalidad']=='No Inmersivo']
ax = sns.barplot(x="Grupo",y="CSE",data=show_df)
plt.show()

show_df = m_df.loc[ m_df['Modalidad']=='No Inmersivo']
show_df = show_df.loc[ m_df['Grupo']=='MPPP']
ax = sns.barplot(x="Sujeto",y="CSE",data=show_df)
plt.show()

#g=sns.FacetGrid(data=m_df,col='True_Block', hue='Modalidad')
#g.map(sns.barplot, 'Grupo', 'CSE', order=['Voluntario Sano', 'Vestibular', 'MPPP'])
#plt.show()

show_df = m_df.loc[m_df['True_Block']=='VisibleTarget_1']
ax = sns.barplot(x="Grupo",y="CSE", hue='Modalidad', order=['Voluntario Sano', 'Vestibular', 'MPPP'],data=show_df).set(title='Target Visible (Control)')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

show_df = m_df.loc[m_df['True_Block']=='VisibleTarget_2']
ax = sns.barplot(x="Grupo",y="CSE", hue='Modalidad', order=['Voluntario Sano', 'Vestibular', 'MPPP'],data=show_df).set(title='Target Visible (Control) 2')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

show_df = m_df.loc[m_df['True_Block']=='VisibleTarget_2']
show_df = show_df.loc[show_df['Grupo']=='MPPP']
ax = sns.barplot(x="Sujeto",y="CSE", hue='Modalidad',data=show_df).set(title='Target Visible (Control) 2')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

show_df = m_df.loc[m_df['True_Block']=='HiddenTarget_1']
ax = sns.barplot(x="Grupo",y="CSE", hue='Modalidad', order=['Voluntario Sano', 'Vestibular', 'MPPP'],data=show_df).set(title='Plataforma escondida (Desafío de navegación)')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

show_df = m_df.loc[m_df['True_Block']=='HiddenTarget_2']
ax = sns.barplot(x="Grupo",y="CSE", hue='Modalidad', order=['Voluntario Sano', 'Vestibular', 'MPPP'],data=show_df).set(title='Plataforma escondida (Desafío de navegación)')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

show_df = m_df.loc[m_df['True_Block']=='HiddenTarget_3']
ax = sns.barplot(x="Grupo",y="CSE", hue='Modalidad', order=['Voluntario Sano', 'Vestibular', 'MPPP'],data=show_df).set(title='Plataforma escondida (Desafío de navegación)')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

#ax = sns.barplot(x="True_Block",y="CSE", hue='Grupo',data=show_df)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#plt.show()


#ax = sns.barplot(x="Grupo",y="CSE", hue='Sujeto',data=show_df)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#plt.show()
#------------------------------
#   Micro script para obtener los valores máximos de la vuelta olímpica
#   Necesita una recorrida limpia de toda la piscina en circular
#t_df = m_df['P_position_x']
#max_x = t_df.max()
#print('Maximo: ',max_x)
#   Como resultado en nuestra versión de Simian obtuvimos que la piscina mide
#   280 unidades originales de radio.
#------------------------------


#GRAFICO PATH"
Subj='P06'
Mod ='No_Inmersivo'
Bloc='HiddenTarget_3'
Plat=True
show_df = m_df.loc[ (m_df['Sujeto']==Subj) & (m_df['Modalidad']==Mod) & (m_df['True_Block']==Bloc)]
ax = sns.lineplot(x="P_position_x", y="P_position_y",hue="True_Trial", data=show_df, linewidth=3, alpha = 0.8, legend='full', palette = sns.color_palette('Blues', as_cmap = True), sort= False) #
#sns.set_palette('Blues',6)

sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})
sns.set_style('whitegrid')
circle= plt.Circle((0,0),0.5, color='b',fill=False)
ax.add_artist(circle)
ax.set(ylim=(-0.535,0.535),xlim=(-0.535,0.535), aspect=1) #incluir el equivalente al cuadrado de la pieza completa.
#el 0.535 considera los 300 pixeles que creo que mide la caja de Simian desde punto centro, divido en los 560 pixeles de Pool Diamater
ax.set(xlabel = 'East-West (virtual units in Pool-Diameters)', ylabel= 'North-South (virtual units in Pool-Diameters)', title = 'Test-Run')
plt.xlabel('East-West (virtual units in Pool-Diameters)',fontsize=18)
plt.ylabel('North-South (virtual units in Pool-Diameters)',fontsize=18)
ax.figure.set_size_inches(7,7)
plt.xticks(np.arange(-0.5, 0.75, 0.25))
plt.yticks(np.arange(-0.5, 0.75, 0.25))
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)
if Plat:
    PSize = (100 / 560)
    rectA = plt.Rectangle((show_df['platformPosition.x'].iloc[0] - (PSize / 2), show_df['platformPosition.y'].iloc[0] - (PSize / 2)), PSize, PSize, linewidth=1, edgecolor='b',
                          facecolor='none')

    ax.add_artist(rectA)

plt.show()

Subj='P06'
Mod ='No Inmersivo'
Bloc='HiddenTarget_3'
Blocs=['FreeNav','Training','VisibleTarget_1','VisibleTarget_2','HiddenTarget_1','HiddenTarget_2','HiddenTarget_3']
count = 0
for B in Blocs:

    if count > 1:
        Plat=True
    else:
        Plat=False

    show_df = m_df.loc[ (m_df['Sujeto']==Subj) & (m_df['Modalidad']==Mod) & (m_df['True Block']==B)]
    ax=0
    ax = sns.lineplot(x="P_position_x", y="P_position_y",hue="True Trial", data=show_df, linewidth=3, alpha = 0.8, legend='full', sort= False)
    if count > 1:
        ax = sns.lineplot(x="P_position_x", y="P_position_y", hue="True Trial", data=show_df, linewidth=3, alpha=0.8, legend='full',palette = sns.color_palette('Blues', as_cmap = True), sort=False)
    sns.set_context("paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 18})
    sns.set_style('whitegrid')
    circle = plt.Circle((0, 0), 0.5, color='b', fill=False)
    ax.add_artist(circle)
    ax.set(ylim=(-0.535, 0.535), xlim=(-0.535, 0.535), aspect=1)  # incluir el equivalente al cuadrado de la pieza completa.
    # el 0.535 considera los 300 pixeles que creo que mide la caja de Simian desde punto centro, divido en los 560 pixeles de Pool Diamater
    ax.set(xlabel='East-West (virtual units in Pool-Diameters)', ylabel='North-South (virtual units in Pool-Diameters)', title=(Subj + '-' + Mod +'-'+ B))
    plt.xlabel('East-West (virtual units in Pool-Diameters)', fontsize=18)
    plt.ylabel('North-South (virtual units in Pool-Diameters)', fontsize=18)
    ax.figure.set_size_inches(7, 7)
    plt.xticks(np.arange(-0.5, 0.75, 0.25))
    plt.yticks(np.arange(-0.5, 0.75, 0.25))
    ax.tick_params(labelsize=13)
    ax.legend(frameon=False, loc='right', bbox_to_anchor=(1.3, 0.5), fontsize=13)
    if Plat:
        PSize= (100 / 560)
        rectA = plt.Rectangle((show_df['platformPosition.x'].iloc[0]-(PSize/2), show_df['platformPosition.y'].iloc[0]-(PSize/2)), PSize, PSize, linewidth=1, edgecolor='b',
                              facecolor='none')

        ax.add_artist(rectA)
    plt.show()
    count+=1

