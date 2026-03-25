
from scipy import stats
import pandas as pd

def describe_df(df):
    '''
    Descripción: Para cada columna del dataframe, devuelve el tipo de variable, el
    porcentaje de nulos, los valores únicos y porcentaje de cardinalidad.

    Argumentos:
    df (dataframe)

    Retorna:
    dataframe: dataframe con las columnas del dataframe original, y con las filas
    de los tipos de variables, el tanto por ciento de nulos, los valores únicos
    y el porcentaje de cardinalidad.
    '''

    # Data_type
    data_type = df.dtypes

    # Nulos
    nulos = df.isna().sum()
    porcentaje_nulos = (nulos / len(df) * 100).round(2)

    # Valores únicos
    unicos = df.nunique()

    # Cardinalidad
    card = (unicos / len(df) * 100).round(2)

    # Creación tabla
    tabla = pd.DataFrame({
        "DATA_TYPE": data_type,
        "MISSINGS (%)": porcentaje_nulos,
        "UNIQUE_VALUES": unicos,
        "CARDIN (%)": card
    },
    index = pd.Index(df.columns, name = "COL_N"))

    return tabla.T # devuelve la traspuesta
 

def tipifica_variables(df, umbral_categoria, umbral_continua):
    '''
    Descripción:
    Clasifica las variables del dataframe df en Binaria, Categórica, Numérica continua
    o Numérica Discreta, según el siguiente criterio:
    - Si la cardinalidad es 2, asignara "Binaria"
    - Si la cardinalidad es menor que `umbral_categoria` asignara "Categórica"
    - Si la cardinalidad es mayor o igual que `umbral_categoria`, entonces entra en juego el tercer argumento:
        * Si además el porcentaje de cardinalidad es superior o igual a `umbral_continua`, asigna "Numerica Continua"
        * En caso contrario, asigna "Numerica Discreta"

    Argumentos:
    df (dataframe)
    umbral_categoria (entero): para determinar si una variable es categórica
    umbral_continua (float): para determinar si una variable numérica es continua o discreta

    Retorna:
    dataframe: dataframe con dos columnas "nombre_variable" y "tipo_sugerido".
    '''

    # 

    df_descripcion = describe_df(df).T # aprovechamos la función describe_df que ya calcula la cardinalidad y hacemos la transpuesta, para que los nombres de
                                     # las variables sean las filas y no las columnas

    # Corrección para cuando solo tengo un valor
    df_descripcion.loc[df_descripcion["UNIQUE_VALUES"] == 1, "CARDIN (%)"] = 0.00

    # Creamos una nueva columna "tipo_sugerido", partiendo de que todas son categóricas por ejemplo
    df_descripcion["tipo_sugerido"] = "Categórica"

    # Clasificamos
    df_descripcion.loc[df_descripcion["UNIQUE_VALUES"] == 2, "tipo_sugerido"] = "Binaria"
    df_descripcion.loc[df_descripcion["UNIQUE_VALUES"] >= umbral_categoria, "tipo_sugerido"] = "Numérica discreta"
    df_descripcion.loc[df_descripcion["CARDIN (%)"] >= umbral_continua, "tipo_sugerido"] = "Numérica continua"

    # Reseteamos el index para que pase a ser la columna "nombre_variable"
    df_descripcion = df_descripcion.reset_index(names = ["nombre_variable"])

    # Eliminamos columnas dejando solo "nombre_variable" y "tipo_sugerido"
    df_descripcion = df_descripcion.drop(columns = ["DATA_TYPE", "MISSINGS (%)", "UNIQUE_VALUES", "CARDIN (%)"])

    return df_descripcion

from scipy import stats
import pandas as pd
def get_features_num_regression(df, target_col, umbral_corr, pvalue = None):
    '''
    Descripción:
    Selecciona columnas numéricas que tengan una relación fuerte con la variable objetivo. 
    Solo se eligen las que superan el umbral de correlación y opcionalmente las que tienen un p-valor significativo.

    Argumentos: 
    df (dataframe)
    target_col (columna): columna de df, que debería ser el target de un hipotético modelo de regresión. Es una variable numérica con alta cardinalidad.
    umbral_corr (float): debe estar entre 0 y 1.
    pvalue (float, por defecto None): 

    Retorna:
    Lista con los nombres de las columnas numéricas seleccionadas que cumplen los criterios de correlación y, si aplica, de p-valor.
    En caso de error en los parámetros, retorna None.

    '''
    # 1. Comprobaciones básicas

    # Comprobamos que target_col exista en el dataframe  
    if target_col not in df.columns:
        print(f"ERROR: la columna '{target_col}' no existe en el DataFrame.")
        return None

    # Comprobamos que target_col es numérica
    if df[target_col].dtype not in ["int64", "float64"]:
        print("ERROR: 'target_col' debe ser una columna numérica.")
        return None
    
    # Comprobamos que umbral_corr está entre 0 y 1
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("ERROR: 'umbral_corr' debe ser un número entre 0 y 1.")
        return None
    
    # Comprobamos pvalue si no es None
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
            print("ERROR: 'pvalue' debe ser un número entre 0 y 1 o None.")
            return None

     # 2. Seleccionar columnas numéricas 
    
    columnas_numericas = []

    for col in df.columns:
        # añadimos solo columnas numéricas y que no sean el target
        if df[col].dtype in ["int64", "float64"] and col != target_col:
            columnas_numericas.append(col)

     # Si no hay columnas numéricas, devolvemos lista vacía
    if len(columnas_numericas) == 0:
        print("No hay columnas numéricas aparte del target.")
        return []

    columnas_seleccionadas = []
        
    # 3. Calcular correlaciones y p-valores
   
    for col in columnas_numericas:

        # Seleccionamos target + columna actual y quitamos filas con nulos
        datos = df[[target_col, col]].dropna()

        # evitar columnas con muy pocos datos
        if len(datos) < 3:
            continue
        # calculamos la correlación entre el target y una variable numérica,
        try:    
            # Cálculo de correlación y p-valor
            # stats.pearsonr(x, y) devuelve:
            #   - correlación entre x e y
            #   - p-valor del test de hipótesis
            corr, p_val = stats.pearsonr(datos[target_col], datos[col])
            
        except:
            continue

    # 4. Filtro por correlación
     
        if abs(corr) <= umbral_corr:
            continue

    # 5. Filtro por p-valor

        if pvalue is not None:
            if p_val > pvalue:
                continue

        columnas_seleccionadas.append(col)


    return columnas_seleccionadas


from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_features_num_regression(df, target_col, columns, umbral_corr, pvalue):
    '''
    Descripción: 
    Selecciona las columnas numéricas del dataframe cuya relación con la variable objetivo supera un umbral indicado. Opcionalmente, 
    también filtra por significación estadística si se proporciona un valor p.

    Argumentos:
        df (DataFrame): 
            Conjunto de datos que contiene la variable objetivo y las 
            variables numéricas a analizar.
        target_col (str): 
            Nombre de la columna objetivo. Debe ser numérica continua 
            o discreta con alta cardinalidad.
        umbral_corr (float): 
            Valor entre 0 y 1 que define la relación mínima exigida para 
            considerar una variable relevante.
        pvalue (float o None): 
            Umbral de significación estadística. Si es None, no se aplica 
            este filtro adicional.

    Retorna:
        list: 
            Lista de nombres de columnas numéricas que cumplen los criterios 
            establecidos. Devuelve None si los argumentos no son válidos.
    '''
            

     #  COMPROBACIONES BÁSICAS
   
    # Comprobamos el target en el DataFrame
    if target_col not in df.columns:
        print("ERROR: la columna objetivo no existe.")
        return None

    # Comprobamos si el El target es numérico
    if df[target_col].dtype not in ["int64", "float64"]:
        print("ERROR: el target debe ser numérico.")
        return None

    # Comprobamos el umbral
    if not (0 <= umbral_corr <= 1):
        print("ERROR: umbral_corr debe estar entre 0 y 1.")
        return None

    # Comprobamos si el pvalue es válido
    if pvalue is not None:
        if not (0 < pvalue < 1):
            print("ERROR: pvalue debe ser un número entre 0 y 1 o None.")
            return None

    #2. Usaremos la columnas numéricas
    if columns == []:
        
        columnas_numericas = [
            col for col in df.columns
            if df[col].dtype in ["int64", "float64"] and col != target_col
        ]
    else:
        
        columnas_numericas = [
            col for col in columns
            if col in df.columns and df[col].dtype in ["int64", "float64"]
        ]

    # Si no hay columnas numéricas:
    if len(columnas_numericas) == 0:
        print("No hay columnas numéricas válidas para representar.")
        return []

    # 3. Usamos get_features_num_regression 
    # Creamos un DF reducido solo con target + columnas numéricas
    df_reducido = df[[target_col] + columnas_numericas]

    # Seleccionar variables relevantes
    seleccionadas = get_features_num_regression(
        df_reducido,
        target_col,
        umbral_corr,
        pvalue
    )

    # Si no hay variables seleccionadas:
    if seleccionadas is None:
        return None

    if len(seleccionadas) == 0:
        print("Ninguna variable numérica supera el umbral de correlación.")
        return []

    # 4. Realizamos los gráficos
       
    max_cols_plot = 5

    for i in range(0, len(seleccionadas), max_cols_plot - 1):       
        batch = seleccionadas[i:i + (max_cols_plot - 1)]

        cols_plot = [target_col] + batch

        # Creamos el pairplot
        sns.pairplot(df[cols_plot].dropna())
        plt.show()

  
    return seleccionadas




import pandas as pd
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest


def get_features_cat_regression(df, target_col, pvalue = 0.05):
    '''
    Descripción: esta función selecciona las columnas categóricas del dataframe cuya relación con la variable objetivo cumple ciertos
    criterios estadísticos y elige el tipo de test estadístico adecuado según el número de categorías.
    Argumentos:
    df (dataframe)
    target_col (columna): la columna objetivo, que debe ser numérica continua o discreta con alta cardinalidad.
    pvalue (float): entre 0 y 1.

    Retorna: una lista con los nombres de las columnas categóricas seleccionadas que cumplen los criterios de regresión.
    
    '''

    # primero comprueba si cada argumento que recibe es el indicado: 
    if not isinstance(df, (pd.DataFrame)):  # el tipo de objeto df debe ser un df; si no, printa el error
        print("Error: df debe ser un DataFrame de pandas")
        return None
    
    if not isinstance(target_col, str): # el tipo del objeto target_col debe ser str; si no, printa el error
        print("Error: target_col debe ser una cadena de texto")
        return None
    
    if target_col not in df.columns: # si la col target no está en las columnas del df, printa error
        print(f"Error: target_col '{target_col}' no existe en el dataframe")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]): # pd.api.types = función para comprobar el tipo de dato de una columna
        print("Error: target_col debe ser numérica para regresión") # si el tipo de dato de la col target_col no es de tipo numérico, printa error
        return None
    
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1): # si el tipo de objeto pvalue no es de tipo int/float o pvalue no está entre 0 y 1,
        print("Error: pvalue debe ser un número entre 0 y 1") # printa error
        return None
    
    # luego, vamos a buscar las variables categóricas:
    def get_cat_columns(df, target_col):
        cat_cols = [] # creamos la lista vacía donde guardarlas

        for col in df.columns:
            if col != target_col and df[col].dtype == 'object': # recorremos las columnas
                cat_cols.append(col) # y si cumple lo que necesitamos, la apenda a nuestra lista vacía
        # aquí simplemente nos las devuelve:
        if cat_cols: # si hay valores en la lista,
            return cat_cols # nos las devuelve.
        else: # si no hay, printa que no:
            print("No hay variables categóricas para analizar")
            return None
        
    cat_cols = get_cat_columns(df, target_col) # aquí llama a mi función y la guarda en cat_cols
    if cat_cols is None: # y comprueba que si no hay ninguna columna categórica
        print("No hay ninguna columna categórica") # nos avisa
        return None # y termina la función
    
    # ahora vamos a iterar sobre cada columna categórica:
    cols_seleccionadas = [] # creamos nuestra lista vacía de las que nos valdrán

    for col in cat_cols:
        data = df[[col, target_col]].dropna()  # eliminamos filas con NaN
        grupos = [data[target_col][data[col] == cat] for cat in data[col].unique()]
        n_cats = len(grupos)

        if n_cats < 2:
            # solo hay una categoría, no se puede testear
            continue

        try:
            if n_cats == 2:
                # t-test de Welch
                stat, p_val = stats.ttest_ind(grupos[0], grupos[1], equal_var=False)
            else:
                # ANOVA de un factor
                stat, p_val = stats.f_oneway(*grupos)
        except Exception as e:
            print(f"Advertencia: no se pudo evaluar la columna '{col}'. Motivo: {e}")
            continue

    if p_val < pvalue:
        cols_seleccionadas.append(col)

    return cols_seleccionadas




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest



def plot_features_cat_regression(df, target_col = "", columns = [], pvalue = 0.05, with_individual_plot = False):
    '''
    Descripción: esta función selecciona las columnas categóricas del dataframe cuya relación con la variable objetivo cumple
    ciertos criterios estadísticos, elige el tipo de test estadístico adecuado según el número de categorías, y genera histogramas
    del target para cada categoría de las columnas seleccionadas.

    Argumentos:
    df (dataframe)
    target_col (columna): la columna objetivo, que debe ser numérica continua o discreta con alta cardinalidad.
    pvalue (float): entre 0 y 1.

    Retorna: una lista con los nombres de las columnas categóricas seleccionadas que cumplen los criterios de regresión.
    Y también genera gráficos de distribución del target para cada categoría de las columnas seleccionadas.
    '''
    # igual que nuestra otra función, con sus comprobaciones de entrada:     
    if not isinstance(df, (pd.DataFrame)):
        print("Error: df debe ser un DataFrame de pandas")
        return None
    
    if not isinstance(target_col, str):
        print("Error: target_col debe ser una cadena de texto")
        return None
    
    if target_col not in df.columns:
        print(f"Error: target_col '{target_col}' no existe en el dataframe")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser numérica para regresión")
        return None
    
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
        print("Error: pvalue debe ser un número entre 0 y 1")
        return None
    
    if not isinstance(with_individual_plot, bool):
        print("Error: with_individual_plot debe ser True o False")
        return None

    # luego, vamos a buscar las variables categóricas:
    def get_cat_columns(df, target_col):
        cat_cols = []
        for col in df.columns:
            if col != target_col and df[col].dtype == 'object':
                cat_cols.append(col)
        if cat_cols:
            return cat_cols
        else:
            print("No hay variables categóricas para analizar")
            return None
        
    cat_cols = get_cat_columns(df, target_col)
    if cat_cols is None:
        print("No hay ninguna columna categórica")
        return None

# y aquí va a iterar sobre la col y aplicar los tests:    
    cols_seleccionadas = []

    for col in cat_cols:
        data = df[[col, target_col]].dropna()  # eliminamos filas con NaN
        grupos = [data[target_col][data[col] == cat] for cat in data[col].unique()] # creamos la lista de listas
        n_cats = len(grupos) # número de categorías

        if n_cats < 2: # solo hay una categoría, no se puede testear
            continue

        try:
            if n_cats == 2: 
                # t-test:
                stat, p_val = stats.ttest_ind(grupos[0], grupos[1], equal_var=False)
            else:
                # test ANOVA de un factor
                stat, p_val = stats.f_oneway(*grupos) # el * es para desempaquetar la lista de listas
        except Exception as e: # si hay algún error en el test, lo captura y avisa
            print(f"Advertencia: no se pudo evaluar la columna '{col}'. Motivo: {e}") 
            continue

        if p_val < pvalue:
            
            cols_seleccionadas.append(col)
            # y el histograma individual:
            if with_individual_plot:
                plt.figure(figsize=(8, 5))
                sns.histplot(data=df, x=target_col, hue=col, multiple="stack", palette="Set2")
                plt.title(f"Histograma de {target_col} por {col}")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.show()

    if not cols_seleccionadas: # si la lista está vacía
        print("Ninguna columna pasó el test de significancia")
        return [] # lista vacía si no hay ninguna columna seleccionada

    return cols_seleccionadas




