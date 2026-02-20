from langchain_core.runnables import RunnableLambda

# Los Runnables son objetos que representan una función o una operación que se puede ejecutar. En este ejemplo, creamos dos Runnables: uno para generar un texto a partir de un número y otro para duplicar ese texto.

# El primer Runnable, `paso1`, utiliza una función lambda para convertir un número en una cadena de texto con el formato "Número {x}".
paso1 = RunnableLambda(lambda x: f"Número {x}")

# El segundo Runnable, `paso2`, utiliza una función definida previamente, `duplicar_texto`, que toma un texto y lo duplica en una lista.
def duplicar_texto(texto):
    return [texto] * 2

paso2 = RunnableLambda(duplicar_texto)

# Luego, combinamos ambos Runnables utilizando el operador `|`, lo que crea una nueva cadena de Runnables que ejecutará ambos pasos en secuencia. Cuando invocamos esta cadena con el número 43, primero se generará el texto "Número 43" y luego se duplicará en una lista.
cadena = paso1 | paso2

# Finalmente, invocamos la cadena de Runnables con el número 43 y almacenamos el resultado en la variable `resultado`.
resultado = cadena.invoke(43)

# Imprimimos el resultado, que será una lista con dos elementos, ambos siendo "Número 43".
print(resultado)  # Output: ['Número 43', 'Número 43']