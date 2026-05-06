# Learn2Slither

Implementación modular del juego Snake con un agente de Q-learning que aprende a jugar por sí solo.

## Características

- Tablero fijo de 10x10.
- Dos manzanas verdes y una manzana roja.
- Snake inicial de longitud 3, colocada aleatoriamente y de forma contigua.
- Entrenamiento por episodios con Q-table.
- Exportación e importación de modelos.
- Interfaz gráfica con Pygame.
- Modo paso a paso y control de velocidad.

## Requisitos

```bash
python3 -m pip install -r requirements.txt
```

Si no tienes un backend gráfico disponible, puedes ejecutar el entrenamiento en modo sin interfaz con `--no-render`. El núcleo del proyecto sigue funcionando en ese modo.

## Uso

Entrenar 1000 sesiones con interfaz visible:

```bash
./snake -sessions 1000 -visual on -speed human
```

Entrenar sin interfaz para acelerar el proceso:

```bash
./snake -sessions 5000 -visual off -quiet
```

Entrenar con características heurísticas (recomendado, aprende más rápido):

```bash
./snake -sessions 100000 -visual off -quiet -heuristics -save models/qtable_100k.json
```

La interfaz gráfica requiere `pygame` instalado en el entorno.

Evaluar un modelo sin seguir aprendiendo:

```bash
./snake -load models/qtable_100k.json -sessions 100 -dontlearn -visual on -epsilon 0 -speed human
```

Modo paso a paso:

```bash
./snake -sessions 1 -step-by-step -visual on -speed human
```

## Archivos de modelo

Los modelos se guardan en formato JSON con la configuración, hiperparámetros y metadatos, incluyendo la Q-table completa.

Ejemplo:
```bash
./snake -sessions 100000 -visual off -save models/qtable_100k.json
```
