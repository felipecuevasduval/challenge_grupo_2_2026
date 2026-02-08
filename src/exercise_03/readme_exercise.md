
# Exercise 1: Learn a linear function with PyTorch

## Objective

Estimation of a unknown function by a machine learning model

## Task Formalization

Write your answer here

### Task Formalization (Inference)

Write your answer here
### Task Formalization (Training)

Write your answer here

## Evaluation metrics

Write your answer here

## Data Considerations

### Dataset description

Write your answer here

### Data preparation and preprocessing

Write your answer here

### Data augmentation

Write your answer here

## Model Considerations

Write your answer here

### Suitable Loss Functions

Write your answer here

### Selected Loss Function

Write your answer here

### Possible architectures

Write your answer here

### Last layer activation

Write your answer here

### Other Considerations

Write your answer here

## Training

Write your answer here

### Training hyperparameters

Write your answer here

### Loss function graph

![image](../../outs/exercise_03/loss_plot.png)

### Discussion of the training process

Write your answer here

## Evaluation

### Evaluation metrics

Write your answer here

![image](../../outs/exercise_03/train_regression_plot.png)

![image](../../outs/exercise_03/validation_regression_plot.png)

![image](../../outs/exercise_03/test_regression_plot.png)

Metrics for each dataset is depicted: 

![image](../../outs/exercise_03/metrics.png)

### Evaluation results

Here you have examples of evaluation results for train, validation and test sets.

Example for train set:

![image](../../outs/exercise_03/train_data_points_plot.png)


Example for validation set:

![image](../../outs/exercise_03/validation_data_points_plot.png)


Example for test set:

![image](../../outs/exercise_03/test_data_points_plot.png)


### Discussion of the results

How the model solves the problem?
Lo que hicimos fue entrenar una red (MLP) para que aprenda la relación x → y viendo muchos pares de entrada/salida. Las capas con ReLU le permiten capturar la forma no lineal, y la salida queda lineal (Identity) porque es regresión.

Is there overfitting, underfitting or any other issues? 
Nos fijamos en la curva de train vs validation loss (la que se guarda como loss_plot.png). Si ambas bajan parecidas y no se separan mucho, no hay overfitting fuerte. Si ambas se quedan altas, sería underfitting.

How can we improve the model?
Cambios simples que suelen ayudar: normalizar x, ajustar lr (un poco arriba/abajo), y bajar/subir neuronas o epochs según lo que se vea en la loss. Si empezara a sobreajustar, meter un poquito de regularización (weight decay) o parar antes (early stopping).

How this model will generalize to new data?
Generaliza bien mientras los datos nuevos se parezcan a los de entrenamiento (mismo rango y misma “forma” del problema). La señal principal es que el rendimiento en validation/test no caiga respecto a train (métricas en metrics.png).

## Design Feedback loops

Describe the process you have followed to improve the model and the evolution of performance of the model during the process.

Lo que hicimos fue pasar de tener un modelo que ya ajustaba bien la curva a hacerlo más “robusto” y consistente a nivel de entrenamiento y evaluación: mantuvimos la idea del MLP para capturar la no linealidad, pero mejoramos el pipeline para que todo fuera estable (normalización de la entrada para que el entrenamiento converja más suave, ajuste fino de hiperparámetros como learning rate/epochs/batch para no entrenar de más, y guardado del mejor checkpoint por validación). Además, en el ejercicio 3 dejamos cerrado el tema de CUDA para evitar el error típico de mezclar CPU y GPU en evaluate, moviendo siempre inputs/targets al mismo device del modelo y pasando a CPU solo al momento de graficar. Con eso, el rendimiento se mantuvo consistente entre train/validation/test y el flujo quedó más limpio para repetir experimentos sin problemas.

You can include a table stating the chanched parameters and the obtained results after the process.
| Cambio          | Antes                    | Ahora                                | Para qué lo hicimos                     |
| --------------- | ------------------------ | ------------------------------------ | --------------------------------------- |
| Entrada x     | Sin normalizar           | Normalizada (ej. x/100)              | Entrenar más estable y rápido           |
| Modelo          | Más simple               | MLP (3 capas, 2 ReLU, salida lineal) | Capturar mejor no linealidad            |
| Capacidad       | Menos neuronas           | Más neuronas (p.ej. 256)             | Mejor aproximación de la función        |
| Optimizer       | Adam                     | AdamW        | Mejor control del ajuste                |
| Batch size      | Menor                    | Mayor (p.ej. 256)                    | Mejor eficiencia (especialmente en GPU) |
| CUDA / evaluate | A veces mezclaba devices | Inputs/targets al device del modelo  | Evitar error CPU vs CUDA                |


## Questions

Pleaser answer the following questions. Include graphs if necessary. Store the graphs in the `outs/exercise_03` folder.

### Which are the differences you found between previous model and this one?
Lo que cambio fue principalmente la capacidad y la estabilidad: pasamos a un MLP con ReLU (mejor para no linealidad), con entrada normalizada. 

### Does the model generalizes well to new data?
Si, si los nuevos datos son del mismo tipo. Lo comprobamos comparando train/validation/test: si las métricas y los plots se mantienen parecidos entre splits, entonces generaliza razonablemente bien.





