
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

![image](../../outs/exercise_02/loss_plot.png)

### Discussion of the training process

Write your answer here

## Evaluation

### Evaluation metrics

Write your answer here

![image](../../outs/exercise_02/train_regression_plot.png)

![image](../../outs/exercise_02/validation_regression_plot.png)

![image](../../outs/exercise_02/test_regression_plot.png)

Metrics for each dataset is depicted: 

![image](../../outs/exercise_02/metrics.png)

### Evaluation results

Here you have examples of evaluation results for train, validation and test sets.

Example for train set:

![image](../../outs/exercise_02/train_data_points_plot.png)


Example for validation set:

![image](../../outs/exercise_02/validation_data_points_plot.png)


Example for test set:

![image](../../outs/exercise_02/test_data_points_plot.png)


### Discussion of the results

How the model solves the problem?
Lo que hicimos fue entrenar un MLP para aprender la relación x,y con ruido. Como la curva es no lineal, las ReLU ayudan a que la red siga bien esa forma.

Is there overfitting, underfitting or any other issues? 
No se ve overfitting: train y validation loss bajan casi juntas. Tampoco underfitting, porque las predicciones se pegan mucho a los valores reales en train/val/test.

How can we improve the model?
Parar antes (early stopping) porque la loss luego mejora muy poco, normalizar x para entrenar más estable, y ajustar neuronas/regularización solo si en otro dataset se empieza a separar train vs val.

How this model will generalize to new data?
Generaliza bien si los nuevos datos son parecidos (mismo rango y misma forma). En test se comporta similar a train/val, eso es buena señal.

## Design Feedback loops

Describe the process you have followed to improve the model and the evolution of performance of the model during the process.

Lo que hicimos fue: entrenar luego mirar curvas y plots luego ajustar (capacidad, epochs y/o escala) y volver a entrenar. La loss cae rápido al inicio y luego se estabiliza, indicando que el modelo aprende la tendencia principal temprano.

## Questions

Pleaser answer the following questions. Include graphs if necessary. Store the graphs in the `outs/exercise_02` folder.

### Which are the differences you found between previous model and this one?
Del ejercicio 1 al 2, pasamos de una base más simple a un modelo con más capacidad (MLP con ReLU) porque aquí la relación es más no lineal. Por eso el ajuste visual y las métricas salen mejores y consistentes.
![image](../../outs/exercise_01/train_data_points_plot.png)
![image](../../outs/exercise_02/train_data_points_plot.png)

### Does the model generalizes well to new data?
Sí, para datos similares. Train/val/test tienen resultados parecidos y las curvas de loss no se separan.






