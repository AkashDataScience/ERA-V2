# Step 1
## Target:
1. Get Setup Right
2. Write a light skeleton of our model
3. Add batch normalization

## Results:
1. Parameters: 6140
2. Best train accuracy: 99.40
3. Best test accuracy: 99.06

## Analysis:
1. In the last epoch, difference between train and test accuracy is high. That means the model is overfitting. Adding regularization might help.
2. The model is stuck around 99. We have to increase model capacity.
3. adding GAP might be a good idea as we increase model capacity.

# Step 2
## Target:
1. Add regularisation
2. Increase model capacity
3. Add GAP

## Results:
1. Parameters: 7760
2. Best train accuracy: 98.95
3. Best test accuracy: 99.22

## Analysis:
1. The model is underfitting.
2. Adding slight rotation and other image augmentation might be helpful.
3. Playing with learning rate might help with underfitting.

# Step 3
## Target:
1. Add image augmentation
2. Try different learning rates and step sizes in the scheduler

## Results:
1. Parameters: 7760
2. Best train accuracy: 98.65
3. Best test accuracy: 99.45

## Analysis:
1. This model achieved 99.45 accuracy on test data in 15 epochs.
2. The model satisfies all criteria required for this assignment.
3. Using grid search for LR and step_size might be a good idea.