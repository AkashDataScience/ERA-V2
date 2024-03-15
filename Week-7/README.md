# Step 1
## Target:
1. Get Setup Right
2. Write light skeleton of our model
3. Add batch normalization

## Results:
1. Parameters: 6140
2. Best train accuracy: 99.40
3. Best test accuracy: 99.06

## Analysis:
1. In last epoch difference between train and test accuracy is high. That means, model is over fitting. Adding regularisation might help.
2. Model is stuck around 99. We have to increase model capacity.
3. As we are increasing model capacity, adding GAP might be a good idea.

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
1. The model is under fitting.
2. Adding slight rotation and other image augmentation might be helpfull.
3. Playing with learning rate might help with under fitting.

# Step 3
## Target:
1. Add image augmentation
2. Try different learning rate and step size in scheduler

## Results:
1. Parameters: 7760
2. Best train accuracy: 98.65
3. Best test accuracy: 99.45

## Analysis:
1. This model achieved 99.45 accuracy on test data in 15 epochs.
2. Model statisfies all criterion required for this assignment.
3. Using grid search for LR and step_size might be a good idea.