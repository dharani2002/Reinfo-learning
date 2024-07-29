import tensorflow as tf

def adjust_model_parameters(model, action):


    # Define action space (example: adjust learning rate and dropout rate)
    num_actions = 3  # Example: 3 actions - no change, increase, decrease
    learning_rate_range = [1e-4, 1e-2]
    dropout_range = [0.2, 0.5]
    
    # Convert action index to corresponding parameter adjustment
    learning_rate_action = action // 2  # 0: no change, 1: increase, 2: decrease
    dropout_action = action % 2         # 0: no change, 1: increase/decrease

    # Get current model parameters
    current_learning_rate = model.optimizer.learning_rate.numpy()
    current_dropout_rate = model.get_layer("dropout").rate

    # Adjust learning rate
    if learning_rate_action == 1:  # Increase learning rate
        new_learning_rate = min(current_learning_rate * 2, learning_rate_range[1])
    elif learning_rate_action == 2:  # Decrease learning rate
        new_learning_rate = max(current_learning_rate / 2, learning_rate_range[0])
    else:  # No change
        new_learning_rate = current_learning_rate

    # Adjust dropout rate
    if dropout_action == 1:  # Increase dropout rate
        new_dropout_rate = min(current_dropout_rate + 0.1, dropout_range[1])
    elif dropout_action == 2:  # Decrease dropout rate
        new_dropout_rate = max(current_dropout_rate - 0.1, dropout_range[0])
    else:  # No change
        new_dropout_rate = current_dropout_rate

    # Update model parameters
    model.optimizer.learning_rate.assign(new_learning_rate)
    model.get_layer("dropout").rate = new_dropout_rate

    return model
