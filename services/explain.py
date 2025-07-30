from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='f1_macro'
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('learning_curve.png')  # Saves to disk
    print("Saved learning_curve.png")