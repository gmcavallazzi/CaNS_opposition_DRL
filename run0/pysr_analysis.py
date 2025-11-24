import numpy as np
import argparse
import os
from pysr import PySRRegressor

def main():
    parser = argparse.ArgumentParser(description='Run PySR analysis on generated dataset')
    parser.add_argument('--input', type=str, default='pysr_dataset.npz', help='Input dataset path')
    parser.add_argument('--equations', type=int, default=10, help='Number of equations to output')
    parser.add_argument('--time', type=float, default=3600, help='Time limit in seconds')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Dataset {args.input} not found.")
        return

    # Load data
    print(f"Loading data from {args.input}...")
    data = np.load(args.input)
    X = data['X']
    y = data['y']
    feature_names = list(data['feature_names'])
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features: {feature_names}")

    # Initialize PySR
    model = PySRRegressor(
        niterations=100,  # Run for 100 iterations (or until timeout)
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "sin",
            "cos",
            "exp",
            "tanh",  # Very important for saturation!
            "square",
            "cube"
        ],
        model_selection="best",  # Select best model based on score
        timeout_in_seconds=args.time,
        maxsize=20,  # Limit complexity
    )

    # Fit model
    print("Starting symbolic regression search...")
    model.fit(X, y, variable_names=feature_names)

    # Output results
    print("\n" + "="*50)
    print("Top Equations Found:")
    print("="*50)
    
    print(model.equations_)
    
    # Save best equation
    best_eq = model.sympy()
    print(f"\nBest Equation (SymPy): {best_eq}")
    
    # Save model
    model_path = "pysr_model.pkl"
    print(f"Saving model to {model_path}")
    # Note: PySR pickle saving might be version dependent, 
    # but the csv of equations is always saved by default to `hall_of_fame.csv`

if __name__ == "__main__":
    main()
