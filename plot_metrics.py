import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(results_path = 'runs/classify/train/results.csv'):
    # Read the results from the CSV file into a pandas DataFrame
    results = pd.read_csv(results_path)

    # Create a single window with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the first graph (Loss vs epochs)
    ax1.plot(results['                  epoch'], results['             train/loss'], label='train loss')
    ax1.plot(results['                  epoch'], results['               val/loss'], label='val loss', c='red')
    ax1.grid()
    ax1.set_title('Loss vs epochs')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax1.legend()

    # Plot the second graph (Validation accuracy vs epochs)
    ax2.plot(results['                  epoch'], results['  metrics/accuracy_top1'] * 100)
    ax2.grid()
    ax2.set_title('Validation accuracy vs epochs')
    ax2.set_ylabel('accuracy (%)')
    ax2.set_xlabel('epochs')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

# Example usage
plot_metrics()
