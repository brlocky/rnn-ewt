import matplotlib.pyplot as plt  # Import the matplotlib.pyplot module


def print_plot_model_loss(output_filename, loss, val_loss):

    fig, ax1 = plt.subplots()

    color = 'tab:green'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Validation Loss', color=color)
    ax2.plot(val_loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Model loss')
    plt.savefig(output_filename)  # Save the plot as an image file
    plt.clf()  # Clear the current figure


def print_plot_prediction_close(filename, original, forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(original['Date'], original['Close'], label='Actual Close')
    plt.plot(forecast['Date'], forecast['Wave 1'], label='Wave 1')
    plt.title('Wave Classification')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(filename)
    plt.clf()


prediction_columns = ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']


def print_plot_prediction_waves(filename, original, forecast):
    plt.figure(figsize=(12, 6))

    # Create the primary y-axis for actual Close values
    ax1 = plt.gca()
    ax1.plot(original['Date'], original['Close'],
             label='Actual Close', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Value', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create the secondary y-axis for wave predictions
    ax2 = ax1.twinx()

    # Plot waves as bars on the secondary y-axis
    for idx, wave_col in enumerate(prediction_columns):
        ax2.bar(forecast['Date'], forecast[wave_col],
                label=wave_col, alpha=0.5)

    ax2.plot(forecast['Date'], forecast['Largest_Wave'],
             label='Largest Wave', linestyle='dashed', linewidth=2, color='red')

    ax2.set_ylabel('Wave Prediction', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Wave Classification')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Combine legends from both y-axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
