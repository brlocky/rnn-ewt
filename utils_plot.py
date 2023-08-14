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


def print_plot_prediction_close(filename, original, forecast, feature):
    plt.figure(figsize=(12, 6))
    plt.plot(original['Date'], original[feature], label='Actual ' + feature)
    plt.plot(forecast['Date'], forecast[feature],
             label='Forecasted ' + feature)
    plt.title('Actual vs Forecasted ' + feature)
    plt.xlabel('Date')
    plt.ylabel(feature + ' Value')
    plt.legend()
    plt.savefig(filename)
    plt.clf()
