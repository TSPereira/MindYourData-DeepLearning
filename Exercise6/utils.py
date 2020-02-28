import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')


def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        # Display Image
        h = ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                      cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')


def plot_loss(train_loss, val_loss=None, ma_n_periods=5):
    fig, ax = plt.subplots(figsize=(20, 10))
    losses = (train_loss,) if val_loss is None else (train_loss, val_loss)
    losses_colors = ('b-', 'r-')
    losses_names = ('train', 'val')

    _ylim_max = None
    _ylim_min = 0
    for name, loss, color in zip(losses_names, losses, losses_colors):
        epochs = tuple(range(1, len(loss) + 1))

        # Calculate moving averages and plot limits
        ma_loss = moving_average(loss, n_periods=ma_n_periods)
        _ylim_min = 10 ** (np.floor(np.log10(min(loss))))

        diff_mag = np.floor(np.log10(np.abs(loss[0]-loss[-1])))
        if diff_mag > 1:
            _ylim_max = 10 ** (np.floor(np.log10(np.array(loss).mean())) + 1)

        # plot losses
        ax.plot(epochs, np.array(loss), color, label=name, alpha=0.3)
        ax.plot(epochs, np.array(ma_loss), color, label=f'{name} (avg {ma_n_periods})', alpha=0.8)

    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')

    # set axes limits
    _cur_lims = ax.get_ylim()
    if (_ylim_max is not None) and (_ylim_max < _cur_lims[1]):
        ax.set_ylim(_ylim_min, _ylim_max)

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()

    # add legend
    ax.legend(lines, labels, loc=0, frameon=False)
    fig.suptitle('Loss evolution over epochs')

    plt.show()


def moving_average(y_values, n_periods=20, exclude_zeros=False):
    """Calculate the moving average for one line (given as two lists, one
    for its x-values and one for its y-values).

    :param list|tuple|np.ndarray y_values: y-coordinate of each value.
    :param int n_periods: number of x values to use
    :return list result_y: result_y are the y-values of the line averaged for the previous n_periods.
    """

    # sanity checks
    assert isinstance(y_values, (list, tuple, np.ndarray))
    assert isinstance(n_periods, int)

    result_y, last_ys = [], []
    running_sum = 0
    # use a running sum here instead of avg(), should be slightly faster
    for y_val in y_values:
        if not (exclude_zeros & (y_val == 0)):
            last_ys.append(y_val)
            running_sum += y_val
            if len(last_ys) > n_periods:
                poped_y = last_ys.pop(0)
                running_sum -= poped_y
        result_y.append((float(running_sum) / float(len(last_ys))) if last_ys else 0)

    return result_y
