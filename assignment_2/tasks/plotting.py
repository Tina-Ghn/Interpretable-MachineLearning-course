import sys
sys.path.insert(0, "")

from utils.styled_plot import plt


def plot_bar(x, y, title=None, ylabel=None):
    """
    Displays a bar plot on given x and y.
    
    Inputs:
        x (list or np.ndarray): Labels. Each entry is associated to one bar.
        y (list or np.ndarray): Values for x. Same shape as x.
        title (str): Title.
        ylabel (str): Label for y-axis.
        
    Returns:
        plt (matplotlib.pyplot or utils.styled_plot.plt)
    """
    my_cmap = plt.get_cmap("viridis")
    plt.figure()

    # do something
    # ...
    plt.bar(x, y, width=0.4)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f"{title}.png")

    # show the figure (needs to be done after saving)
    # ...
    plt.show()
    return plt


if __name__ == "__main__":
    plt = plot_bar(
        x=["A", "B", "C"],
        y=[0.2, 0.5, 0.4],
        title="Test",
        ylabel="Test"
    )
    

    

    