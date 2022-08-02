import matplotlib.pyplot as plt


def plot_probe(data,i,ind,sub,ax):
    if ind == 1:
        ax.plot(data)
        ax.set_title('Track_{}'.format(i))
        ax.set_ylabel(r'$\mu m$')
    else:
        if i!=0 and i%ind ==0:
            sub += 1
        if i<=8:
            ax[sub,i%ind].plot(data)
            ax[sub,i%ind].set_title('Track_{}'.format(i))

        if i==0:
            ax[sub,i%ind].set_ylabel(r'$\mu m$')

    return ax,sub

def find_plot_size(num_data):
    plot_ind = 1
    while True:
        if (plot_ind*plot_ind)>=num_data:
            break
        else:
            plot_ind += 1
    return plot_ind
