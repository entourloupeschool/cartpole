import matplotlib.pyplot as plt
import random as rdm

def plot_metrics(metrics, name, save=False, save_path=None, show=True):
    # Create a figure and a 1x2 grid of subplots
    plt.figure(figsize=(12, 6))

    # Plot loss
    x = [metric['episode'] for metric in metrics]

    # Plot total reward
    y_reward = [metric['total_reward'] for metric in metrics]
    plt.title("Total reward")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.plot(x, y_reward)
    
    if show:
        # Show the figure
        plt.show()
        
    # Save the figure if needed
    if save:
        plt.savefig(save_path +
                    'metrics'+name+str(x[-1])+'.png')

    # Close the figure
    
    plt.close()
