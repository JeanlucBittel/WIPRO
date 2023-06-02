import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    
    def plot_loss(train, val):
        """
        train and val are dicts in the form of {'loss_name': [values]}
        """
        loss_functions = list(train.keys())
        epochs = range(1, len(train.get(loss_functions[0]))+1)
        len_loss_functions = len(loss_functions)

        plt.style.use('dark_background')
        plt.subplots(1, len_loss_functions, figsize=(8*len_loss_functions, 8))
        
        for i, loss in enumerate(loss_functions):
            
            plt.subplot(1, len_loss_functions, i+1)

            plt.title('Training and Validation Loss {}'.format(loss_functions[i]))
            plt.xticks(np.arange(1, epochs[-1]+1, 5))
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            
            plt.plot(np.arange(1, epochs[-1]+1), train.get(loss_functions[i]), label='Training Loss {}'.format(loss_functions[i]))
            plt.plot(np.arange(1, epochs[-1]+1), val.get(loss_functions[i]), label='Validation Loss {}'.format(loss_functions[i]))
            
            plt.legend(loc='best')

        plt.show()
        
    def plot_images(inputs, labels, masks, outputs):
        plt.style.use('dark_background')
        plt.subplots(1, 6, figsize=(16, 16))

        # T1
        plt.subplot(1, 6, 1)
        plt.xlabel('T1')
        plt.imshow(inputs[0,:,:,0], cmap='gray', vmin=0, vmax=1)
        plt.grid(False)

        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        # T1GD
        plt.subplot(1, 6, 2)
        plt.xlabel('T1GD')
        plt.imshow(inputs[1,:,:,0], cmap='gray', vmin=0, vmax=1)
        plt.grid(False)

        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        # T2
        plt.subplot(1, 6, 3)
        plt.xlabel('T2')
        plt.imshow(inputs[2,:,:,0], cmap='gray', vmin=0, vmax=1)
        plt.grid(False)

        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        # Real Flair
        plt.subplot(1, 6, 4)
        plt.xlabel('Real Flair')
        plt.imshow(labels[0,:,:,-1], cmap='gray', vmin=0, vmax=1)
        plt.grid(False)

        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        # Generated Flair
        plt.subplot(1, 6, 5)
        plt.xlabel('Generated Flair')
        plt.imshow(outputs[0,:,:], cmap='gray', vmin=0, vmax=1)
        plt.grid(False)

        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        # Mask
        plt.subplot(1, 6, 6)
        plt.xlabel('Mask')
        plt.imshow(masks[0,:,:,0], cmap='gray', vmin=0, vmax=1)
        plt.grid(False)

        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        plt.tight_layout()
        plt.show()