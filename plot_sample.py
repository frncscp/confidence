from keras_efficiency import Metrics
import matplotlib.pyplot as plt
import itertools

final_stats = {}

#if you have all your models inside the same folder, if you don't you can change it
model_dir = ['path/to/models'] 
modelos = os.listdir(model_dir)

positive_folders = {'folder1' : r"path/to/folder", 'folder2' : r"path/to/folder"}
negative_folders = {'folder3' : r"path/to/folder", 'folder4' : r"path/to/folder"}

folders = positive_folders.copy()
folders.update(negative_folders)

for modelo in modelos:
    final_stats[modelo] = {}
    for label, _ in folders.items():
        final_stats[modelo][label] = None

def execute_folder(final_stats, model_dir, folder_name, folder_dir, data_class, modelo):
    metricas = Metrics(os.path.join(model_dir, modelo), folder_dir, dataset_class= data_class, score_threshold= .8, avg_prediction_bias= 1.2)    
    final_stats[modelo][folder_name] = metricas.get_stats()

for modelo in modelos:
    for folder_name, folder_dir in positive_folders.items():
        execute_folder(final_stats, model_dir, folder_name, folder_dir, True, modelo)
    for folder_name, folder_dir in negative_folders.items():
        execute_folder(final_stats, model_dir, folder_name, folder_dir, False, modelo)

def plot_metrics(final_stats):
    metrics = ['avg_prediction', 'normalized_score', 'efficiency']
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'grey', 'pink', 'yellow', 'black', 'cyan', 'magenta', 'navy', 'olive', 'teal', 'coral', 'maroon', 'indigo', 'turquoise', 'violet', 'crimson', 'chartreuse', 'beige', 'lavender', 'tan']
    results = {}
    for metric in metrics:
        fig, ax = plt.subplots()
        color_index = 0
        model_avg_efficiency = {}
        for model_name, folder_stats in final_stats.items():
            x = []
            y = []
            avg_efficiency = 0
            for folder_name, metrics in folder_stats.items():
                x.append(folder_name)
                y.append(metrics[metric])
                avg_efficiency += metrics[metric]
            avg_efficiency /= len(folder_stats)
            model_avg_efficiency[model_name] = avg_efficiency
            ax.plot(x, y, c=colors[color_index], label=model_name)
            color_index = (color_index + 1) % len(colors)
        results[metric] = model_avg_efficiency
        ax.legend(fontsize=4)
        ax.set_xlabel("Folders")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric.title()} for Different Folders and Models")
        plt.xticks(rotation=90)
        #plt.savefig(os.path.join('path/to','graph-for-{metric}-(1.1 to 1.10).png'), dpi=500, bbox_inches='tight')
        plt.ylim(0, 1)
        plt.show()
    return results
        
plot_metrics(final_stats)
