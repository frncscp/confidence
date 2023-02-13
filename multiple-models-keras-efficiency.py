from keras_efficiency import Metrics

final_stats = {}
model_dir = 'path/to/models'

positive_folders = {'folder1' : r"path/to/folder", 'folder2' : r"path/to/folder"}
negative_folders = {'folder3' : r"path/to/folder", 'folder4' : r"path/to/folder"}

folders = positive_folders.copy()
folders.update(negative_folders)

modelos = os.listdir(model_dir)

for modelo in modelos:
    final_stats[modelo] = {}
    for label, _ in folders.items():
        final_stats[modelo][label] = None

def execute_folder(final_stats, model_dir, folder_name, folder_dir, data_class, modelo):
    metricas = Metrics(os.path.join(model_dir, modelo), folder_dir, dataset_class= True, score_threshold= .8, avg_prediction_bias= 1.2)    
    final_stats[modelo][folder_name] = metricas.get_stats()

for modelo in modelos:
    for folder_name, folder_dir in positive_folders.items():
        execute_folder(final_stats, model_dir, folder_name, folder_dir, True, modelo)
    for folder_name, folder_dir in negative_folders.items():
        execute_folder(final_stats, model_dir, folder_name, folder_dir, False, modelo)
 
metrics = ['score', 'avg_prediction', 'model size', 'normalized_score', 'efficiency']

finalissima = {}
for modelo in modelos:
    finalissima[modelo] = {}
for modelo in modelos:  
    for folder, _ in final_stats[modelo].items():
        for metrica in metrics:
            finalissima[modelo][metrica] = 0
for modelo in modelos:
    for folder, _ in final_stats[modelo].items():
        for metrica in metrics:
            finalissima[modelo][metrica] += final_stats[modelo][folder][metrica]
for modelo in modelos:
    for folder, _ in final_stats[modelo].items():
        for metrica in metrics:
            finalissima[modelo][metrica] /= len(final_stats[modelo])         
            
#final stats = the dictionary returned from above, in this case it's called "finalissima"
#save image = True if user wants to save the image, it'll be saved with os.getcwd() (currrent directory)
#name = name of the image
#img_format = format of the image
#dict results: returns a dict with the average prediction, efficiency and score for all folders if True
#add more colors if you need to. I made this implementation so the models get the same color.

def plot_metrics(final_stats = finalissima, save_image = False, name = 'efficiency_graph', img_format = '.png', dict_results = False):
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
        if save_image:
            plt.savefig(os.path.join(os.getcwd(), name+img_format), dpi=500, bbox_inches='tight')
        plt.show()
    if dict_results: 
        return results 
