import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pandas.api.types import CategoricalDtype
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score


def compare_confusion_matrices(class1, 
                               class2, 
                               plot1_title = fr"$PS_{1}$ Pose Multi.",
                               plot2_title = r"$PS_{10}$ Pose Multi.",
                               save_fig = False,
                               fig_path = 'pose_multi_cm.pdf'):
    
    # Compute confusion matrices
    cm1 = confusion_matrix(class1.y_test, class1.test_predictions)
    cm2 = confusion_matrix(class2.y_test, class2.test_predictions)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 10), constrained_layout=True)
    
    # Panel (a)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=class1.class_names)
    disp1.plot(cmap="Blues", xticks_rotation=45, ax=axes[0], colorbar=False)
    axes[0].set_title(plot1_title)
    axes[0].text(-0.1, 1.02, "(a)", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontweight="bold")
    
    # Panel (b)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=class1.class_names)
    disp2.plot(cmap="Blues", xticks_rotation=45, ax=axes[1], colorbar=False)
    axes[1].set_title(plot2_title)
    axes[1].text(-0.1, 1.02, "(b)", transform=axes[1].transAxes,
                 ha="left", va="bottom", fontweight="bold")
    
    plt.show()
    if save_fig:
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)


def compare_pr_curves(class1,
                      model1,
                      class2,
                      model2,
                      binary = True,
                      plot1_title = fr"Pose Precision-Recall Curve Multi. ($PS_{1}$)",
                      plot2_title = r"Pose Precision-Recall Curve Multi. ($PS_{10}$)",
                      save_fig = False,
                      fig_path = 'pose_multi_pr.pdf'):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    y_score1 = model1.predict_proba(class1.X_test)
    y_score2 = model2.predict_proba(class2.X_test)

    if binary:
        y_score_pos1 = y_score1[:, 1]
        precision1, recall1, _ = precision_recall_curve(class1.y_test, y_score_pos1)
        avg_precision1 = average_precision_score(class1.y_test, y_score_pos1)

        y_score_pos2 = y_score2[:, 1]
        precision2, recall2, _ = precision_recall_curve(class2.y_test, y_score_pos2)
        avg_precision2 = average_precision_score(class2.y_test, y_score_pos2)

        axes[0].plot(recall1, precision1, lw=2, label=f'AP = {avg_precision1:.2f}')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title(plot1_title)
        axes[0].legend(loc='lower left')
        axes[0].grid(True)
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlim([0.0, 1.0])
        axes[0].text(-0.1, 1.02, "(a)", transform=axes[0].transAxes,
                     ha="left", va="bottom", fontweight="bold")

        axes[1].plot(recall2, precision2, lw=2, label=f'AP = {avg_precision2:.2f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(plot2_title)
        axes[1].legend(loc='lower left')
        axes[1].grid(True)
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlim([0.0, 1.0])
        axes[1].text(-0.1, 1.02, "(b)", transform=axes[1].transAxes,
                     ha="left", va="bottom", fontweight="bold")
        
        plt.show()
        if save_fig:
            fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    else:
        class_names = ['None', 'Rock', 'Flap', 'Flap-Rock']
        classes = np.array([0, 1, 2, 3])
        
        y_test_bin1 = label_binarize(class1.y_test, classes=classes)
        precision1 = dict()
        recall1 = dict()
        avg_precision1 = dict()
        for i in range(len(classes)):
            precision1[i], recall1[i], _ = precision_recall_curve(y_test_bin1[:, i], y_score1[:, i])
            avg_precision1[i] = average_precision_score(y_test_bin1[:, i], y_score1[:, i])

        y_test_bin2 = label_binarize(class2.y_test, classes=classes)
        precision2 = dict()
        recall2 = dict()
        avg_precision2 = dict()
        for i in range(len(classes)):
            precision2[i], recall2[i], _ = precision_recall_curve(y_test_bin2[:, i], y_score2[:, i])
            avg_precision2[i] = average_precision_score(y_test_bin2[:, i], y_score2[:, i])
            
        for i in range(len(classes)):
            axes[0].plot(recall1[i], precision1[i], lw=2, label=f'{class_names[i]} (AP = {avg_precision1[i]:.2f})')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title(plot1_title)
        axes[0].legend(loc="lower left")
        axes[0].grid(True)
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlim([0.0, 1.0])
        axes[0].text(-0.1, 1.02, "(a)", transform=axes[0].transAxes,
                     ha="left", va="bottom", fontweight="bold")

        for i in range(len(classes)):
            axes[1].plot(recall2[i], precision2[i], lw=2, label=f'{class_names[i]} (AP = {avg_precision2[i]:.2f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(plot2_title)
        axes[1].legend(loc="lower left")
        axes[1].grid(True)
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlim([0.0, 1.0])
        axes[1].text(-0.1, 1.02, "(b)", transform=axes[1].transAxes,
                     ha="left", va="bottom", fontweight="bold")
        plt.show()
        if save_fig:
            fig.savefig(fig_path, bbox_inches="tight", dpi=300)

def sum_importances_per_sensor(importances, sensors, feature_names=None, n_per=10):
    if feature_names is None:
        assert len(importances) == len(sensors) * n_per
        return np.array([
            importances[i*n_per:(i+1)*n_per].sum()
        for i in range(len(sensors))])
    else:
        prefix = [n.split("_", 1)[0] for n in feature_names]
        sums = {s: 0.0 for s in sensors}
        for imp, pref in zip(importances, prefix):
            if pref in sums:
                sums[pref] += imp
        return np.array([sums[s] for s in sensors])


def compare_feature_importances(class1, 
                                model1,
                                class2,
                                model2,
                                plot1_title = fr"Pose $PS_{1}$ Binary - Feature Importances",
                                plot2_title = r"Pose $PS_{10}$ Binary - Feature Importances",
                                save_fig = False,
                                fig_path = 'pose_feature_importances.pdf'):

        
    if class1.input_modality == 'accelerometer':
        sensors = sensors1 = sensors2 = sensor_names2 = ["Torso", "LWrist", "RWrist"]
        feature_names_10 = [f"Torso_{i}" for i in range(1, 11)] + [f"LWrist_{i}" for i in range(1, 11)] + [f"RWrist_{i}" for i in range(1, 11)]
    else:
        sensors = sensors1 = sensors2 = sensor_names2 = sensor_names = ["Head", "RWrist", "LWrist", "RShoulder", "LShoulder", "Chest", "Head_Accel", "RWrist_Accel", "LWrist_Accel", "RShoulder_Accel", "LShoulder_Accel", "Chest_Accel"]

        feature_names_10 = [f"Head_{i}" for i in range(1, 11)] + \
                [f"RWrist_{i}" for i in range(1, 11)] + \
                [f"LWrist_{i}" for i in range(1, 11)] + \
                [f"RShoulder_{i}" for i in range(1, 11)] + \
                [f"LShoulder_{i}" for i in range(1, 11)] + \
                [f"Chest_{i}" for i in range(1, 11)] + \
                [f"Head_Accel_{i}" for i in range(1, 11)] + \
                [f"RWrist_Accel_{i}" for i in range(1, 11)] + \
                [f"LWrist_Accel_{i}" for i in range(1, 11)] + \
                [f"RShoulder_Accel_{i}" for i in range(1, 11)] + \
                [f"LShoulder_Accel_{i}" for i in range(1, 11)] + \
                [f"Chest_Accel_{i}" for i in range(1, 11)]

    if class1.optimize_feature_space:
        feature_mask1 = class1.best_feature_selection
        sensors1 = [s for s, keep in zip(sensor_names, feature_mask1) if keep]
        
    if class2.optimize_feature_space:
        feature_mask2 = class2.best_feature_selection
        sensors2 = [s for s, keep in zip(sensors, feature_mask2) if keep]
        sensor_names2 =  [s for s, keep in zip(sensor_names, feature_mask2) if keep]
        feature_names_10 = [f for f in feature_names_10 if any(f.startswith(p) for p in sensors2)]

    imp1 = model1.feature_importances_    
    
    imp2_sum = sum_importances_per_sensor(
        model2.feature_importances_, sensors2, feature_names=feature_names_10, n_per=10
    )
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), constrained_layout=True, sharey=False)

    # (a) model1
    axes[0].bar(np.arange(len(sensors1)), imp1)
    axes[0].set_xticks(np.arange(len(sensors1)))
    axes[0].set_xticklabels(sensors1, rotation=45)
    axes[0].set_ylabel("Importances", family = "Times New Roman", fontsize = 15)
    axes[0].set_title(plot1_title, family = "Times New Roman", fontsize = 20)
    axes[0].text(-0.05, 1.02, "(a)", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontweight="bold")
    
    # (b) model2
    axes[1].bar(np.arange(len(sensors2)), imp2_sum)
    axes[1].set_xticks(np.arange(len(sensors2)))
    axes[1].set_xticklabels(sensor_names2, rotation=45)
    axes[1].set_ylabel("Summed Importances", family = "Times New Roman", fontsize = 15)
    
    axes[1].set_title(plot2_title,  family = "Times New Roman", fontsize = 20)
    axes[1].text(-0.05, 1.02, "(b)", transform=axes[1].transAxes,
                 ha="left", va="bottom", fontweight="bold")
    
    plt.show()
    if save_fig:
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)



def explained_variance_plots(X_features):
    
    pca = PCA(n_components=min(X_features.shape[1], 50)) 
    X_pca = pca.fit_transform(X_features)
    
    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label="95% Variance Threshold")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.show()

def umap_reduction(X_features, Feature_extraction_class):
    umap_reducer = umap.UMAP(n_components = 3, random_state = 42, n_jobs = -1)
    X_umap = umap_reducer.fit_transform(X_features)
    df_umap = pd.DataFrame(X_umap, columns = ['UMAP1', 'UMAP2', 'UMAP3'])
    df_umap['PersonID'] = Feature_extraction_class.person_ids
    df_umap['Annotations'] = Feature_extraction_class.annotations
    return df_umap


def plot_interactive(df_umap_tda, color_by_anno=True, save_html=False, output_name = 'anno_tda'):

    if color_by_anno:
        color = df_umap_tda['Annotations'].astype(str)
    else:
        color = df_umap_tda['PersonID'].astype(str)

    fig = px.scatter_3d(df_umap_tda, x='UMAP1', y='UMAP2', z='UMAP3', 
                     color=color,  
                     title="Interactive TDA Features UMAP",
                     opacity=0.2)
    fig.update_traces(marker=dict(size=2)) 
    fig.show()
    if save_html:
        fig.write_html(f'{output_name}.html')


def rgb_string_to_tuple(rgb_string):
    parts = rgb_string.strip('rgb()').split(',')
    return tuple(int(p) / 255 for p in parts)


def plot_UMAP_comparison(df_umap, x_axis = 'UMAP1', y_axis = 'UMAP2', method = r'$PS_{1}$', save_fig = False, output_path = 'umap_figure.pdf'):
  
    plotly_safe_rgb = px.colors.qualitative.Safe
    plotly_safe_mpl = [rgb_string_to_tuple(c) for c in plotly_safe_rgb]
    
    label_map = {
        0: 'None',
        1: 'Rock',
        2: 'Flap',
        3: 'Flap Rock'
    }
    label_order = ['None', 'Rock', 'Flap', 'Flap Rock']
    color_sequence = ['#000000', '#E69F00', '#56B4E9', '#009E73']
    cat_type = CategoricalDtype(categories=label_order, ordered=True)
    
    # Subsample For Clean Look based on PersonID
    df_sampled = df_umap.groupby('PersonID').apply(lambda x: x.sample(frac=0.3)).reset_index(drop=True)
    
    df_sampled['PersonID'] = df_sampled['PersonID'].astype(str)
    df_sampled['Label'] = df_sampled['Annotations'].map(label_map).astype(cat_type)
    
    unique_pids = sorted(df_sampled['PersonID'].unique())
    color_map_pid = dict(zip(unique_pids, plotly_safe_mpl[:len(unique_pids)]))
    color_map_label = dict(zip(label_order, color_sequence))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel (a): Colored by Participant
    sns.scatterplot(
        data=df_sampled,
        x=x_axis,
        y=y_axis,
        hue='PersonID',
        palette=color_map_pid,
        s=30,
        alpha=0.6,
        linewidth=0,
        ax=ax1
    )
    ax1.set_title(f"2D UMAP Accelerometer Feature Space {method} – Colored by Child", fontsize=14, family="Times New Roman")
    ax1.set_xlabel(x_axis, fontsize=12, family="Times New Roman")
    ax1.set_ylabel(y_axis, fontsize=12, family="Times New Roman")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.text(-0.1, 1.05, "(a)", transform=ax1.transAxes, fontsize=14, fontweight="bold", family="Times New Roman")
    ax1.legend(title='Child', title_fontsize=10, fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'family': 'Times New Roman'})
    
    # Panel (b): Colored by Stereotypy
    sns.scatterplot(
        data=df_sampled,
        x=x_axis,
        y=y_axis,
        hue='Label',
        palette=color_map_label,
        s=30,
        alpha=0.6,
        linewidth=0,
        ax=ax2
    )
    ax2.set_title(f"2D UMAP Accelerometer Feature Space {method} – Colored by Stereotypy", fontsize=14, family="Times New Roman")
    ax2.set_xlabel(x_axis, fontsize=12, family="Times New Roman")
    ax2.set_ylabel(y_axis, fontsize=12, family="Times New Roman")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.text(-0.1, 1.05, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold", family="Times New Roman")
    ax2.legend(title='Class', title_fontsize=10, fontsize=9, prop={'family': 'Times New Roman'})
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()
