from matplotlib import rcParams


rcParams['font.family'] = 'DejaVu Sans'


from .tree_plot import tree_plots, confusion_matrix, roc_curve, metric_importance, decision_tree


__all__ = ['tree_plots', 'confusion_matrix', 'roc_curve', 'metric_importance', 'decision_tree']
