import numpy as np
from scipy import stats
import pandas as pd

def calculate_auc_significance(dataset_name, mdfnet_results, baseline_results_dict, alpha=0.05):
    """
    计算单个数据集上MDF-Net与其他方法的AUC显著性比较
    
    参数:
    dataset_name: 数据集名称
    mdfnet_results: MDF-Net的AUC结果数组
    baseline_results_dict: 其他方法的AUC结果字典
    alpha: 显著性水平
    """
    results = []
    
    for method_name, baseline_results in baseline_results_dict.items():
        # Wilcoxon检验
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
            mdfnet_results, 
            baseline_results,
            alternative='greater'  # 验证MDF-Net是否显著优于baseline
        )
        
        # t检验
        t_stat, t_p = stats.ttest_rel(
            mdfnet_results,
            baseline_results,
            alternative='greater'
        )
        
        # 检查正态性
        _, normality_p = stats.shapiro(mdfnet_results - baseline_results)
        is_normal = normality_p > 0.05
        
        # 选择合适的p值
        p_value = t_p if is_normal else wilcoxon_p
        
        # 添加结果
        results.append({
            'Dataset': dataset_name,
            'Method': method_name,
            'p_value': p_value,
            'Test_Used': 't-test' if is_normal else 'Wilcoxon',
            'Significance': get_significance_stars(p_value)
        })
    
    return pd.DataFrame(results)

def get_significance_stars(p_value):
    """返回显著性星号标记"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return 'ns'

def format_latex_table(df):
    """将结果格式化为LaTeX表格"""
    latex_table = []
    latex_table.append("\\begin{table}[htbp]")
    latex_table.append("\\centering")
    latex_table.append("\\caption{Statistical Significance Analysis of AUC Scores}")
    latex_table.append("\\begin{tabular}{lccc}")
    latex_table.append("\\toprule")
    latex_table.append("Method & DRIVE & CHASE\\_DB1 & STARE \\\\")
    latex_table.append("\\midrule")
    
    # 获取所有方法
    methods = df['Method'].unique()
    
    for method in methods:
        row = [method]
        for dataset in ['DRIVE', 'CHASE_DB1', 'STARE']:
            result = df[(df['Method'] == method) & (df['Dataset'] == dataset)]
            if not result.empty:
                p_value = result.iloc[0]['p_value']
                stars = result.iloc[0]['Significance']
                row.append(f"{p_value:.4f}$^{{{stars}}}$" if stars != 'ns' else f"{p_value:.4f}")
        latex_table.append(" & ".join(row) + " \\\\")
    
    latex_table.append("\\bottomrule")
    latex_table.append("\\end{tabular}")
    latex_table.append("\\\\[2pt]")
    latex_table.append("\\raggedright *** $p<0.001$, ** $p<0.01$, * $p<0.05$, ns: not significant")
    latex_table.append("\\end{table}")
    
    return "\n".join(latex_table)

if __name__ == "__main__":
    # 示例数据 - 请替换为实际的实验结果
    # 每个数组应包含在测试集上每张图像的AUC值
    
    # DRIVE数据集结果
    drive_results = {
        'MDF-Net': np.array([0.98, 0.97, 0.99, 0.98, 0.97]),  # 示例数据
        'U-Net': np.array([0.95, 0.94, 0.96, 0.95, 0.94]),
        'DeepVessel': np.array([0.94, 0.93, 0.95, 0.94, 0.93]),
        'DUNet': np.array([0.96, 0.95, 0.97, 0.96, 0.95])
    }
    
    # CHASE_DB1数据集结果
    chase_results = {
        'MDF-Net': np.array([0.97, 0.96, 0.98, 0.97, 0.96]),
        'U-Net': np.array([0.94, 0.93, 0.95, 0.94, 0.93]),
        'DeepVessel': np.array([0.93, 0.92, 0.94, 0.93, 0.92]),
        'DUNet': np.array([0.95, 0.94, 0.96, 0.95, 0.94])
    }
    
    # STARE数据集结果
    stare_results = {
        'MDF-Net': np.array([0.98, 0.97, 0.99, 0.98, 0.97]),
        'U-Net': np.array([0.95, 0.94, 0.96, 0.95, 0.94]),
        'DeepVessel': np.array([0.94, 0.93, 0.95, 0.94, 0.93]),
        'DUNet': np.array([0.96, 0.95, 0.97, 0.96, 0.95])
    }
    
    # 计算每个数据集的显著性
    all_results = []
    
    for dataset_name, results in [
        ('DRIVE', drive_results),
        ('CHASE_DB1', chase_results),
        ('STARE', stare_results)
    ]:
        # 准备baseline结果
        baseline_results = {k: v for k, v in results.items() if k != 'MDF-Net'}
        
        # 计算显著性
        df = calculate_auc_significance(
            dataset_name,
            results['MDF-Net'],
            baseline_results
        )
        all_results.append(df)
    
    # 合并所有结果
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 打印结果
    print("\nStatistical Significance Analysis Results:")
    print("==========================================")
    print(final_df.to_string(index=False))
    
    # 生成LaTeX表格
    print("\nLaTeX Table:")
    print("============")
    print(format_latex_table(final_df)) 