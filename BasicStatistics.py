import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class BasicStatistics:
    def __init__(self, filename, file_prefix):
        # CSVファイルからデータを読み込む
        self.df = pd.read_csv(filename)
        self.file_prefix = file_prefix

    def generate_basic_statistics(self):
        # 基本統計量の算出とCSVファイルへの保存
        summary = self.df.describe()
        summary.to_csv(self.file_prefix +'basic_statistics.csv')

        # ヒートマップの表示とPNGファイルへの保存
        plt.figure(figsize=(10, 8))
        sns.heatmap(summary, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Basic Statistics Heatmap')
        plt.xlabel('Variables')
        plt.ylabel('Statistics')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.file_prefix + 'basic_statistics_heatmap.png')
        plt.close()

    def generate_boxplots(self, filename='boxplots.png'):
        # 数値カラムのボックスプロットの作成とPNGファイルへの保存
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        # 列数を5で固定し、行数をデータフレームの要素数に基づいて計算
        num_rows = int(np.ceil(len(numeric_columns) / 5))
        
        plt.figure(figsize=(15, 5 * num_rows))  # グラフのサイズを行数に基づいて調整
        for i, col in enumerate(numeric_columns):
            # グリッドの位置を計算（行数、列数、インデックス+1）
            plt.subplot(num_rows, 5, i + 1)
            # ボックスプロットは縦向きに描画
            sns.boxplot(y=self.df[col])
            plt.title(col)
        
        plt.tight_layout()
        plt.savefig(self.file_prefix + filename)
        plt.close()


    def generate_histograms(self, filename='histograms.png'):
        # データフレームの列数に基づいてグリッドサイズを決定
        num_columns = len(self.df.columns)
        num_rows = int(np.ceil(num_columns / 3))  # 3列と仮定して行数を計算

        plt.figure(figsize=(20, 5 * num_rows))  # グラフのサイズを行数に基づいて調整

        for i, col in enumerate(self.df.columns):
            plt.subplot(num_rows, 3, i + 1)  # 動的に決定されたグリッドサイズでサブプロットを配置
            sns.histplot(self.df[col], kde=True, color='skyblue')
            plt.title(col)

        plt.tight_layout()
        plt.savefig(self.file_prefix + filename)
        plt.close()

    def generate_correlation_heatmap(self, filename='correlation_heatmap.png'):
        # 相関係数行列とヒートマップの描画、PNGファイルへの保存
        corr_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title('Correlation Heatmap of Diabetes Dataset Features')
        plt.savefig(self.file_prefix + filename)
        plt.close()

    def generate_scatter_matrix(self, filename='scatter_matrix.png'):
        # 数値カラムのみを抽出
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        
        # seabornのpairplotを使って散布図行列を作成
        sns.pairplot(self.df, vars=numeric_columns, hue='target', kind='reg')

        # グラフをPNGファイルに保存
        plt.savefig(self.file_prefix + filename)
        plt.close()  # グラフのクリア

    def perform_pca_and_plot1(self, n_components=0.8):
        # 数値カラムのみを抽出
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # target列があれば削除
        if 'target' in numeric_columns:
            numeric_columns.remove('target')
            
        # データを標準化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[numeric_columns])
        
        # PCAを実行
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(scaled_data)
        
        # 累積寄与率のパレート図を作成し、ファイルに保存
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(8, 6))
        plt.plot(cumulative_variance_ratio, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Ratio')
        plt.title('Cumulative Variance Ratio by Number of Components')
        plt.grid(True)
        plt.savefig(file_prefix + 'cumulative_variance_ratio.png')
        plt.close()

        # PC1とPC2の散布図を作成し、ファイルに保存
        pca_df = pd.DataFrame(data=pca_data, columns=['PC{}'.format(i+1) for i in range(pca_data.shape[1])])
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', data=pca_df, palette='viridis', s=100)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Scatter Plot with Variables Directions')
        plt.grid(True)
        plt.savefig(file_prefix + 'pca_scatter_plot.png')
        plt.close()

    def perform_pca_and_plot(self, n_components=0.8, magnification = 8):
        # 数値カラムのみを抽出
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # target列があれば削除し、変数に保存
        target_column = None
        if 'target' in self.df.columns:
            target_column = self.df['target']
            numeric_columns.remove('target')
            
        # データを標準化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[numeric_columns])
        
        # PCAを実行
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(scaled_data)
        
        # 累積寄与率のパレート図を作成し、ファイルに保存
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(8, 6))
        plt.plot(cumulative_variance_ratio, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Ratio')
        plt.title('Cumulative Variance Ratio by Number of Components')
        plt.grid(True)
        plt.savefig(self.file_prefix + 'cumulative_variance_ratio.png')
        plt.close()

        # PC1とPC2の散布図を作成し、ファイルに保存
        pca_df = pd.DataFrame(data=pca_data, columns=['PC{}'.format(i+1) for i in range(pca_data.shape[1])])
        plt.figure(figsize=(10, 8))
        if target_column is not None:
            pca_df['target'] = target_column.values  # target列をPCAデータフレームに追加
            sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue='target', palette='viridis', s=100)
        else:
            sns.scatterplot(x='PC1', y='PC2', data=pca_df, s=100)

        # 各変数のPCAコンポーネント方向を矢印で示す
        for i, variable in enumerate(numeric_columns[:10]):  # 最初の10変数のみを表示
            plt.arrow(0, 0, pca.components_[0, i]*magnification, pca.components_[1, i]*magnification, color='red', alpha=0.5, head_width=0.02)
            plt.text(pca.components_[0, i]*(magnification+1), pca.components_[1, i]*(magnification+1), variable, color='red', fontsize=9, ha='center', va='center')

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Scatter Plot with Variables Directions')
        plt.grid(True)
        plt.savefig(self.file_prefix + 'pca_scatter_plot.png')
        plt.close()

        # PCAの各主成分に対する変数の重みを抽出
        weights = pca.components_
        sorted_indices = np.argsort(np.abs(weights), axis=1)[:, ::-1]
        sorted_columns = [np.array(numeric_columns)[sorted_indices[i]] for i in range(len(pca.components_))]

        # 各主成分の重みを示すバープロットを作成し、ファイルに保存
        for i, component in enumerate(weights, 1):
            plt.figure(figsize=(8, 6))
            plt.barh(sorted_columns[i-1][:10], component[sorted_indices[i-1]][:10], color='skyblue')
            plt.title(f'PC{i} Weights')
            plt.xlabel('Weight')
            plt.ylabel('Variable')
            plt.tight_layout()
            plt.grid()
            plt.savefig(self.file_prefix + f'pc{i}_weights.png')
            plt.close()


if __name__ == '__main__':
    filename = 'wine_data.csv'  # ここにCSVファイル名を指定
    file_prefix = 'wine_'
    bs = BasicStatistics(filename, file_prefix)
    bs.generate_basic_statistics()
    bs.generate_boxplots()
    bs.generate_histograms()
    bs.generate_correlation_heatmap()
    bs.generate_scatter_matrix() #項目数が多い場合はやめた方がいい
    bs.perform_pca_and_plot() 
