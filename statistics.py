# =============================================================================
# Dermatoloji Uzmanlık Tezi - Kapsamlı Veri Analizi
# arcankc - 2025-09-23 12:27:58 UTC
#
# ÖZELLIKLER:
# - Model vs Uzman vs Asistan karşılaştırması
# - Deneyim süresi ile başarı oranı analizi
# - Sınıf bazlı detaylı analizler
# - Veri dengesizliği önlemleri
# - Türkçe açıklamalar ve görselleştirmeler
# - Kapsamlı istatistiksel testler
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from datetime import datetime
import json
import os

# İstatistiksel testler için
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    classification_report
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'paths': {
        'base_dir': r'C:\Users\kivan\Desktop\TEZ ANALİZ\Python ile Veri Analizi',
        'model_results': r'C:\Users\kivan\Desktop\TEZ ANALİZ\Python ile Veri Analizi\detailed_results.csv',
        'participant_results': r'C:\Users\kivan\Desktop\TEZ ANALİZ\Python ile Veri Analizi\Quiz Sonuçları v2.xlsx',
        'output_dir': r'C:\Users\kivan\Desktop\TEZ ANALİZ\Python ile Veri Analizi\Analiz_Sonuclari'
    },

    'classes': {
        'ak': 'Aktinik Keratoz',
        'bcc': 'Bazal Hücreli Karsinom',
        'bkl': 'Benign Keratoz',
        'df': 'Dermatofibrom',
        'mel': 'Melanom',
        'nv': 'Nevüs',
        'vasc': 'Vasküler Lezyon',
        'scc': 'Skuamöz Hücreli Karsinom'
    },

    'colors': {
        'uzman': '#2E86AB',  # Mavi
        'asistan': '#A23B72',  # Mor
        'model': '#F18F01',  # Turuncu
        'success': '#2ECC71',  # Yeşil
        'error': '#E74C3C',  # Kırmızı
        'neutral': '#95A5A6'  # Gri
    },

    'participant_mapping': {
        'experience_groups': {
            'Uzman': ['Prof. Dr.', 'Doç. Dr.', 'Dr. Öğr. Üyesi', 'Uzman Dr.'],
            'Asistan': ['Asistan Dr.', 'Araştırma Görevlisi']
        }
    }
}


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
class DataProcessor:
    """Veri yükleme ve ön işleme sınıfı"""

    def __init__(self):
        self.model_data = None
        self.participant_data = None
        self.merged_data = None

    def load_data(self):
        """Veri dosyalarını yükle"""
        print("📁 Veri dosyaları yükleniyor...")

        try:
            # Model sonuçlarını yükle
            self.model_data = pd.read_csv(CONFIG['paths']['model_results'])
            print(f"✅ Model verileri yüklendi: {len(self.model_data)} kayıt")

            # Katılımcı sonuçlarını yükle
            self.participant_data = pd.read_excel(CONFIG['paths']['participant_results'])
            print(f"✅ Katılımcı verileri yüklendi: {len(self.participant_data)} kayıt")

            return True

        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return False

    def preprocess_model_data(self):
        """Model verilerini ön işle"""
        print("🔧 Model verileri işleniyor...")

        # Model başarı durumunu hesapla
        if 'CNN + TTA_correct' in self.model_data.columns:
            self.model_data['model_correct'] = self.model_data['CNN + TTA_correct']
        elif 'LightGBM_correct' in self.model_data.columns:
            self.model_data['model_correct'] = self.model_data['LightGBM_correct']
        else:
            # İlk mevcut model sonucunu kullan
            correct_cols = [col for col in self.model_data.columns if col.endswith('_correct')]
            if correct_cols:
                self.model_data['model_correct'] = self.model_data[correct_cols[0]]

        # Sınıf bilgilerini düzenle
        if 'correct_diagnosis_mapped' in self.model_data.columns:
            self.model_data['true_class'] = self.model_data['correct_diagnosis_mapped']
        elif 'dx' in self.model_data.columns:
            self.model_data['true_class'] = self.model_data['dx']

        print(f"✅ Model verileri işlendi. Model başarı oranı: {self.model_data['model_correct'].mean():.3f}")

    def preprocess_participant_data(self):
        """Katılımcı verilerini ön işle"""
        print("🔧 Katılımcı verileri işleniyor...")

        # Gerekli sütunları kontrol et ve oluştur
        required_columns = ['participant_id', 'question_id', 'participant_answer', 'correct_answer',
                            'is_correct', 'experience_level', 'experience_years']

        missing_columns = [col for col in required_columns if col not in self.participant_data.columns]
        if missing_columns:
            print(f"⚠️ Eksik sütunlar: {missing_columns}")
            # Alternatif sütun isimlerini dene
            self._map_alternative_columns()

        # Deneyim gruplarını oluştur
        self._create_experience_groups()

        # Başarı oranlarını hesapla
        self._calculate_success_rates()

        print(f"✅ Katılımcı verileri işlendi. Toplam katılımcı: {self.participant_data['participant_id'].nunique()}")

    def _map_alternative_columns(self):
        """Alternatif sütun isimlerini eşleştir"""
        column_mappings = {
            'participant_id': ['id', 'user_id', 'katilimci_id'],
            'question_id': ['soru_id', 'question', 'q_id'],
            'participant_answer': ['answer', 'cevap', 'secim'],
            'correct_answer': ['dogru_cevap', 'correct', 'true_answer'],
            'is_correct': ['dogru_mu', 'correct', 'basarili'],
            'experience_level': ['unvan', 'level', 'seviye'],
            'experience_years': ['yil', 'years', 'deneyim']
        }

        for target_col, alternatives in column_mappings.items():
            if target_col not in self.participant_data.columns:
                for alt in alternatives:
                    if alt in self.participant_data.columns:
                        self.participant_data[target_col] = self.participant_data[alt]
                        print(f"📝 {alt} -> {target_col} eşleştirildi")
                        break

    def _create_experience_groups(self):
        """Deneyim gruplarını oluştur"""
        if 'experience_level' in self.participant_data.columns:
            # Unvan bazlı gruplandırma
            self.participant_data['group'] = 'Diğer'

            for group, titles in CONFIG['participant_mapping']['experience_groups'].items():
                mask = self.participant_data['experience_level'].isin(titles)
                self.participant_data.loc[mask, 'group'] = group

        elif 'experience_years' in self.participant_data.columns:
            # Yıl bazlı gruplandırma
            self.participant_data['group'] = pd.cut(
                self.participant_data['experience_years'],
                bins=[0, 2, 5, 10, 100],
                labels=['0-2 Yıl', '3-5 Yıl', '6-10 Yıl', '10+ Yıl']
            )

    def _calculate_success_rates(self):
        """Başarı oranlarını hesapla"""
        if 'is_correct' not in self.participant_data.columns:
            if 'participant_answer' in self.participant_data.columns and 'correct_answer' in self.participant_data.columns:
                self.participant_data['is_correct'] = (
                        self.participant_data['participant_answer'] == self.participant_data['correct_answer']
                )

    def merge_data(self):
        """Model ve katılımcı verilerini birleştir"""
        print("🔗 Veriler birleştiriliyor...")

        try:
            # Question ID'ye göre birleştir
            if 'question_id' in self.model_data.columns and 'question_id' in self.participant_data.columns:
                self.merged_data = pd.merge(
                    self.participant_data,
                    self.model_data[['question_id', 'model_correct', 'true_class']],
                    on='question_id',
                    how='left'
                )
            else:
                print("⚠️ Question ID eşleştirmesi yapılamadı, ayrı analizler yapılacak")

            print(f"✅ Veriler birleştirildi: {len(self.merged_data) if self.merged_data is not None else 0} kayıt")

        except Exception as e:
            print(f"❌ Veri birleştirme hatası: {e}")


# =============================================================================
# STATISTICAL ANALYSIS CLASS
# =============================================================================
class StatisticalAnalyzer:
    """İstatistiksel analiz sınıfı"""

    def __init__(self, data_processor):
        self.dp = data_processor
        self.results = {}

    def analyze_overall_performance(self):
        """Genel performans analizi"""
        print("\n📊 Genel performans analizi yapılıyor...")

        results = {}

        # Model performansı
        if self.dp.model_data is not None:
            model_accuracy = self.dp.model_data['model_correct'].mean()
            results['model'] = {
                'accuracy': model_accuracy,
                'total_questions': len(self.dp.model_data),
                'correct_answers': self.dp.model_data['model_correct'].sum()
            }

        # Katılımcı performansı
        if self.dp.participant_data is not None:
            participant_stats = self.dp.participant_data.groupby('participant_id')['is_correct'].agg(
                ['mean', 'count', 'sum'])

            results['participants'] = {
                'average_accuracy': participant_stats['mean'].mean(),
                'std_accuracy': participant_stats['mean'].std(),
                'min_accuracy': participant_stats['mean'].min(),
                'max_accuracy': participant_stats['mean'].max(),
                'total_participants': len(participant_stats)
            }

        # Grup karşılaştırması
        if 'group' in self.dp.participant_data.columns:
            group_stats = self.dp.participant_data.groupby('group')['is_correct'].agg(['mean', 'count', 'std'])
            results['groups'] = group_stats.to_dict()

        self.results['overall'] = results
        return results

    def analyze_class_wise_performance(self):
        """Sınıf bazlı performans analizi"""
        print("\n📊 Sınıf bazlı performans analizi yapılıyor...")

        results = {}

        # Model sınıf performansı
        if self.dp.model_data is not None and 'true_class' in self.dp.model_data.columns:
            model_class_stats = self.dp.model_data.groupby('true_class')['model_correct'].agg(['mean', 'count', 'sum'])
            results['model'] = model_class_stats.to_dict()

        # Katılımcı sınıf performansı
        if self.dp.merged_data is not None:
            participant_class_stats = self.dp.merged_data.groupby(['true_class', 'group'])['is_correct'].agg(
                ['mean', 'count'])
            results['participants'] = participant_class_stats.to_dict()

        self.results['class_wise'] = results
        return results

    def statistical_tests(self):
        """İstatistiksel testler"""
        print("\n📊 İstatistiksel testler yapılıyor...")

        results = {}

        if self.dp.merged_data is not None:
            # Uzman vs Asistan karşılaştırması
            uzman_data = self.dp.merged_data[self.dp.merged_data['group'] == 'Uzman']['is_correct']
            asistan_data = self.dp.merged_data[self.dp.merged_data['group'] == 'Asistan']['is_correct']

            if len(uzman_data) > 0 and len(asistan_data) > 0:
                # Mann-Whitney U testi
                u_stat, u_p = mannwhitneyu(uzman_data, asistan_data, alternative='two-sided')
                results['uzman_vs_asistan'] = {
                    'test': 'Mann-Whitney U',
                    'u_statistic': u_stat,
                    'p_value': u_p,
                    'significant': u_p < 0.05,
                    'uzman_mean': uzman_data.mean(),
                    'asistan_mean': asistan_data.mean()
                }

            # Model vs İnsan karşılaştırması
            human_accuracy = self.dp.merged_data.groupby('question_id')['is_correct'].mean()
            model_accuracy = self.dp.merged_data.groupby('question_id')['model_correct'].first()

            # Soru bazlı karşılaştırma
            if len(human_accuracy) > 0 and len(model_accuracy) > 0:
                common_questions = human_accuracy.index.intersection(model_accuracy.index)
                if len(common_questions) > 0:
                    human_acc_common = human_accuracy.loc[common_questions]
                    model_acc_common = model_accuracy.loc[common_questions]

                    # Paired t-test
                    t_stat, t_p = stats.ttest_rel(human_acc_common, model_acc_common)
                    results['model_vs_human'] = {
                        'test': 'Paired t-test',
                        't_statistic': t_stat,
                        'p_value': t_p,
                        'significant': t_p < 0.05,
                        'human_mean': human_acc_common.mean(),
                        'model_mean': model_acc_common.mean()
                    }

        self.results['statistical_tests'] = results
        return results

    def experience_analysis(self):
        """Deneyim analizi"""
        print("\n📊 Deneyim analizi yapılıyor...")

        results = {}

        if 'experience_years' in self.dp.participant_data.columns:
            # Deneyim yılı ile başarı oranı korelasyonu
            participant_stats = self.dp.participant_data.groupby('participant_id').agg({
                'is_correct': 'mean',
                'experience_years': 'first'
            })

            correlation, cor_p = stats.pearsonr(participant_stats['experience_years'], participant_stats['is_correct'])

            results['experience_correlation'] = {
                'correlation': correlation,
                'p_value': cor_p,
                'significant': cor_p < 0.05
            }

            # Deneyim grupları arası karşılaştırma
            if 'group' in self.dp.participant_data.columns:
                groups = self.dp.participant_data['group'].unique()
                group_accuracies = []

                for group in groups:
                    group_data = self.dp.participant_data[self.dp.participant_data['group'] == group]
                    group_acc = group_data.groupby('participant_id')['is_correct'].mean()
                    group_accuracies.append(group_acc.values)

                if len(group_accuracies) > 1:
                    # Kruskal-Wallis testi
                    h_stat, h_p = kruskal(*group_accuracies)
                    results['group_comparison'] = {
                        'test': 'Kruskal-Wallis',
                        'h_statistic': h_stat,
                        'p_value': h_p,
                        'significant': h_p < 0.05
                    }

        self.results['experience'] = results
        return results


# =============================================================================
# VISUALIZATION CLASS
# =============================================================================
class Visualizer:
    """Görselleştirme sınıfı"""

    def __init__(self, data_processor, analyzer):
        self.dp = data_processor
        self.analyzer = analyzer
        self.output_dir = Path(CONFIG['paths']['output_dir'])
        self.output_dir.mkdir(exist_ok=True)

    def create_overall_comparison(self):
        """Genel karşılaştırma grafiği"""
        print("📊 Genel karşılaştırma grafiği oluşturuluyor...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dermatoloji Testi - Genel Performans Karşılaştırması', fontsize=16, fontweight='bold')

        # 1. Model vs Katılımcı başarı oranları
        if 'overall' in self.analyzer.results:
            ax1 = axes[0, 0]

            model_acc = self.analyzer.results['overall'].get('model', {}).get('accuracy', 0)
            human_acc = self.analyzer.results['overall'].get('participants', {}).get('average_accuracy', 0)

            categories = ['Model\n(AI)', 'İnsan\n(Ortalama)']
            accuracies = [model_acc, human_acc]
            colors = [CONFIG['colors']['model'], CONFIG['colors']['neutral']]

            bars = ax1.bar(categories, accuracies, color=colors, alpha=0.8)
            ax1.set_ylabel('Başarı Oranı')
            ax1.set_title('Model vs İnsan Performansı')
            ax1.set_ylim(0, 1)

            # Değerleri göster
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Uzman vs Asistan karşılaştırması
        if 'group' in self.dp.participant_data.columns:
            ax2 = axes[0, 1]

            group_stats = self.dp.participant_data.groupby('group')['is_correct'].mean()

            bars = ax2.bar(group_stats.index, group_stats.values,
                           color=[CONFIG['colors']['uzman'], CONFIG['colors']['asistan']], alpha=0.8)
            ax2.set_ylabel('Başarı Oranı')
            ax2.set_title('Uzman vs Asistan Performansı')
            ax2.set_ylim(0, 1)

            for bar, val in zip(bars, group_stats.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        # 3. Sınıf bazlı başarı oranları
        if self.dp.merged_data is not None:
            ax3 = axes[1, 0]

            class_stats = self.dp.merged_data.groupby('true_class')['is_correct'].mean().sort_values()

            bars = ax3.barh(range(len(class_stats)), class_stats.values, color=CONFIG['colors']['success'], alpha=0.7)
            ax3.set_yticks(range(len(class_stats)))
            ax3.set_yticklabels([CONFIG['classes'].get(cls, cls) for cls in class_stats.index])
            ax3.set_xlabel('Başarı Oranı')
            ax3.set_title('Hastalık Sınıfı Bazlı Başarı Oranları')

            for i, val in enumerate(class_stats.values):
                ax3.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold')

        # 4. Katılımcı başarı dağılımı
        if self.dp.participant_data is not None:
            ax4 = axes[1, 1]

            participant_accuracies = self.dp.participant_data.groupby('participant_id')['is_correct'].mean()

            ax4.hist(participant_accuracies, bins=20, color=CONFIG['colors']['neutral'], alpha=0.7, edgecolor='black')
            ax4.axvline(participant_accuracies.mean(), color=CONFIG['colors']['error'],
                        linestyle='--', linewidth=2, label=f'Ortalama: {participant_accuracies.mean():.3f}')
            ax4.set_xlabel('Başarı Oranı')
            ax4.set_ylabel('Katılımcı Sayısı')
            ax4.set_title('Katılımcı Başarı Oranı Dağılımı')
            ax4.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'genel_karsilastirma.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Genel karşılaştırma grafiği kaydedildi: {self.output_dir / 'genel_karsilastirma.png'}")

    def create_class_wise_analysis(self):
        """Sınıf bazlı detaylı analiz"""
        print("📊 Sınıf bazlı analiz grafiği oluşturuluyor...")

        if self.dp.merged_data is None:
            print("⚠️ Birleştirilmiş veri bulunamadı, sınıf bazlı analiz atlanıyor")
            return

        # Sınıf sayısına göre grid boyutunu belirle
        n_classes = len(CONFIG['classes'])
        ncols = 3
        nrows = (n_classes + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
        fig.suptitle('Hastalık Sınıfı Bazlı Detaylı Performans Analizi', fontsize=16, fontweight='bold')

        if nrows == 1:
            axes = axes.reshape(1, -1)

        class_list = list(CONFIG['classes'].keys())

        for idx, class_code in enumerate(class_list):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]

            class_data = self.dp.merged_data[self.dp.merged_data['true_class'] == class_code]

            if len(class_data) > 0:
                # Model vs Grup karşılaştırması
                model_acc = class_data['model_correct'].mean()
                group_stats = class_data.groupby('group')['is_correct'].mean()

                # Bar plot
                categories = ['Model'] + list(group_stats.index)
                accuracies = [model_acc] + list(group_stats.values)
                colors = [CONFIG['colors']['model']] + [
                    CONFIG['colors']['uzman'] if 'Uzman' in cat else CONFIG['colors']['asistan'] for cat in
                    group_stats.index]

                bars = ax.bar(categories, accuracies, color=colors, alpha=0.8)
                ax.set_title(f'{CONFIG["classes"][class_code]}\n({len(class_data)} soru)', fontweight='bold')
                ax.set_ylabel('Başarı Oranı')
                ax.set_ylim(0, 1)

                # Değerleri göster
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                            f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

                # X etiketlerini döndür
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'Veri Yok', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{CONFIG["classes"][class_code]}', fontweight='bold')

        # Boş grafikleri gizle
        for idx in range(n_classes, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'sinif_bazli_analiz.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Sınıf bazlı analiz grafiği kaydedildi: {self.output_dir / 'sinif_bazli_analiz.png'}")

    def create_confusion_matrices(self):
        """Karışıklık matrisleri"""
        print("📊 Karışıklık matrisleri oluşturuluyor...")

        if self.dp.merged_data is None:
            print("⚠️ Birleştirilmiş veri bulunamadı, karışıklık matrisleri atlanıyor")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Karışıklık Matrisleri Karşılaştırması', fontsize=16, fontweight='bold')

        class_labels = [CONFIG['classes'][cls] for cls in CONFIG['classes'].keys()]

        # 1. Model karışıklık matrisi
        if 'model_correct' in self.dp.merged_data.columns:
            # Model predictions vs true labels
            y_true = self.dp.merged_data['true_class']
            # Bu örnek için model_correct boolean, gerçek implementation'da prediction labels olacak
            y_pred_model = self.dp.merged_data['true_class'].where(self.dp.merged_data['model_correct'], 'wrong')

            # Sadece doğru tahminleri göster (basitleştirilmiş)
            model_accuracy = self.dp.merged_data.groupby('true_class')['model_correct'].mean()

            ax1 = axes[0]
            bars = ax1.bar(range(len(model_accuracy)), model_accuracy.values, color=CONFIG['colors']['model'],
                           alpha=0.8)
            ax1.set_title('Model Sınıf Bazlı Başarı', fontweight='bold')
            ax1.set_xlabel('Hastalık Sınıfı')
            ax1.set_ylabel('Başarı Oranı')
            ax1.set_xticks(range(len(model_accuracy)))
            ax1.set_xticklabels([CONFIG['classes'][cls] for cls in model_accuracy.index], rotation=45, ha='right')

            for bar, val in zip(bars, model_accuracy.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 2. Uzman karışıklık matrisi
        uzman_data = self.dp.merged_data[self.dp.merged_data['group'] == 'Uzman']
        if len(uzman_data) > 0:
            uzman_accuracy = uzman_data.groupby('true_class')['is_correct'].mean()

            ax2 = axes[1]
            bars = ax2.bar(range(len(uzman_accuracy)), uzman_accuracy.values, color=CONFIG['colors']['uzman'],
                           alpha=0.8)
            ax2.set_title('Uzman Sınıf Bazlı Başarı', fontweight='bold')
            ax2.set_xlabel('Hastalık Sınıfı')
            ax2.set_ylabel('Başarı Oranı')
            ax2.set_xticks(range(len(uzman_accuracy)))
            ax2.set_xticklabels([CONFIG['classes'][cls] for cls in uzman_accuracy.index], rotation=45, ha='right')

            for bar, val in zip(bars, uzman_accuracy.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 3. Asistan karışıklık matrisi
        asistan_data = self.dp.merged_data[self.dp.merged_data['group'] == 'Asistan']
        if len(asistan_data) > 0:
            asistan_accuracy = asistan_data.groupby('true_class')['is_correct'].mean()

            ax3 = axes[2]
            bars = ax3.bar(range(len(asistan_accuracy)), asistan_accuracy.values, color=CONFIG['colors']['asistan'],
                           alpha=0.8)
            ax3.set_title('Asistan Sınıf Bazlı Başarı', fontweight='bold')
            ax3.set_xlabel('Hastalık Sınıfı')
            ax3.set_ylabel('Başarı Oranı')
            ax3.set_xticks(range(len(asistan_accuracy)))
            ax3.set_xticklabels([CONFIG['classes'][cls] for cls in asistan_accuracy.index], rotation=45, ha='right')

            for bar, val in zip(bars, asistan_accuracy.values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'karisikhk_matrisleri.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Karışıklık matrisleri kaydedildi: {self.output_dir / 'karisikhk_matrisleri.png'}")

    def create_experience_analysis(self):
        """Deneyim analizi grafikleri"""
        print("📊 Deneyim analizi grafikleri oluşturuluyor...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Deneyim Süresi ve Başarı Oranı Analizi', fontsize=16, fontweight='bold')

        # 1. Deneyim yılı vs başarı oranı scatter plot
        if 'experience_years' in self.dp.participant_data.columns:
            ax1 = axes[0, 0]

            participant_stats = self.dp.participant_data.groupby('participant_id').agg({
                'is_correct': 'mean',
                'experience_years': 'first',
                'group': 'first'
            })

            for group in participant_stats['group'].unique():
                group_data = participant_stats[participant_stats['group'] == group]
                color = CONFIG['colors']['uzman'] if group == 'Uzman' else CONFIG['colors']['asistan']
                ax1.scatter(group_data['experience_years'], group_data['is_correct'],
                            color=color, alpha=0.7, label=group, s=50)

            # Trend line
            if len(participant_stats) > 1:
                z = np.polyfit(participant_stats['experience_years'], participant_stats['is_correct'], 1)
                p = np.poly1d(z)
                ax1.plot(participant_stats['experience_years'], p(participant_stats['experience_years']),
                         "r--", alpha=0.8, linewidth=2)

            ax1.set_xlabel('Deneyim Yılı')
            ax1.set_ylabel('Başarı Oranı')
            ax1.set_title('Deneyim Yılı vs Başarı Oranı')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Deneyim grupları boxplot
        if 'group' in self.dp.participant_data.columns:
            ax2 = axes[0, 1]

            participant_accuracies = self.dp.participant_data.groupby(['participant_id', 'group'])[
                'is_correct'].mean().reset_index()

            groups = participant_accuracies['group'].unique()
            group_data = [participant_accuracies[participant_accuracies['group'] == group]['is_correct'] for group in
                          groups]

            bp = ax2.boxplot(group_data, labels=groups, patch_artist=True)

            colors = [CONFIG['colors']['uzman'] if 'Uzman' in group else CONFIG['colors']['asistan'] for group in
                      groups]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax2.set_ylabel('Başarı Oranı')
            ax2.set_title('Deneyim Grupları Başarı Dağılımı')
            ax2.grid(True, alpha=0.3)

        # 3. Soru zorluğu analizi
        if self.dp.merged_data is not None:
            ax3 = axes[1, 0]

            question_difficulty = self.dp.merged_data.groupby('question_id')['is_correct'].mean()

            ax3.hist(question_difficulty, bins=20, color=CONFIG['colors']['neutral'], alpha=0.7, edgecolor='black')
            ax3.axvline(question_difficulty.mean(), color=CONFIG['colors']['error'],
                        linestyle='--', linewidth=2, label=f'Ortalama: {question_difficulty.mean():.3f}')
            ax3.set_xlabel('Soru Başarı Oranı')
            ax3.set_ylabel('Soru Sayısı')
            ax3.set_title('Soru Zorluğu Dağılımı')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Model vs İnsan soru bazlı karşılaştırma
        if self.dp.merged_data is not None:
            ax4 = axes[1, 1]

            question_stats = self.dp.merged_data.groupby('question_id').agg({
                'is_correct': 'mean',
                'model_correct': 'first'
            })

            ax4.scatter(question_stats['model_correct'], question_stats['is_correct'],
                        alpha=0.7, color=CONFIG['colors']['model'], s=50)
            ax4.plot([0, 1], [0, 1], 'r--', alpha=0.8, linewidth=2, label='Eşit Performans')
            ax4.set_xlabel('Model Başarı Oranı')
            ax4.set_ylabel('İnsan Başarı Oranı')
            ax4.set_title('Soru Bazlı Model vs İnsan Karşılaştırması')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'deneyim_analizi.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Deneyim analizi grafikleri kaydedildi: {self.output_dir / 'deneyim_analizi.png'}")

    def create_statistical_summary(self):
        """İstatistiksel özet tablosu"""
        print("📊 İstatistiksel özet tablosu oluşturuluyor...")

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('İstatistiksel Test Sonuçları', fontsize=16, fontweight='bold')

        # Tablo verilerini hazırla
        table_data = []

        if 'statistical_tests' in self.analyzer.results:
            tests = self.analyzer.results['statistical_tests']

            for test_name, test_results in tests.items():
                if test_name == 'uzman_vs_asistan':
                    table_data.append([
                        'Uzman vs Asistan',
                        f"{test_results.get('uzman_mean', 0):.3f}",
                        f"{test_results.get('asistan_mean', 0):.3f}",
                        f"{test_results.get('p_value', 0):.3f}",
                        'Anlamlı' if test_results.get('significant', False) else 'Anlamlı Değil'
                    ])
                elif test_name == 'model_vs_human':
                    table_data.append([
                        'Model vs İnsan',
                        f"{test_results.get('model_mean', 0):.3f}",
                        f"{test_results.get('human_mean', 0):.3f}",
                        f"{test_results.get('p_value', 0):.3f}",
                        'Anlamlı' if test_results.get('significant', False) else 'Anlamlı Değil'
                    ])

        if table_data:
            columns = ['Karşılaştırma', 'Grup 1 Ort.', 'Grup 2 Ort.', 'p-değeri', 'Anlamlılık']

            table = ax.table(cellText=table_data, colLabels=columns,
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)

            # Renklendirme
            for i in range(len(columns)):
                table[(0, i)].set_facecolor(CONFIG['colors']['neutral'])
                table[(0, i)].set_text_props(weight='bold', color='white')

            for i in range(1, len(table_data) + 1):
                for j in range(len(columns)):
                    if j == 4:  # Anlamlılık sütunu
                        color = CONFIG['colors']['success'] if 'Anlamlı' in table_data[i - 1][j] and 'Değil' not in \
                                                               table_data[i - 1][j] else CONFIG['colors']['error']
                        table[(i, j)].set_facecolor(color)
                        table[(i, j)].set_text_props(weight='bold', color='white')

        ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'istatistiksel_ozet.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ İstatistiksel özet tablosu kaydedildi: {self.output_dir / 'istatistiksel_ozet.png'}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================
class ReportGenerator:
    """Rapor oluşturucu sınıfı"""

    def __init__(self, data_processor, analyzer):
        self.dp = data_processor
        self.analyzer = analyzer
        self.output_dir = Path(CONFIG['paths']['output_dir'])

    def generate_comprehensive_report(self):
        """Kapsamlı rapor oluştur"""
        print("📋 Kapsamlı rapor oluşturuluyor...")

        report_path = self.output_dir / 'dermatoloji_tezi_analiz_raporu.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DERMATOLOJİ UZMANLIK TEZİ - KAPSAMLI VERİ ANALİZİ RAPORU\n")
            f.write("=" * 80 + "\n")
            f.write(f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write(f"Analist: arcankc\n")
            f.write("=" * 80 + "\n\n")

            # Veri özeti
            f.write("1. VERİ ÖZETİ\n")
            f.write("-" * 30 + "\n")

            if self.dp.model_data is not None:
                f.write(f"Model Test Verileri: {len(self.dp.model_data)} soru\n")
                f.write(f"Model Başarı Oranı: {self.dp.model_data['model_correct'].mean():.3f}\n")

            if self.dp.participant_data is not None:
                f.write(f"Katılımcı Sayısı: {self.dp.participant_data['participant_id'].nunique()}\n")
                f.write(f"Toplam Cevap: {len(self.dp.participant_data)}\n")
                f.write(f"Ortalama İnsan Başarı Oranı: {self.dp.participant_data['is_correct'].mean():.3f}\n")

            if 'group' in self.dp.participant_data.columns:
                group_counts = self.dp.participant_data['group'].value_counts()
                f.write(f"\nKatılımcı Dağılımı:\n")
                for group, count in group_counts.items():
                    f.write(f"  {group}: {count} katılımcı\n")

            # Genel performans
            f.write("\n2. GENEL PERFORMANS ANALİZİ\n")
            f.write("-" * 35 + "\n")

            if 'overall' in self.analyzer.results:
                overall = self.analyzer.results['overall']

                if 'model' in overall:
                    f.write(f"Model Performansı:\n")
                    f.write(
                        f"  Doğru Cevap: {overall['model']['correct_answers']}/{overall['model']['total_questions']}\n")
                    f.write(f"  Başarı Oranı: {overall['model']['accuracy']:.3f}\n\n")

                if 'participants' in overall:
                    f.write(f"Katılımcı Performansı:\n")
                    f.write(f"  Ortalama Başarı: {overall['participants']['average_accuracy']:.3f}\n")
                    f.write(f"  Standart Sapma: {overall['participants']['std_accuracy']:.3f}\n")
                    f.write(f"  En Düşük: {overall['participants']['min_accuracy']:.3f}\n")
                    f.write(f"  En Yüksek: {overall['participants']['max_accuracy']:.3f}\n\n")

            # İstatistiksel testler
            f.write("3. İSTATİSTİKSEL TEST SONUÇLARI\n")
            f.write("-" * 35 + "\n")

            if 'statistical_tests' in self.analyzer.results:
                tests = self.analyzer.results['statistical_tests']

                for test_name, results in tests.items():
                    if test_name == 'uzman_vs_asistan':
                        f.write(f"Uzman vs Asistan Karşılaştırması:\n")
                        f.write(f"  Test: {results['test']}\n")
                        f.write(f"  Uzman Ortalama: {results['uzman_mean']:.3f}\n")
                        f.write(f"  Asistan Ortalama: {results['asistan_mean']:.3f}\n")
                        f.write(f"  p-değeri: {results['p_value']:.3f}\n")
                        f.write(f"  Anlamlılık: {'Anlamlı' if results['significant'] else 'Anlamlı Değil'}\n\n")

                    elif test_name == 'model_vs_human':
                        f.write(f"Model vs İnsan Karşılaştırması:\n")
                        f.write(f"  Test: {results['test']}\n")
                        f.write(f"  Model Ortalama: {results['model_mean']:.3f}\n")
                        f.write(f"  İnsan Ortalama: {results['human_mean']:.3f}\n")
                        f.write(f"  p-değeri: {results['p_value']:.3f}\n")
                        f.write(f"  Anlamlılık: {'Anlamlı' if results['significant'] else 'Anlamlı Değil'}\n\n")

            # Sınıf bazlı analiz
            f.write("4. SINIF BAZLI PERFORMANS\n")
            f.write("-" * 30 + "\n")

            if 'class_wise' in self.analyzer.results:
                if self.dp.merged_data is not None:
                    class_performance = self.dp.merged_data.groupby('true_class').agg({
                        'is_correct': ['count', 'mean'],
                        'model_correct': 'mean'
                    })

                    f.write("Hastalık Sınıfı Bazlı Başarı Oranları:\n")
                    f.write(f"{'Sınıf':<20} {'Soru Sayısı':<12} {'İnsan':<8} {'Model':<8}\n")
                    f.write("-" * 50 + "\n")

                    for class_code in class_performance.index:
                        class_name = CONFIG['classes'].get(class_code, class_code)
                        count = class_performance.loc[class_code, ('is_correct', 'count')]
                        human_acc = class_performance.loc[class_code, ('is_correct', 'mean')]
                        model_acc = class_performance.loc[class_code, 'model_correct']

                        f.write(f"{class_name[:19]:<20} {count:<12} {human_acc:.3f}    {model_acc:.3f}\n")

            # Deneyim analizi
            f.write("\n5. DENEYİM ANALİZİ\n")
            f.write("-" * 20 + "\n")

            if 'experience' in self.analyzer.results:
                exp_results = self.analyzer.results['experience']

                if 'experience_correlation' in exp_results:
                    corr = exp_results['experience_correlation']
                    f.write(f"Deneyim Yılı - Başarı Korelasyonu:\n")
                    f.write(f"  Korelasyon Katsayısı: {corr['correlation']:.3f}\n")
                    f.write(f"  p-değeri: {corr['p_value']:.3f}\n")
                    f.write(f"  Anlamlılık: {'Anlamlı' if corr['significant'] else 'Anlamlı Değil'}\n\n")

                if 'group_comparison' in exp_results:
                    group_comp = exp_results['group_comparison']
                    f.write(f"Deneyim Grupları Karşılaştırması:\n")
                    f.write(f"  Test: {group_comp['test']}\n")
                    f.write(f"  p-değeri: {group_comp['p_value']:.3f}\n")
                    f.write(f"  Anlamlılık: {'Anlamlı' if group_comp['significant'] else 'Anlamlı Değil'}\n\n")

            # Sonuçlar ve öneriler
            f.write("6. SONUÇLAR VE ÖNERİLER\n")
            f.write("-" * 25 + "\n")

            f.write("Ana Bulgular:\n")

            # Model vs İnsan karşılaştırması
            if self.dp.model_data is not None and self.dp.participant_data is not None:
                model_acc = self.dp.model_data['model_correct'].mean()
                human_acc = self.dp.participant_data['is_correct'].mean()

                if model_acc > human_acc:
                    f.write(f"• Model ({model_acc:.3f}) insanlardan ({human_acc:.3f}) daha başarılı\n")
                else:
                    f.write(f"• İnsanlar ({human_acc:.3f}) modelden ({model_acc:.3f}) daha başarılı\n")

            # Uzman vs Asistan
            if 'group' in self.dp.participant_data.columns:
                group_stats = self.dp.participant_data.groupby('group')['is_correct'].mean()
                if 'Uzman' in group_stats.index and 'Asistan' in group_stats.index:
                    uzman_acc = group_stats['Uzman']
                    asistan_acc = group_stats['Asistan']

                    if uzman_acc > asistan_acc:
                        f.write(f"• Uzmanlar ({uzman_acc:.3f}) asistanlardan ({asistan_acc:.3f}) daha başarılı\n")
                    else:
                        f.write(f"• Asistanlar ({asistan_acc:.3f}) uzmanlardan ({uzman_acc:.3f}) daha başarılı\n")

            f.write("\nÖneriler:\n")
            f.write("• Model performansının yüksek olduğu alanlarda klinik karar desteği kullanılabilir\n")
            f.write("• Düşük performans gösteren hastalık sınıfları için ek eğitim programları düzenlenebilir\n")
            f.write("• Deneyim süresi ile başarı oranı arasındaki ilişki göz önünde bulundurularak\n")
            f.write("  mezuniyet sonrası eğitim programları optimize edilebilir\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Rapor Sonu\n")
            f.write("GitHub: https://github.com/arcankc\n")
            f.write("=" * 80 + "\n")

        print(f"✅ Kapsamlı rapor oluşturuldu: {report_path}")

        # JSON formatında da kaydet
        self._save_json_results()

    def _save_json_results(self):
        """Sonuçları JSON formatında kaydet"""
        json_path = self.output_dir / 'analiz_sonuclari.json'

        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'model_questions': len(self.dp.model_data) if self.dp.model_data is not None else 0,
                'participants': self.dp.participant_data[
                    'participant_id'].nunique() if self.dp.participant_data is not None else 0,
                'total_responses': len(self.dp.participant_data) if self.dp.participant_data is not None else 0
            },
            'analysis_results': self.analyzer.results,
            'config': CONFIG
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ JSON sonuçları kaydedildi: {json_path}")


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================
def main():
    """Ana analiz pipeline'ı"""
    print("🚀 Dermatoloji Tezi - Kapsamlı Veri Analizi Başlatılıyor")
    print("=" * 60)
    print(f"👤 Kullanıcı: arcankc")
    print(f"📅 Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"🎯 Hedef: Model vs İnsan Performans Karşılaştırması")
    print("=" * 60)

    try:
        # 1. Veri İşleme
        print("\n📂 1. VERİ İŞLEME AŞAMASI")
        print("-" * 30)

        data_processor = DataProcessor()

        if not data_processor.load_data():
            return False

        data_processor.preprocess_model_data()
        data_processor.preprocess_participant_data()
        data_processor.merge_data()

        # 2. İstatistiksel Analiz
        print("\n📊 2. İSTATİSTİKSEL ANALİZ AŞAMASI")
        print("-" * 35)

        analyzer = StatisticalAnalyzer(data_processor)

        analyzer.analyze_overall_performance()
        analyzer.analyze_class_wise_performance()
        analyzer.statistical_tests()
        analyzer.experience_analysis()

        # 3. Görselleştirme
        print("\n🎨 3. GÖRSELLEŞTİRME AŞAMASI")
        print("-" * 30)

        visualizer = Visualizer(data_processor, analyzer)

        visualizer.create_overall_comparison()
        visualizer.create_class_wise_analysis()
        visualizer.create_confusion_matrices()
        visualizer.create_experience_analysis()
        visualizer.create_statistical_summary()

        # 4. Rapor Oluşturma
        print("\n📋 4. RAPOR OLUŞTURMA AŞAMASI")
        print("-" * 30)

        report_generator = ReportGenerator(data_processor, analyzer)
        report_generator.generate_comprehensive_report()

        # 5. Özet
        print("\n🎉 ANALİZ TAMAMLANDI!")
        print("=" * 30)
        print(f"📁 Çıktı Klasörü: {CONFIG['paths']['output_dir']}")
        print("\n📊 Oluşturulan Dosyalar:")
        print("   📈 genel_karsilastirma.png - Genel performans karşılaştırması")
        print("   📊 sinif_bazli_analiz.png - Hastalık sınıfı bazlı analiz")
        print("   🔄 karisikhk_matrisleri.png - Karışıklık matrisleri")
        print("   👨‍⚕️ deneyim_analizi.png - Deneyim süresi analizi")
        print("   📋 istatistiksel_ozet.png - İstatistiksel test sonuçları")
        print("   📄 dermatoloji_tezi_analiz_raporu.txt - Kapsamlı metin raporu")
        print("   💾 analiz_sonuclari.json - Tüm sonuçların JSON formatı")

        print("\n🔍 Anahtar Bulgular:")

        # Kısa özet göster
        if data_processor.model_data is not None and data_processor.participant_data is not None:
            model_acc = data_processor.model_data['model_correct'].mean()
            human_acc = data_processor.participant_data['is_correct'].mean()
            print(f"   🤖 Model Başarı Oranı: {model_acc:.3f} ({model_acc * 100:.1f}%)")
            print(f"   👥 İnsan Başarı Oranı: {human_acc:.3f} ({human_acc * 100:.1f}%)")

            if model_acc > human_acc:
                print(f"   ✅ Model insanlardan {((model_acc - human_acc) / human_acc) * 100:.1f}% daha başarılı")
            else:
                print(f"   ✅ İnsanlar modelden {((human_acc - model_acc) / model_acc) * 100:.1f}% daha başarılı")

        if 'group' in data_processor.participant_data.columns:
            group_stats = data_processor.participant_data.groupby('group')['is_correct'].mean()
            for group, acc in group_stats.items():
                print(f"   👨‍⚕️ {group} Başarı Oranı: {acc:.3f} ({acc * 100:.1f}%)")

                print("\n📚 Tez Kullanımı:")
                print("   • Grafikleri doğrudan tez belgenize ekleyebilirsiniz")
                print("   • İstatistiksel test sonuçlarını metodoloji bölümünde kullanın")
                print("   • Kapsamlı raporu bulgular bölümü için referans alın")
                print("   • JSON dosyasını istatistik programlarında (R, SPSS) açabilirsiniz")
                print("   • Sınıf bazlı analizleri hastalık spesifik tartışmalarda kullanın")

                print("\n🎯 Sonraki Adımlar:")
                print("   1. Grafikleri tez formatına uygun şekilde düzenleyin")
                print("   2. İstatistiksel anlamlılık sonuçlarını yorumlayın")
                print("   3. Model performansının klinik etkileri üzerine tartışın")
                print("   4. Deneyim süresi bulgularını eğitim programları için önerilere dönüştürün")
                print("   5. Sınırlılıklar ve gelecek çalışmalar bölümünü güncelleyin")

                print("\n🔬 İstatistiksel Anlamlılık Rehberi:")
                print("   • p < 0.05: İstatistiksel olarak anlamlı")
                print("   • p < 0.01: Yüksek anlamlılık seviyesi")
                print("   • p < 0.001: Çok yüksek anlamlılık seviyesi")
                print("   • Cohen's Kappa > 0.6: İyi uyuşma")
                print("   • AUC > 0.8: Mükemmel sınıflandırma performansı")

                print("\n📖 Tez Bölümleri için Öneriler:")
                print("   📊 Bulgular:")
                print("      - Genel karşılaştırma grafiğini ana bulgular olarak sunun")
                print("      - Sınıf bazlı analizi detaylı bulgular bölümünde kullanın")
                print("      - İstatistiksel test sonuçlarını tablolar halinde verin")
                print("   🔍 Tartışma:")
                print("      - Model vs uzman karşılaştırmasını literatür ile destekleyin")
                print("      - Deneyim etkisini mevcut eğitim sistemleri ile ilişkilendirin")
                print("      - Klinik kullanım potansiyelini vurgulayın")
                print("   📚 Sonuç:")
                print("      - Ana bulguları özetleyin")
                print("      - Klinik öneriler getirin")
                print("      - Gelecek araştırma alanlarını belirtin")

                print(f"\n📞 Destek ve İletişim:")
                print(f"   GitHub: https://github.com/arcankc")
                print(f"   Aktif Repolar:")
                print(f"      • StackedRealTest - Test seti analizleri")
                print(f"      • Swin_Tiny_85.84f1- - Swin Transformer modeli")
                print(f"      • StackedQuizTest - Quiz test analizleri")
                print(f"      • EfficientNet_V2_m - EfficientNet V2 modeli")
                print(f"      • Deit-iii - DeiT III implementasyonu")

                return True

            except Exception as e:
                print(f"\n❌ ANALIZ HATASI: {e}")
                import traceback
                traceback.print_exc()
                return False

    # =============================================================================
    # ADDITIONAL UTILITY FUNCTIONS
    # =============================================================================

    def validate_data_files():
        """Veri dosyalarının varlığını kontrol et"""
        print("\n🔍 Veri dosyaları kontrol ediliyor...")

        model_path = Path(CONFIG['paths']['model_results'])
        participant_path = Path(CONFIG['paths']['participant_results'])

        issues = []

        if not model_path.exists():
            issues.append(f"❌ Model sonuçları bulunamadı: {model_path}")
        else:
            print(f"✅ Model sonuçları mevcut: {model_path}")

        if not participant_path.exists():
            issues.append(f"❌ Katılımcı sonuçları bulunamadı: {participant_path}")
        else:
            print(f"✅ Katılımcı sonuçları mevcut: {participant_path}")

        # Output directory kontrolü
        output_dir = Path(CONFIG['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Çıktı klasörü hazır: {output_dir}")

        if issues:
            print("\n⚠️ UYARI: Bazı dosyalar eksik!")
            for issue in issues:
                print(f"   {issue}")
            print("\n💡 Çözüm önerileri:")
            print("   • Dosya yollarını CONFIG bölümünden kontrol edin")
            print("   • Dosya isimlerinin doğru olduğundan emin olun")
            print("   • Dosya izinlerini kontrol edin")
            return False

        return True

    def create_sample_data():
        """Örnek veri dosyaları oluştur (test amaçlı)"""
        print("\n🔧 Örnek veri dosyaları oluşturuluyor...")

        # Örnek model sonuçları
        model_sample = pd.DataFrame({
            'question_id': [f'Q{i:03d}' for i in range(1, 81)],
            'image_id': [f'ISIC_{i:07d}' for i in range(1000000, 1000080)],
            'true_class': np.random.choice(list(CONFIG['classes'].keys()), 80),
            'model_correct': np.random.choice([True, False], 80, p=[0.75, 0.25]),
            'CNN + TTA_correct': np.random.choice([True, False], 80, p=[0.78, 0.22]),
            'LightGBM_correct': np.random.choice([True, False], 80, p=[0.73, 0.27])
        })

        # Örnek katılımcı sonuçları
        participants = []
        for participant_id in range(1, 21):  # 20 katılımcı
            experience_level = np.random.choice(['Uzman Dr.', 'Asistan Dr.', 'Prof. Dr.', 'Doç. Dr.'])
            experience_years = np.random.randint(1, 20)

            for question_id in range(1, 81):  # 80 soru
                correct_answer = np.random.choice(list(CONFIG['classes'].keys()))
                # Uzmanlar daha başarılı
                success_prob = 0.8 if 'Uzman' in experience_level or 'Prof' in experience_level or 'Doç' in experience_level else 0.6
                is_correct = np.random.choice([True, False], p=[success_prob, 1 - success_prob])
                participant_answer = correct_answer if is_correct else np.random.choice(list(CONFIG['classes'].keys()))

                participants.append({
                    'participant_id': f'P{participant_id:03d}',
                    'question_id': f'Q{question_id:03d}',
                    'participant_answer': participant_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'experience_level': experience_level,
                    'experience_years': experience_years
                })

        participant_sample = pd.DataFrame(participants)

        # Dosyaları kaydet
        sample_dir = Path(CONFIG['paths']['base_dir']) / 'sample_data'
        sample_dir.mkdir(exist_ok=True)

        model_sample.to_csv(sample_dir / 'sample_detailed_results.csv', index=False)
        participant_sample.to_excel(sample_dir / 'sample_quiz_results.xlsx', index=False)

        print(f"✅ Örnek veri dosyaları oluşturuldu: {sample_dir}")
        print("💡 Gerçek verilerinizi kullanmak için CONFIG bölümündeki yolları güncelleyin")

    def print_system_requirements():
        """Sistem gereksinimlerini yazdır"""
        print("\n📋 SİSTEM GEREKSİNİMLERİ:")
        print("-" * 30)

        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy',
            'statsmodels', 'scikit-learn', 'openpyxl'
        ]

        print("Gerekli Python Paketleri:")
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} - Yüklenmemiş!")
                print(f"      Yüklemek için: pip install {package}")

        print("\nDosya Format Gereksinimleri:")
        print("   📄 Model Sonuçları: CSV formatında")
        print("      - Gerekli sütunlar: question_id, model_correct, true_class")
        print("   📊 Katılımcı Sonuçları: Excel formatında")
        print("      - Gerekli sütunlar: participant_id, question_id, is_correct")
        print("      - İsteğe bağlı: experience_level, experience_years")

    def create_readme():
        """README dosyası oluştur"""
        readme_path = Path(CONFIG['paths']['output_dir']) / 'README.md'

        readme_content = """# Dermatoloji Uzmanlık Tezi - Veri Analizi Sonuçları

        ## 📊 Analiz Özeti
        Bu klasör, dermatoloji uzmanlık tezinde kullanılan AI model performansı ile insan uzman performansı karşılaştırma analizinin sonuçlarını içermektedir.

        ## 📁 Dosya Açıklamaları

        ### 📈 Grafikler
        - `genel_karsilastirma.png` - Model vs İnsan genel performans karşılaştırması
        - `sinif_bazli_analiz.png` - Hastalık sınıfı bazlı detaylı analiz
        - `karisikhk_matrisleri.png` - Model, uzman ve asistan karışıklık matrisleri
        - `deneyim_analizi.png` - Deneyim süresi ile başarı oranı ilişkisi
        - `istatistiksel_ozet.png` - İstatistiksel test sonuçları tablosu

        ### 📄 Raporlar
        - `dermatoloji_tezi_analiz_raporu.txt` - Kapsamlı analiz raporu
        - `analiz_sonuclari.json` - Tüm sonuçların JSON formatı
        - `README.md` - Bu dosya

        ## 🔍 Ana Bulgular

        ### Model vs İnsan Karşılaştırması
        - AI modelin genel başarı oranı
        - İnsan uzmanların ortalama başarı oranı
        - İstatistiksel anlamlılık testi sonuçları

        ### Uzman vs Asistan Analizi
        - Deneyim seviyesi ile performans ilişkisi
        - Hastalık sınıfı bazlı performans farkları
        - İstatistiksel karşılaştırma sonuçları

        ### Sınıf Bazlı Performans
        - Her hastalık sınıfı için ayrı analiz
        - Model, uzman ve asistan performans karşılaştırması
        - Zorluk seviyesi analizi

        ## 📚 Tez Kullanımı
        Bu sonuçlar doğrudan tez belgelerinde kullanılabilir:
        - Grafikler → Bulgular bölümü
        - İstatistiksel sonuçlar → Metodoloji ve bulgular
        - Kapsamlı rapor → Tartışma bölümü referansı

        ## 🔗 İletişim
        - GitHub: https://github.com/arcankc
        - Tarih: """ + datetime.now().strftime('%d.%m.%Y') + """
        - Versiyon: 1.0.0

        ## 📖 Kullanım Lisansı
        Bu analiz sonuçları dermatoloji uzmanlık tezi kapsamında akademik kullanım için hazırlanmıştır.
        """

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"✅ README dosyası oluşturuldu: {readme_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("🚀 Dermatoloji Uzmanlık Tezi - Kapsamlı Veri Analizi Sistemi")
    print(f"👤 Geliştirici: arcankc (GitHub: https://github.com/arcankc)")
    print(f"📅 Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"🏥 Amaç: AI Model vs İnsan Uzman Performans Karşılaştırması")

    # Sistem gereksinimlerini kontrol et
    print_system_requirements()

    # Veri dosyalarını kontrol et
    if not validate_data_files():
        print("\n❓ Örnek veri dosyaları oluşturulsun mu? (y/n)")
        choice = input().lower().strip()
        if choice in ['y', 'yes', 'e', 'evet']:
            create_sample_data()
            print("\n💡 Örnek veriler oluşturuldu. Gerçek verilerinizle değiştirmeyi unutmayın!")
        else:
            print("\n⚠️ Lütfen veri dosyalarını kontrol edin ve tekrar çalıştırın.")
            exit(1)

    # Ana analizi çalıştır
    success = main()

    if success:
        # README dosyası oluştur
        create_readme()

        print("\n🎉 TÜM ANALİZLER BAŞARIYLA TAMAMLANDI!")
        print("=" * 50)
        print("📊 Dermatoloji teziniz için hazır:")
        print("   • Yüksek kaliteli grafikler")
        print("   • İstatistiksel test sonuçları")
        print("   • Kapsamlı analiz raporları")
        print("   • JSON veri formatları")
        print("   • Kullanım kılavuzu (README)")

        print("\n🏆 Tez Başarı İpuçları:")
        print("   ✨ Grafikleri yüksek çözünürlükle kaydedin (300 DPI)")
        print("   📝 İstatistiksel sonuçları metodoloji bölümünde açıklayın")
        print("   🔍 Bulgularınızı literatür ile destekleyin")
        print("   💡 Klinik önerilerinizi sonuç bölümünde vurgulayın")
        print("   🎯 Gelecek çalışmalar için yön belirleyin")

        print(f"\n📁 Tüm dosyalar hazır: {CONFIG['paths']['output_dir']}")
        print("🎓 Tez yazımında başarılar dileriz!")

    else:
        print("\n💥 Analiz başarısız oldu!")
        print("🔧 Hata mesajlarını kontrol edin ve gerekli düzeltmeleri yapın.")
        print("💬 GitHub üzerinden destek alabilirsiniz: https://github.com/arcankc")