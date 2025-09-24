#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
DERMATOLOJI UZMANLIK TEZİ - KAPSAMLI VERİ ANALİZ SİSTEMİ
=============================================================================

Bu program dermatoloji uzmanlık tezi için yapay zeka modeli ile uzman/asistan
performansının karşılaştırmalı analizini yapmaktadır.

Özellikler:
- Katılımcı veri temizleme ve filtreleme
- 8 sınıf bazlı dermatolojik hastalık analizi
- İstatistiksel testler (p < 0.05)
- Kapsamlı görselleştirmeler (Türkçe)
- Tez için hazır çıktılar

Geliştirici: arcankc
GitHub: https://github.com/arcankc
Tarih: 24.09.2025
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime
import json
import traceback

# Veri işleme ve analiz
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, pearsonr, spearmanr, shapiro
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind, ttest_rel
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm

# Görselleştirme
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

# Uyarıları filtrele
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için matplotlib ayarları
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'paths': {
        'base_dir': r'C:\Users\kivan\Desktop\tez_analiz\Deit Quiz ve Test',
        'participant_results': r'C:\Users\kivan\Desktop\tez_analiz\Deit Quiz ve Test\quiz_sonuclari322.csv',
        'model_quiz_results': r'C:\Users\kivan\Desktop\tez_analiz\Deit Quiz ve Test\Deit Quiz Output\quiz_detailed_results.csv',
        'model_test_results': r'C:\Users\kivan\Desktop\tez_analiz\Deit Quiz ve Test\DeiT_Test_Output\test_detailed_results.csv',
        'output_dir': r'C:\Users\kivan\Desktop\tez_analiz\Deit Quiz ve Test\Veri_analizi_python'
    },

    'data_cleaning': {
        'skip_first_n_rows': 14,
        'duplicate_strategy': 'keep_highest_score',
        'exclude_conditions': {
            'position': 'resident',
            'experience': '<1'
        }
    },

    'classes': {
        'nv': 'Nevüs (Benign)',
        'mel': 'Melanom (Malign)',
        'bcc': 'Bazal Hücreli Karsinom',
        'ak': 'Aktinik Keratoz',
        'bkl': 'Benign Keratoz',
        'df': 'Dermatofibrom',
        'vasc': 'Vasküler Lezyon',
        'scc': 'Skuamöz Hücreli Karsinom'
    },

    'colors': {
        'model': '#3498DB',  # Mavi - AI Model
        'uzman': '#2ECC71',  # Yeşil - Uzman
        'asistan': '#E74C3C',  # Kırmızı - Asistan
        'resident': '#F39C12',  # Turuncu - Resident
        'success': '#27AE60',  # Koyu yeşil
        'error': '#E67E22',  # Turuncu
        'neutral': '#95A5A6',  # Gri
        'background': '#ECF0F1'  # Açık gri
    },

    'analysis': {
        'significance_level': 0.05,
        'confidence_interval': 0.95,
        'bootstrap_samples': 1000,
        'random_state': 42
    }
}


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Logging sistemini kur"""
    output_dir = Path(CONFIG['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"analiz_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


# =============================================================================
# DATA PROCESSOR CLASS
# =============================================================================

class DermatologyDataProcessor:
    """Dermatoloji verilerini işleme sınıfı"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.participant_data = None
        self.model_quiz_data = None
        self.model_test_data = None
        self.cleaned_data = None
        self.merged_data = None

    def load_data(self):
        """Veri dosyalarını yükle"""
        self.logger.info("📁 Veri dosyaları yükleniyor...")

        try:
            # Katılımcı sonuçları
            self.participant_data = pd.read_csv(CONFIG['paths']['participant_results'])
            self.logger.info(f"✅ Katılımcı verileri yüklendi: {len(self.participant_data)} kayıt")

            # Model quiz sonuçları
            self.model_quiz_data = pd.read_csv(CONFIG['paths']['model_quiz_results'])
            self.logger.info(f"✅ Model quiz verileri yüklendi: {len(self.model_quiz_data)} kayıt")

            # Model test sonuçları
            self.model_test_data = pd.read_csv(CONFIG['paths']['model_test_results'])
            self.logger.info(f"✅ Model test verileri yüklendi: {len(self.model_test_data)} kayıt")

            return True

        except Exception as e:
            self.logger.error(f"❌ Veri yükleme hatası: {e}")
            return False

    def clean_participant_data(self):
        """Katılımcı verilerini temizle"""
        self.logger.info("🧹 Katılımcı verileri temizleniyor...")

        if self.participant_data is None:
            self.logger.error("❌ Katılımcı verileri yüklenmemiş!")
            return False

        original_count = len(self.participant_data)

        # 1. İlk 14 satırı atla
        skip_rows = CONFIG['data_cleaning']['skip_first_n_rows']
        if skip_rows > 0:
            self.participant_data = self.participant_data.iloc[skip_rows:].reset_index(drop=True)
            self.logger.info(f"📝 İlk {skip_rows} satır atlandı")

        # 2. Duplike participant_id'leri temizle (en yüksek skoru tut)
        if 'participant_id' in self.participant_data.columns and 'success_rate' in self.participant_data.columns:
            # Success rate string ise float'a çevir
            if self.participant_data['success_rate'].dtype == 'object':
                self.participant_data['success_rate'] = self.participant_data['success_rate'].str.rstrip('%').astype(
                    float) / 100

            # En yüksek skorlu kayıtları tut
            self.participant_data = self.participant_data.loc[
                self.participant_data.groupby('participant_id')['success_rate'].idxmax()
            ].reset_index(drop=True)
            self.logger.info(f"🔄 Duplike participant_id'ler temizlendi (en yüksek skor tutuldu)")

        # 3. Deneyim filtreleme: position=resident ve experience=<1 olanları çıkar
        exclude_condition = (
                (self.participant_data['position'] == 'resident') &
                (self.participant_data['experience'] == '<1')
        )
        excluded_count = exclude_condition.sum()
        self.participant_data = self.participant_data[~exclude_condition].reset_index(drop=True)

        if excluded_count > 0:
            self.logger.info(f"🚫 {excluded_count} katılımcı deneyim kriteri nedeniyle çıkarıldı")

        # 4. Grup tanımlamaları oluştur
        self.participant_data['group'] = 'Diğer'

        # Position bazlı gruplandırma
        uzman_positions = ['specialist', 'attending', 'consultant', 'expert']
        resident_positions = ['resident', 'trainee']

        for pos in uzman_positions:
            mask = self.participant_data['position'].str.lower().str.contains(pos, na=False)
            self.participant_data.loc[mask, 'group'] = 'Uzman'

        for pos in resident_positions:
            mask = self.participant_data['position'].str.lower().str.contains(pos, na=False)
            self.participant_data.loc[mask, 'group'] = 'Resident'

        # Veri türlerini düzenle
        self._process_participant_columns()

        final_count = len(self.participant_data)
        self.logger.info(f"✅ Veri temizleme tamamlandı: {original_count} → {final_count} kayıt")

        # Grup dağılımını göster
        group_counts = self.participant_data['group'].value_counts()
        self.logger.info("👥 Grup dağılımı:")
        for group, count in group_counts.items():
            self.logger.info(f"   {group}: {count} katılımcı")

        return True

    def _process_participant_columns(self):
        """Katılımcı sütunlarını işle"""
        # Soru bazlı doğru cevap verilerini çıkar
        question_columns = []
        for i in range(1, 81):  # 80 soruya kadar
            q_correct_col = f'q{i}_is_correct'
            if q_correct_col in self.participant_data.columns:
                question_columns.append(q_correct_col)

        # Her soru için ayrı satır oluştur (long format)
        participant_long = []

        for idx, row in self.participant_data.iterrows():
            participant_id = row['participant_id']
            group = row['group']
            position = row['position']
            experience = row['experience']

            for i in range(1, 81):
                q_id_col = f'q{i}_id'
                q_isic_col = f'q{i}_isic_id'
                q_correct_col = f'q{i}_is_correct'
                q_selected_col = f'q{i}_selected_code'
                q_answer_col = f'q{i}_correct_code'

                if all(col in self.participant_data.columns for col in [q_id_col, q_correct_col]):
                    participant_long.append({
                        'participant_id': participant_id,
                        'group': group,
                        'position': position,
                        'experience': experience,
                        'question_id': row[q_id_col],
                        'isic_id': row.get(q_isic_col, ''),
                        'is_correct': row[q_correct_col] == 'TRUE' if pd.notna(row[q_correct_col]) else False,
                        'selected_answer': row.get(q_selected_col, ''),
                        'correct_answer': row.get(q_answer_col, '')
                    })

        self.cleaned_data = pd.DataFrame(participant_long)
        self.logger.info(f"📊 Long format veri oluşturuldu: {len(self.cleaned_data)} kayıt")

    def prepare_model_data(self):
        """Model verilerini hazırla"""
        self.logger.info("🤖 Model verileri hazırlanıyor...")

        if self.model_quiz_data is None:
            self.logger.error("❌ Model quiz verileri yüklenmemiş!")
            return False

        # Model performans sütunlarını belirle
        model_columns = [col for col in self.model_quiz_data.columns if col.endswith('_correct')]

        if not model_columns:
            self.logger.error("❌ Model performans sütunları bulunamadı!")
            return False

        # En iyi performanslı modeli seç (Simple Ensemble tercih edilir)
        preferred_models = ['Simple Ensemble_correct', 'CNN + TTA_correct', 'CNN Standard_correct']

        selected_model = None
        for model in preferred_models:
            if model in model_columns:
                selected_model = model
                break

        if selected_model is None:
            selected_model = model_columns[0]

        self.model_quiz_data['model_correct'] = self.model_quiz_data[selected_model]
        self.logger.info(f"🎯 Seçilen model: {selected_model}")

        # Sınıf etiketlerini düzenle
        if 'correct_diagnosis_mapped' in self.model_quiz_data.columns:
            self.model_quiz_data['true_class'] = self.model_quiz_data['correct_diagnosis_mapped']
        elif 'correct_diagnosis_code' in self.model_quiz_data.columns:
            self.model_quiz_data['true_class'] = self.model_quiz_data['correct_diagnosis_code']

        model_accuracy = self.model_quiz_data['model_correct'].mean()
        self.logger.info(f"📈 Model başarı oranı: {model_accuracy:.3f} ({model_accuracy * 100:.1f}%)")

        return True

    def merge_data(self):
        """Katılımcı ve model verilerini birleştir"""
        self.logger.info("🔗 Veriler birleştiriliyor...")

        if self.cleaned_data is None or self.model_quiz_data is None:
            self.logger.error("❌ Temizlenmiş veri bulunamadı!")
            return False

        # Question ID üzerinden birleştir
        merge_keys = ['question_id']
        if 'isic_id' in self.cleaned_data.columns and 'isic_id' in self.model_quiz_data.columns:
            merge_keys.append('isic_id')

        self.merged_data = pd.merge(
            self.cleaned_data,
            self.model_quiz_data[['question_id', 'isic_id', 'model_correct', 'true_class']],
            on='question_id',
            how='left'
        )

        merge_success = self.merged_data['model_correct'].notna().sum()
        total_records = len(self.merged_data)

        self.logger.info(f"✅ Veri birleştirme tamamlandı: {merge_success}/{total_records} kayıt eşleşti")

        return merge_success > 0


# =============================================================================
# STATISTICAL ANALYZER CLASS
# =============================================================================

class DermatologyStatAnalyzer:
    """İstatistiksel analiz sınıfı"""

    def __init__(self, data_processor):
        self.dp = data_processor
        self.logger = logging.getLogger(__name__)
        self.results = {}

    def descriptive_statistics(self):
        """Tanımlayıcı istatistikler"""
        self.logger.info("📊 Tanımlayıcı istatistikler hesaplanıyor...")

        results = {}

        # Katılımcı istatistikleri
        if self.dp.participant_data is not None:
            participant_summary = self.dp.participant_data.groupby('group').agg({
                'participant_id': 'count',
                'success_rate': ['mean', 'std', 'min', 'max', 'median'],
                'total_time_spent': ['mean',
                                     'std'] if 'total_time_spent' in self.dp.participant_data.columns else 'count'
            })

            results['participant_summary'] = participant_summary

            # Genel istatistikler
            total_participants = len(self.dp.participant_data)
            results['total_participants'] = total_participants

            # Grup dağılımı
            group_distribution = self.dp.participant_data['group'].value_counts()
            results['group_distribution'] = group_distribution.to_dict()

        # Model performansı
        if self.dp.model_quiz_data is not None:
            model_accuracy = self.dp.model_quiz_data['model_correct'].mean()
            model_performance_by_class = self.dp.model_quiz_data.groupby('true_class')['model_correct'].agg(
                ['mean', 'count'])

            results['model_performance'] = {
                'overall_accuracy': model_accuracy,
                'class_performance': model_performance_by_class.to_dict()
            }

        # Sınıf dağılımı
        if self.dp.merged_data is not None:
            class_distribution = self.dp.merged_data['true_class'].value_counts()
            results['class_distribution'] = class_distribution.to_dict()

        self.results['descriptive'] = results
        self.logger.info("✅ Tanımlayıcı istatistikler tamamlandı")

        return results

    def group_comparisons(self):
        """Grup karşılaştırmaları"""
        self.logger.info("⚖️ Grup karşılaştırmaları yapılıyor...")

        if self.dp.cleaned_data is None:
            self.logger.warning("⚠️ Temizlenmiş veri bulunamadı")
            return {}

        results = {}

        # Katılımcı bazlı başarı oranları
        participant_stats = self.dp.cleaned_data.groupby(['participant_id', 'group'])['is_correct'].mean().reset_index()

        # Uzman vs Resident karşılaştırması
        uzman_scores = participant_stats[participant_stats['group'] == 'Uzman']['is_correct']
        resident_scores = participant_stats[participant_stats['group'] == 'Resident']['is_correct']

        if len(uzman_scores) > 0 and len(resident_scores) > 0:
            # Normallik testi
            uzman_normal = shapiro(uzman_scores)[1] > 0.05 if len(uzman_scores) > 3 else False
            resident_normal = shapiro(resident_scores)[1] > 0.05 if len(resident_scores) > 3 else False

            if uzman_normal and resident_normal:
                # T-test
                t_stat, p_value = ttest_ind(uzman_scores, resident_scores)
                test_name = "Bağımsız Örneklem t-Testi"
            else:
                # Mann-Whitney U
                u_stat, p_value = mannwhitneyu(uzman_scores, resident_scores, alternative='two-sided')
                test_name = "Mann-Whitney U Testi"
                t_stat = u_stat

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(uzman_scores) - 1) * uzman_scores.var() +
                                  (len(resident_scores) - 1) * resident_scores.var()) /
                                 (len(uzman_scores) + len(resident_scores) - 2))
            cohens_d = (uzman_scores.mean() - resident_scores.mean()) / pooled_std if pooled_std > 0 else 0

            results['uzman_vs_resident'] = {
                'test_name': test_name,
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < CONFIG['analysis']['significance_level'],
                'uzman_mean': uzman_scores.mean(),
                'uzman_std': uzman_scores.std(),
                'uzman_n': len(uzman_scores),
                'resident_mean': resident_scores.mean(),
                'resident_std': resident_scores.std(),
                'resident_n': len(resident_scores),
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
            }

            self.logger.info(f"📈 Uzman vs Resident: p={p_value:.4f}, Cohen's d={cohens_d:.3f}")

        # Zaman karşılaştırması (eğer mevcut ise)
        if 'total_time_spent' in self.dp.participant_data.columns:
            time_comparison = self._compare_time_performance()
            results['time_comparison'] = time_comparison

        self.results['group_comparisons'] = results
        return results

    def model_vs_human_analysis(self):
        """Model vs İnsan analizi"""
        self.logger.info("🤖👨‍⚕️ Model vs İnsan analizi yapılıyor...")

        if self.dp.merged_data is None:
            self.logger.warning("⚠️ Birleştirilmiş veri bulunamadı")
            return {}

        results = {}

        # Soru bazlı karşılaştırma
        question_performance = self.dp.merged_data.groupby('question_id').agg({
            'is_correct': 'mean',
            'model_correct': 'first'
        }).dropna()

        if len(question_performance) > 0:
            human_acc = question_performance['is_correct']
            model_acc = question_performance['model_correct']

            # Paired t-test
            t_stat, p_value = ttest_rel(human_acc, model_acc)

            results['question_based'] = {
                'test_name': 'Paired t-Test (Soru Bazlı)',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < CONFIG['analysis']['significance_level'],
                'human_mean': human_acc.mean(),
                'model_mean': model_acc.mean(),
                'difference': model_acc.mean() - human_acc.mean(),
                'questions_analyzed': len(question_performance)
            }

        # Sınıf bazlı karşılaştırma
        class_performance = self.dp.merged_data.groupby('true_class').agg({
            'is_correct': 'mean',
            'model_correct': 'mean'
        })

        results['class_based'] = {}
        for class_name in class_performance.index:
            human_perf = class_performance.loc[class_name, 'is_correct']
            model_perf = class_performance.loc[class_name, 'model_correct']

            results['class_based'][class_name] = {
                'human_accuracy': human_perf,
                'model_accuracy': model_perf,
                'difference': model_perf - human_perf,
                'class_name_tr': CONFIG['classes'].get(class_name, class_name)
            }

        # Genel karşılaştırma
        overall_human = self.dp.merged_data['is_correct'].mean()
        overall_model = self.dp.merged_data['model_correct'].mean()

        results['overall'] = {
            'human_accuracy': overall_human,
            'model_accuracy': overall_model,
            'difference': overall_model - overall_human,
            'improvement_percentage': (
                        (overall_model - overall_human) / overall_human * 100) if overall_human > 0 else 0
        }

        self.results['model_vs_human'] = results
        self.logger.info(f"🎯 Genel karşılaştırma: İnsan={overall_human:.3f}, Model={overall_model:.3f}")

        return results

    def correlation_analysis(self):
        """Korelasyon analizi"""
        self.logger.info("📈 Korelasyon analizi yapılıyor...")

        results = {}

        if self.dp.participant_data is not None:
            # Deneyim ile performans korelasyonu
            if 'experience' in self.dp.participant_data.columns:
                # Experience string değerlerini numerik'e çevir
                exp_numeric = self._convert_experience_to_numeric(self.dp.participant_data['experience'])

                if exp_numeric is not None:
                    correlation, p_value = pearsonr(exp_numeric, self.dp.participant_data['success_rate'])

                    results['experience_performance'] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'significant': p_value < CONFIG['analysis']['significance_level'],
                        'interpretation': self._interpret_correlation(correlation)
                    }

        # Sınıf zorluk analizi
        if self.dp.merged_data is not None:
            class_difficulty = self.dp.merged_data.groupby('true_class')['is_correct'].mean()
            model_class_perf = self.dp.merged_data.groupby('true_class')['model_correct'].mean()

            # Sınıf zorluk-model performans korelasyonu
            common_classes = class_difficulty.index.intersection(model_class_perf.index)
            if len(common_classes) > 2:
                corr, p_val = pearsonr(
                    class_difficulty.loc[common_classes],
                    model_class_perf.loc[common_classes]
                )

                results['class_difficulty_model'] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'significant': p_val < CONFIG['analysis']['significance_level']
                }

        self.results['correlations'] = results
        return results

    def advanced_statistical_tests(self):
        """İleri düzey istatistiksel testler"""
        self.logger.info("🔬 İleri düzey istatistiksel testler yapılıyor...")

        results = {}

        # Çoklu karşılaştırma düzeltmesi için p-değerlerini topla
        p_values = []
        test_names = []

        # McNemar testi (Model vs İnsan için)
        if self.dp.merged_data is not None:
            mcnemar_table = self._create_mcnemar_table()
            if mcnemar_table is not None:
                try:
                    mcnemar_result = mcnemar(mcnemar_table, exact=False)
                    results['mcnemar_test'] = {
                        'statistic': mcnemar_result.statistic,
                        'p_value': mcnemar_result.pvalue,
                        'significant': mcnemar_result.pvalue < CONFIG['analysis']['significance_level']
                    }
                    p_values.append(mcnemar_result.pvalue)
                    test_names.append('McNemar Test')
                except Exception as e:
                    self.logger.warning(f"McNemar testi hesaplanamadı: {e}")

        # Bonferroni düzeltmesi
        if p_values:
            alpha = CONFIG['analysis']['significance_level']
            bonferroni_alpha = alpha / len(p_values)
            results['multiple_comparison_correction'] = {
                'original_alpha': alpha,
                'bonferroni_alpha': bonferroni_alpha,
                'adjusted_results': {}
            }

            for i, (test_name, p_val) in enumerate(zip(test_names, p_values)):
                results['multiple_comparison_correction']['adjusted_results'][test_name] = {
                    'original_p': p_val,
                    'significant_bonferroni': p_val < bonferroni_alpha
                }

        self.results['advanced_tests'] = results
        return results

    def _interpret_effect_size(self, cohens_d):
        """Cohen's d etki boyutunu yorumla"""
        if cohens_d < 0.2:
            return "Çok küçük etki"
        elif cohens_d < 0.5:
            return "Küçük etki"
        elif cohens_d < 0.8:
            return "Orta etki"
        else:
            return "Büyük etki"

    def _interpret_correlation(self, r):
        """Korelasyon katsayısını yorumla"""
        abs_r = abs(r)
        if abs_r < 0.3:
            strength = "Zayıf"
        elif abs_r < 0.7:
            strength = "Orta"
        else:
            strength = "Güçlü"

        direction = "Pozitif" if r > 0 else "Negatif"
        return f"{strength} {direction}"

    def _convert_experience_to_numeric(self, experience_col):
        """Deneyim sütununu numerik'e çevir"""
        try:
            numeric_exp = []
            for exp in experience_col:
                if pd.isna(exp):
                    numeric_exp.append(np.nan)
                elif exp == '<1':
                    numeric_exp.append(0.5)
                elif '-' in str(exp):
                    # "1-2", "3-5" gibi formatlar
                    parts = str(exp).split('-')
                    if len(parts) == 2:
                        numeric_exp.append((float(parts[0]) + float(parts[1])) / 2)
                    else:
                        numeric_exp.append(np.nan)
                elif '+' in str(exp):
                    # "10+" gibi formatlar
                    numeric_exp.append(float(str(exp).replace('+', '')))
                else:
                    try:
                        numeric_exp.append(float(exp))
                    except:
                        numeric_exp.append(np.nan)

            return pd.Series(numeric_exp).dropna()
        except Exception as e:
            self.logger.warning(f"Deneyim dönüşüm hatası: {e}")
            return None

    def _compare_time_performance(self):
        """Zaman performansı karşılaştırması"""
        time_by_group = self.dp.participant_data.groupby('group')['total_time_spent'].agg(['mean', 'std', 'count'])

        uzman_time = self.dp.participant_data[self.dp.participant_data['group'] == 'Uzman']['total_time_spent']
        resident_time = self.dp.participant_data[self.dp.participant_data['group'] == 'Resident']['total_time_spent']

        if len(uzman_time) > 0 and len(resident_time) > 0:
            t_stat, p_value = ttest_ind(uzman_time, resident_time)

            return {
                'test_name': 'Bağımsız Örneklem t-Testi (Süre)',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < CONFIG['analysis']['significance_level'],
                'uzman_mean_time': uzman_time.mean(),
                'resident_mean_time': resident_time.mean(),
                'time_difference': uzman_time.mean() - resident_time.mean()
            }
        return None

    def _create_mcnemar_table(self):
        """McNemar testi için kontenjans tablosu oluştur"""
        try:
            # Model ve insan kararlarını karşılaştır
            human_correct = self.dp.merged_data['is_correct']
            model_correct = self.dp.merged_data['model_correct']

            # 2x2 tablo oluştur
            both_correct = ((human_correct == True) & (model_correct == True)).sum()
            human_only = ((human_correct == True) & (model_correct == False)).sum()
            model_only = ((human_correct == False) & (model_correct == True)).sum()
            both_wrong = ((human_correct == False) & (model_correct == False)).sum()

            table = np.array([[both_correct, human_only],
                              [model_only, both_wrong]])

            return table
        except Exception as e:
            self.logger.warning(f"McNemar tablosu oluşturulamadı: {e}")
            return None


# =============================================================================
# VISUALIZATION CLASS
# =============================================================================

class DermatologyVisualizer:
    """Görselleştirme sınıfı"""

    def __init__(self, data_processor, analyzer):
        self.dp = data_processor
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(CONFIG['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Türkçe font ayarları
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

    def create_main_comparison_chart(self):
        """Ana karşılaştırma grafiği: 8 sınıf bazlı AI + Resident + Uzman"""
        self.logger.info("📊 Ana karşılaştırma grafiği oluşturuluyor...")

        if self.dp.merged_data is None:
            self.logger.warning("⚠️ Birleştirilmiş veri bulunamadı")
            return

        # Sınıf bazlı performansları hesapla
        class_performance = {}

        for class_code in CONFIG['classes'].keys():
            class_data = self.dp.merged_data[self.dp.merged_data['true_class'] == class_code]

            if len(class_data) > 0:
                # Model performansı
                model_acc = class_data['model_correct'].mean()

                # Uzman performansı
                uzman_data = class_data[class_data['group'] == 'Uzman']
                uzman_acc = uzman_data['is_correct'].mean() if len(uzman_data) > 0 else 0

                # Resident performansı
                resident_data = class_data[class_data['group'] == 'Resident']
                resident_acc = resident_data['is_correct'].mean() if len(resident_data) > 0 else 0

                class_performance[class_code] = {
                    'model': model_acc,
                    'uzman': uzman_acc,
                    'resident': resident_acc,
                    'class_name': CONFIG['classes'][class_code]
                }

        # Grafik oluştur
        fig, ax = plt.subplots(figsize=(16, 10))

        classes = list(class_performance.keys())
        class_names = [class_performance[cls]['class_name'] for cls in classes]

        x = np.arange(len(classes))
        width = 0.25

        model_scores = [class_performance[cls]['model'] for cls in classes]
        uzman_scores = [class_performance[cls]['uzman'] for cls in classes]
        resident_scores = [class_performance[cls]['resident'] for cls in classes]

        bars1 = ax.bar(x - width, model_scores, width, label='Yapay Zeka (AI)',
                       color=CONFIG['colors']['model'], alpha=0.8)
        bars2 = ax.bar(x, uzman_scores, width, label='Uzman Doktor',
                       color=CONFIG['colors']['uzman'], alpha=0.8)
        bars3 = ax.bar(x + width, resident_scores, width, label='Asistan Doktor',
                       color=CONFIG['colors']['resident'], alpha=0.8)

        # Değerleri göster
        for i, (m, u, r) in enumerate(zip(model_scores, uzman_scores, resident_scores)):
            ax.text(i - width, m + 0.02, f'{m:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax.text(i, u + 0.02, f'{u:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax.text(i + width, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        ax.set_xlabel('Dermatolojik Hastalık Sınıfları', fontsize=14, fontweight='bold')
        ax.set_ylabel('Başarı Oranı (Doğruluk)', fontsize=14, fontweight='bold')
        ax.set_title('Yapay Zeka vs İnsan Uzman Performans Karşılaştırması\n(8 Dermatolojik Hastalık Sınıfı Bazında)',
                     fontsize=16, fontweight='bold', pad=20)

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Y ekseni formatı
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

        plt.tight_layout()

        output_path = self.output_dir / 'ana_karsilastirma_8_sinif.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✅ Ana karşılaştırma grafiği kaydedildi: {output_path}")

    def create_participant_distribution_charts(self):
        """Katılımcı dağılım grafikleri"""
        self.logger.info("👥 Katılımcı dağılım grafikleri oluşturuluyor...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Katılımcı Demografik Dağılımları', fontsize=16, fontweight='bold')

        # 1. Grup dağılımı
        ax1 = axes[0, 0]
        group_counts = self.dp.participant_data['group'].value_counts()

        colors = [CONFIG['colors']['uzman'] if 'Uzman' in group else
                  CONFIG['colors']['resident'] if 'Resident' in group else
                  CONFIG['colors']['neutral'] for group in group_counts.index]

        wedges, texts, autotexts = ax1.pie(group_counts.values, labels=group_counts.index,
                                           autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Katılımcı Grup Dağılımı', fontweight='bold')

        # 2. Position dağılımı
        ax2 = axes[0, 1]
        if 'position' in self.dp.participant_data.columns:
            position_counts = self.dp.participant_data['position'].value_counts().head(6)
            bars = ax2.bar(range(len(position_counts)), position_counts.values,
                           color=CONFIG['colors']['neutral'], alpha=0.7)
            ax2.set_xticks(range(len(position_counts)))
            ax2.set_xticklabels(position_counts.index, rotation=45, ha='right')
            ax2.set_ylabel('Katılımcı Sayısı')
            ax2.set_title('Pozisyon Dağılımı', fontweight='bold')

            # Değerleri göster
            for bar, val in zip(bars, position_counts.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         str(int(val)), ha='center', va='bottom', fontweight='bold')

        # 3. Başarı oranı dağılımı
        ax3 = axes[1, 0]
        ax3.hist(self.dp.participant_data['success_rate'], bins=20,
                 color=CONFIG['colors']['neutral'], alpha=0.7, edgecolor='black')
        ax3.axvline(self.dp.participant_data['success_rate'].mean(),
                    color=CONFIG['colors']['error'], linestyle='--', linewidth=2,
                    label=f'Ortalama: {self.dp.participant_data["success_rate"].mean():.3f}')
        ax3.set_xlabel('Başarı Oranı')
        ax3.set_ylabel('Katılımcı Sayısı')
        ax3.set_title('Başarı Oranı Dağılımı', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Deneyim dağılımı
        ax4 = axes[1, 1]
        if 'experience' in self.dp.participant_data.columns:
            exp_counts = self.dp.participant_data['experience'].value_counts()
            bars = ax4.bar(range(len(exp_counts)), exp_counts.values,
                           color=CONFIG['colors']['success'], alpha=0.7)
            ax4.set_xticks(range(len(exp_counts)))
            ax4.set_xticklabels(exp_counts.index, rotation=45, ha='right')
            ax4.set_ylabel('Katılımcı Sayısı')
            ax4.set_title('Deneyim Süresi Dağılımı', fontweight='bold')

        plt.tight_layout()

        output_path = self.output_dir / 'katilimci_dagilim.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✅ Katılımcı dağılım grafikleri kaydedildi: {output_path}")

    def create_performance_boxplots(self):
        """Performans kutu grafikleri"""
        self.logger.info("📦 Performans kutu grafikleri oluşturuluyor...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Grup Bazlı Performans Karşılaştırması', fontsize=16, fontweight='bold')

        # Katılımcı bazlı başarı oranları
        if self.dp.cleaned_data is not None:
            participant_perf = self.dp.cleaned_data.groupby(['participant_id', 'group'])[
                'is_correct'].mean().reset_index()

            # 1. Başarı oranı kutu grafik
            ax1 = axes[0]
            groups = participant_perf['group'].unique()
            group_data = [participant_perf[participant_perf['group'] == group]['is_correct'] for group in groups]

            bp1 = ax1.boxplot(group_data, labels=groups, patch_artist=True)

            colors = [CONFIG['colors']['uzman'] if 'Uzman' in group else
                      CONFIG['colors']['resident'] if 'Resident' in group else
                      CONFIG['colors']['neutral'] for group in groups]

            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax1.set_ylabel('Başarı Oranı')
            ax1.set_title('Grup Bazlı Başarı Oranı Dağılımı', fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # 2. Zaman karşılaştırması (eğer mevcut ise)
        ax2 = axes[1]
        if 'total_time_spent' in self.dp.participant_data.columns:
            time_by_group = []
            group_labels = []

            for group in self.dp.participant_data['group'].unique():
                group_times = self.dp.participant_data[self.dp.participant_data['group'] == group]['total_time_spent']
                if len(group_times) > 0:
                    time_by_group.append(group_times)
                    group_labels.append(group)

            if time_by_group:
                bp2 = ax2.boxplot(time_by_group, labels=group_labels, patch_artist=True)

                for patch, group in zip(bp2['boxes'], group_labels):
                    color = (CONFIG['colors']['uzman'] if 'Uzman' in group else
                             CONFIG['colors']['resident'] if 'Resident' in group else
                             CONFIG['colors']['neutral'])
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax2.set_ylabel('Tamamlama Süresi (dakika)')
                ax2.set_title('Grup Bazlı Süre Karşılaştırması', fontweight='bold')
                ax2.grid(True, alpha=0.3)
        else:
            # Alternatif: Sınıf bazlı performans
            if self.dp.merged_data is not None:
                class_perf_by_group = self.dp.merged_data.groupby(['true_class', 'group'])['is_correct'].mean().unstack(
                    fill_value=0)

                class_perf_by_group.plot(kind='bar', ax=ax2,
                                         color=[CONFIG['colors']['uzman'], CONFIG['colors']['resident']])
                ax2.set_ylabel('Başarı Oranı')
                ax2.set_title('Sınıf Bazlı Grup Performansı', fontweight='bold')
                ax2.set_xlabel('Hastalık Sınıfı')
                ax2.legend(title='Grup')
                ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        output_path = self.output_dir / 'performans_kutu_grafikler.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✅ Performans kutu grafikleri kaydedildi: {output_path}")

    def create_confusion_matrices(self):
        """Karışıklık matrisleri"""
        self.logger.info("🔲 Karışıklık matrisleri oluşturuluyor...")

        if self.dp.merged_data is None:
            self.logger.warning("⚠️ Birleştirilmiş veri bulunamadı")
            return

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('Sınıf Bazlı Performans Confusion Matrix Karşılaştırması', fontsize=16, fontweight='bold')

        # Sınıf etiketleri
        class_labels = list(CONFIG['classes'].keys())
        class_names = [CONFIG['classes'][cls] for cls in class_labels]

        # 1. Model Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        model_cm = self._calculate_confusion_matrix_for_model()
        if model_cm is not None:
            sns.heatmap(model_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                        xticklabels=class_names[:model_cm.shape[1]],
                        yticklabels=class_names[:model_cm.shape[0]])
            ax1.set_title('Yapay Zeka Modeli\nConfusion Matrix', fontweight='bold')
            ax1.set_xlabel('Tahmin Edilen')
            ax1.set_ylabel('Gerçek')

        # 2. Uzman Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        uzman_cm = self._calculate_confusion_matrix_for_group('Uzman')
        if uzman_cm is not None:
            sns.heatmap(uzman_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
                        xticklabels=class_names[:uzman_cm.shape[1]],
                        yticklabels=class_names[:uzman_cm.shape[0]])
            ax2.set_title('Uzman Doktor\nConfusion Matrix', fontweight='bold')
            ax2.set_xlabel('Tahmin Edilen')
            ax2.set_ylabel('Gerçek')

        # 3. Resident Confusion Matrix
        ax3 = fig.add_subplot(gs[0, 2])
        resident_cm = self._calculate_confusion_matrix_for_group('Resident')
        if resident_cm is not None:
            sns.heatmap(resident_cm, annot=True, fmt='d', cmap='Oranges', ax=ax3,
                        xticklabels=class_names[:resident_cm.shape[1]],
                        yticklabels=class_names[:resident_cm.shape[0]])
            ax3.set_title('Asistan Doktor\nConfusion Matrix', fontweight='bold')
            ax3.set_xlabel('Tahmin Edilen')
            ax3.set_ylabel('Gerçek')

        # 4-6. Sınıf bazlı doğruluk karşılaştırması
        ax4 = fig.add_subplot(gs[1, :])

        class_accuracies = {}
        for class_code in CONFIG['classes'].keys():
            class_data = self.dp.merged_data[self.dp.merged_data['true_class'] == class_code]

            if len(class_data) > 0:
                model_acc = class_data['model_correct'].mean()
                uzman_data = class_data[class_data['group'] == 'Uzman']
                resident_data = class_data[class_data['group'] == 'Resident']

                class_accuracies[class_code] = {
                    'model': model_acc,
                    'uzman': uzman_data['is_correct'].mean() if len(uzman_data) > 0 else 0,
                    'resident': resident_data['is_correct'].mean() if len(resident_data) > 0 else 0
                }

        # Sınıf doğruluk çubuk grafik
        classes = list(class_accuracies.keys())
        x = np.arange(len(classes))
        width = 0.25

        model_accs = [class_accuracies[cls]['model'] for cls in classes]
        uzman_accs = [class_accuracies[cls]['uzman'] for cls in classes]
        resident_accs = [class_accuracies[cls]['resident'] for cls in classes]

        ax4.bar(x - width, model_accs, width, label='AI Model', color=CONFIG['colors']['model'], alpha=0.8)
        ax4.bar(x, uzman_accs, width, label='Uzman', color=CONFIG['colors']['uzman'], alpha=0.8)
        ax4.bar(x + width, resident_accs, width, label='Asistan', color=CONFIG['colors']['resident'], alpha=0.8)

        ax4.set_xlabel('Hastalık Sınıfları')
        ax4.set_ylabel('Doğruluk Oranı')
        ax4.set_title('Sınıf Bazlı Doğruluk Karşılaştırması', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([CONFIG['classes'][cls] for cls in classes], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        output_path = self.output_dir / 'confusion_matrix_karsilastirma.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✅ Confusion matrix grafikleri kaydedildi: {output_path}")

    def create_correlation_heatmap(self):
        """Korelasyon ısı haritası"""
        self.logger.info("🔥 Korelasyon ısı haritası oluşturuluyor...")

        # Numerik veriler için korelasyon matrisi oluştur
        if self.dp.participant_data is not None:
            # Katılımcı başarı oranları
            participant_success = self.dp.participant_data.groupby('participant_id')['success_rate'].first()

            # Deneyim verisi (numerik'e çevir)
            experience_numeric = self.analyzer._convert_experience_to_numeric(self.dp.participant_data['experience'])

            if len(participant_success) > 0 and experience_numeric is not None:
                # Ortak indeksleri bul
                common_participants = participant_success.index.intersection(experience_numeric.index)

                if len(common_participants) > 0:
                    corr_data = pd.DataFrame({
                        'Başarı_Oranı': participant_success.loc[common_participants],
                        'Deneyim_Yılı': experience_numeric.loc[common_participants]
                    })

                    # Sınıf bazlı performansları ekle
                    if self.dp.merged_data is not None:
                        for class_code, class_name in CONFIG['classes'].items():
                            class_performance = self.dp.merged_data[
                                self.dp.merged_data['true_class'] == class_code
                                ].groupby('participant_id')['is_correct'].mean()

                            # Ortak katılımcıları bul
                            common_in_class = corr_data.index.intersection(class_performance.index)
                            if len(common_in_class) > 0:
                                corr_data[f'{class_name}_Başarısı'] = class_performance.loc[common_in_class]

                    # Korelasyon matrisi hesapla
                    correlation_matrix = corr_data.corr()

                    # Isı haritası oluştur
                    fig, ax = plt.subplots(figsize=(12, 10))

                    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

                    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax,
                                fmt='.3f', annot_kws={'size': 10})

                    ax.set_title('Performans Parametreleri Korelasyon Matrisi\n(Pearson Korelasyon Katsayıları)',
                                 fontsize=14, fontweight='bold', pad=20)

                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()

                    output_path = self.output_dir / 'korelasyon_isi_haritasi.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()

                    self.logger.info(f"✅ Korelasyon ısı haritası kaydedildi: {output_path}")

    def create_statistical_results_table(self):
        """İstatistiksel sonuçlar tablosu"""
        self.logger.info("📋 İstatistiksel sonuçlar tablosu oluşturuluyor...")

        fig, ax = plt.subplots(figsize=(16, 10))
        fig.suptitle('İstatistiksel Test Sonuçları Özeti', fontsize=16, fontweight='bold')

        # Tablo verilerini hazırla
        table_data = []

        # Grup karşılaştırmaları
        if 'group_comparisons' in self.analyzer.results:
            comparisons = self.analyzer.results['group_comparisons']

            if 'uzman_vs_resident' in comparisons:
                result = comparisons['uzman_vs_resident']
                table_data.append([
                    result['test_name'],
                    f"Uzman: {result['uzman_mean']:.3f}±{result['uzman_std']:.3f} (n={result['uzman_n']})",
                    f"Resident: {result['resident_mean']:.3f}±{result['resident_std']:.3f} (n={result['resident_n']})",
                    f"{result['p_value']:.4f}",
                    "Anlamlı" if result['significant'] else "Anlamlı Değil",
                    f"d={result['cohens_d']:.3f}\n({result['effect_size_interpretation']})"
                ])

        # Model karşılaştırmaları
        if 'model_vs_human' in self.analyzer.results:
            model_results = self.analyzer.results['model_vs_human']

            if 'question_based' in model_results:
                result = model_results['question_based']
                table_data.append([
                    result['test_name'],
                    f"İnsan: {result['human_mean']:.3f}",
                    f"Model: {result['model_mean']:.3f}",
                    f"{result['p_value']:.4f}",
                    "Anlamlı" if result['significant'] else "Anlamlı Değil",
                    f"Δ={result['difference']:.3f}"
                ])

                f"Δ={result['difference']:.3f}\n({result['questions_analyzed']} soru)"
            ])

            # Korelasyon sonuçları
            if 'correlations' in self.analyzer.results:
                corr_results = self.analyzer.results['correlations']

            if 'experience_performance' in corr_results:
                result = corr_results['experience_performance']
            table_data.append([
            "Deneyim-Performans Korelasyonu",
            "Pearson Correlation",
            f"r={result['correlation']:.3f}",
            f"{result['p_value']:.4f}",
            "Anlamlı" if result['significant'] else "Anlamlı Değil",
            result['interpretation']
        ])

        if table_data:
            columns = ['Test', 'Grup 1', 'Grup 2', 'p-değeri', 'Anlamlılık', 'Etki Boyutu']

        table = ax.table(cellText=table_data, colLabels=columns,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Başlık renklendirme
        for i in range(len(columns)):
            table[(0, i)].set_facecolor(CONFIG['colors']['neutral'])
        table[(0, i)].set_text_props(weight='bold', color='white')

        # Anlamlılık sütunu renklendirme
        for i in range(1, len(table_data) + 1):
            anlamli_col = 4  # Anlamlılık sütunu
        if "Anlamlı" in table_data[i - 1][anlamli_col] and "Değil" not in table_data[i - 1][anlamli_col]:
            color = CONFIG['colors']['success']
        else:
            color = CONFIG['colors']['error']
        table[(i, anlamli_col)].set_facecolor(color)
        table[(i, anlamli_col)].set_text_props(weight='bold', color='white')

        ax.axis('off')
        plt.tight_layout()

        output_path = self.output_dir / 'istatistiksel_sonuclar_tablosu.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✅ İstatistiksel sonuçlar tablosu kaydedildi: {output_path}")

    def create_roc_analysis(self):
        """ROC eğrisi analizi"""
        self.logger.info("📈 ROC eğrisi analizi oluşturuluyor...")

        if self.dp.merged_data is None:
            self.logger.warning("⚠️ Birleştirilmiş veri bulunamadı")
            return

        # Sınıf bazlı ROC eğrileri
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Sınıf Bazlı ROC Eğrisi Karşılaştırması (Model vs İnsan)', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for idx, (class_code, class_name) in enumerate(CONFIG['classes'].items()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            class_data = self.dp.merged_data[self.dp.merged_data['true_class'] == class_code]

            if len(class_data) > 0:
                # Model için ROC
                y_true_model = (class_data['true_class'] == class_code).astype(int)
                y_scores_model = class_data['model_correct'].astype(float)

                # İnsan için ROC
                human_scores = class_data.groupby('participant_id')['is_correct'].mean()

                try:
                    # Model ROC
                    fpr_model, tpr_model, _ = roc_curve(y_true_model, y_scores_model)
                    auc_model = roc_auc_score(y_true_model, y_scores_model)

                    ax.plot(fpr_model, tpr_model, color=CONFIG['colors']['model'],
                            linewidth=2, label=f'AI Model (AUC={auc_model:.3f})')

                    # Diagonal line
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'{class_name}', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                except Exception as e:
                    ax.text(0.5, 0.5, f'ROC hesaplanamadı\n{str(e)[:50]}...',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{class_name}', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Veri Yok', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{class_name}', fontweight='bold')

        plt.tight_layout()

        output_path = self.output_dir / 'roc_egri_analizi.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✅ ROC eğrisi analizi kaydedildi: {output_path}")

    def create_time_analysis(self):
        """Zaman analizi grafikleri"""
        self.logger.info("⏱️ Zaman analizi grafikleri oluşturuluyor...")

        if 'total_time_spent' not in self.dp.participant_data.columns:
            self.logger.warning("⚠️ Zaman verisi bulunamadı")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Zaman Performansı Analizi', fontsize=16, fontweight='bold')

        # 1. Grup bazlı zaman dağılımı
        ax1 = axes[0, 0]
        time_by_group = []
        group_labels = []

        for group in self.dp.participant_data['group'].unique():
            group_times = self.dp.participant_data[self.dp.participant_data['group'] == group]['total_time_spent']
            if len(group_times) > 0:
                time_by_group.append(group_times)
                group_labels.append(group)

        if time_by_group:
            bp = ax1.boxplot(time_by_group, labels=group_labels, patch_artist=True)

            for patch, group in zip(bp['boxes'], group_labels):
                color = (CONFIG['colors']['uzman'] if 'Uzman' in group else
                         CONFIG['colors']['resident'] if 'Resident' in group else
                         CONFIG['colors']['neutral'])
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax1.set_ylabel('Toplam Süre (dakika)')
            ax1.set_title('Grup Bazlı Tamamlama Süresi', fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # 2. Başarı-zaman ilişkisi
        ax2 = axes[0, 1]
        ax2.scatter(self.dp.participant_data['total_time_spent'],
                    self.dp.participant_data['success_rate'],
                    alpha=0.6, color=CONFIG['colors']['neutral'])

        # Trend line
        z = np.polyfit(self.dp.participant_data['total_time_spent'],
                       self.dp.participant_data['success_rate'], 1)
        p = np.poly1d(z)
        ax2.plot(self.dp.participant_data['total_time_spent'],
                 p(self.dp.participant_data['total_time_spent']),
                 "r--", alpha=0.8, linewidth=2)

        ax2.set_xlabel('Toplam Süre (dakika)')
        ax2.set_ylabel('Başarı Oranı')
        ax2.set_title('Süre vs Başarı İlişkisi', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Ortalama cevap süresi analizi
        ax3 = axes[1, 0]
        if 'average_answer_time' in self.dp.participant_data.columns:
            ax3.hist(self.dp.participant_data['average_answer_time'],
                     bins=20, color=CONFIG['colors']['neutral'], alpha=0.7, edgecolor='black')
            ax3.axvline(self.dp.participant_data['average_answer_time'].mean(),
                        color=CONFIG['colors']['error'], linestyle='--', linewidth=2,
                        label=f'Ortalama: {self.dp.participant_data["average_answer_time"].mean():.2f}s')
            ax3.set_xlabel('Ortalama Cevap Süresi (saniye)')
            ax3.set_ylabel('Katılımcı Sayısı')
            ax3.set_title('Ortalama Cevap Süresi Dağılımı', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Deneyim-zaman ilişkisi
        ax4 = axes[1, 1]
        exp_groups = self.dp.participant_data.groupby('experience')['total_time_spent'].mean().sort_values()

        bars = ax4.bar(range(len(exp_groups)), exp_groups.values,
                       color=CONFIG['colors']['success'], alpha=0.7)
        ax4.set_xticks(range(len(exp_groups)))
        ax4.set_xticklabels(exp_groups.index, rotation=45, ha='right')
        ax4.set_ylabel('Ortalama Süre (dakika)')
        ax4.set_title('Deneyim Seviyesi vs Ortalama Süre', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        output_path = self.output_dir / 'zaman_performans_analizi.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✅ Zaman analizi grafikleri kaydedildi: {output_path}")

    def _calculate_confusion_matrix_for_model(self):
        """Model için confusion matrix hesapla"""
        try:
            if self.dp.merged_data is None:
                return None

            # Basitleştirilmiş binary confusion matrix (doğru/yanlış)
            class_accuracy = self.dp.merged_data.groupby('true_class')['model_correct'].sum()
            class_totals = self.dp.merged_data['true_class'].value_counts()

            # Sınıf bazlı doğru/yanlış matrisi
            confusion_data = []
            for class_code in CONFIG['classes'].keys():
                if class_code in class_accuracy.index:
                    correct = class_accuracy[class_code]
                    total = class_totals[class_code]
                    wrong = total - correct
                    confusion_data.append([correct, wrong])
                else:
                    confusion_data.append([0, 0])

            return np.array(confusion_data)

        except Exception as e:
            self.logger.warning(f"Model confusion matrix hesaplanamadı: {e}")
            return None

    def _calculate_confusion_matrix_for_group(self, group_name):
        """Belirtilen grup için confusion matrix hesapla"""
        try:
            if self.dp.merged_data is None:
                return None

            group_data = self.dp.merged_data[self.dp.merged_data['group'] == group_name]

            if len(group_data) == 0:
                return None

            class_accuracy = group_data.groupby('true_class')['is_correct'].sum()
            class_totals = group_data['true_class'].value_counts()

            # Sınıf bazlı doğru/yanlış matrisi
            confusion_data = []
            for class_code in CONFIG['classes'].keys():
                if class_code in class_accuracy.index:
                    correct = class_accuracy[class_code]
                    total = class_totals[class_code]
                    wrong = total - correct
                    confusion_data.append([correct, wrong])
                else:
                    confusion_data.append([0, 0])

            return np.array(confusion_data)

        except Exception as e:
            self.logger.warning(f"{group_name} grubu confusion matrix hesaplanamadı: {e}")
            return None


# =============================================================================
# REPORT GENERATOR CLASS
# =============================================================================

class DermatologyReportGenerator:
    """Rapor oluşturucu sınıfı"""

    def __init__(self, data_processor, analyzer, visualizer):
        self.dp = data_processor
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(CONFIG['paths']['output_dir'])

    def generate_json_report(self):
        """JSON formatında kapsamlı rapor oluştur"""
        self.logger.info("📄 JSON raporu oluşturuluyor...")

        report = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'analysis_type': 'Dermatoloji Uzmanlık Tezi Analizi',
                'version': '1.0.0',
                'researcher': 'arcankc',
                'significance_level': CONFIG['analysis']['significance_level']
            },
            'data_summary': {
                'total_participants': len(self.dp.participant_data) if self.dp.participant_data is not None else 0,
                'total_questions': 80,  # Quiz'de 80 soru
                'model_accuracy': self.dp.model_quiz_data[
                    'model_correct'].mean() if self.dp.model_quiz_data is not None else None,
                'classes_analyzed': list(CONFIG['classes'].keys())
            },
            'statistical_results': self.analyzer.results,
            'data_cleaning_log': {
                'rows_skipped': CONFIG['data_cleaning']['skip_first_n_rows'],
                'exclusion_criteria': CONFIG['data_cleaning']['exclude_conditions'],
                'duplicate_strategy': CONFIG['data_cleaning']['duplicate_strategy']
            }
        }

        # Ek hesaplanan metrikler
        if self.dp.merged_data is not None:
            # Sınıf bazlı performans özeti
            class_summary = {}
            for class_code, class_name in CONFIG['classes'].items():
                class_data = self.dp.merged_data[self.dp.merged_data['true_class'] == class_code]

                if len(class_data) > 0:
                    class_summary[class_code] = {
                        'name_tr': class_name,
                        'total_samples': len(class_data),
                        'model_accuracy': class_data['model_correct'].mean(),
                        'human_accuracy': class_data['is_correct'].mean(),
                        'uzman_accuracy': class_data[class_data['group'] == 'Uzman']['is_correct'].mean()
                        if len(class_data[class_data['group'] == 'Uzman']) > 0 else 0,
                        'resident_accuracy': class_data[class_data['group'] == 'Resident']['is_correct'].mean()
                        if len(class_data[class_data['group'] == 'Resident']) > 0 else 0
                    }

            report['class_performance_summary'] = class_summary

        # JSON dosyasını kaydet
        output_path = self.output_dir / 'kapsamli_analiz_raporu.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"✅ JSON raporu kaydedildi: {output_path}")

        return report

    def generate_excel_report(self):
        """Excel formatında analiz tabloları oluştur"""
        self.logger.info("📊 Excel raporu oluşturuluyor...")

        output_path = self.output_dir / 'dermatoloji_tezi_analiz_sonuclari.xlsx'

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. Tanımlayıcı İstatistikler
            if 'descriptive' in self.analyzer.results:
                desc_data = []

                # Katılımcı özeti
                if self.dp.participant_data is not None:
                    participant_summary = self.dp.participant_data.groupby('group').agg({
                        'participant_id': 'count',
                        'success_rate': ['mean', 'std', 'min', 'max']
                    }).round(4)
                    participant_summary.to_excel(writer, sheet_name='Katılımcı_Özeti')

                # Model performansı
                if self.dp.model_quiz_data is not None:
                    model_perf = self.dp.model_quiz_data.groupby('true_class')['model_correct'].agg([
                        'mean', 'count', 'sum'
                    ]).round(4)
                    model_perf.to_excel(writer, sheet_name='Model_Performansı')

            # 2. İstatistiksel Test Sonuçları
            if 'group_comparisons' in self.analyzer.results:
                test_results = []

                if 'uzman_vs_resident' in self.analyzer.results['group_comparisons']:
                    result = self.analyzer.results['group_comparisons']['uzman_vs_resident']
                    test_results.append({
                        'Test': result['test_name'],
                        'Uzman_Ortalama': result['uzman_mean'],
                        'Resident_Ortalama': result['resident_mean'],
                        'p_değeri': result['p_value'],
                        'Anlamlı': result['significant'],
                        'Cohens_d': result['cohens_d'],
                        'Etki_Boyutu': result['effect_size_interpretation']
                    })

                if test_results:
                    test_df = pd.DataFrame(test_results)
                    test_df.to_excel(writer, sheet_name='İstatistiksel_Testler', index=False)

            # 3. Sınıf Bazlı Performans
            if self.dp.merged_data is not None:
                class_performance = []

                for class_code, class_name in CONFIG['classes'].items():
                    class_data = self.dp.merged_data[self.dp.merged_data['true_class'] == class_code]

                    if len(class_data) > 0:
                        uzman_data = class_data[class_data['group'] == 'Uzman']
                        resident_data = class_data[class_data['group'] == 'Resident']

                        class_performance.append({
                            'Sınıf_Kodu': class_code,
                            'Sınıf_Adı': class_name,
                            'Toplam_Soru': len(class_data),
                            'Model_Başarı': class_data['model_correct'].mean(),
                            'Uzman_Başarı': uzman_data['is_correct'].mean() if len(uzman_data) > 0 else 0,
                            'Resident_Başarı': resident_data['is_correct'].mean() if len(resident_data) > 0 else 0,
                            'Uzman_Sayısı': len(uzman_data),
                            'Resident_Sayısı': len(resident_data)
                        })

                if class_performance:
                    class_df = pd.DataFrame(class_performance)
                    class_df.to_excel(writer, sheet_name='Sınıf_Bazlı_Performans', index=False)

            # 4. Model vs İnsan Karşılaştırması
            if 'model_vs_human' in self.analyzer.results:
                comparison_data = []

                if 'overall' in self.analyzer.results['model_vs_human']:
                    overall = self.analyzer.results['model_vs_human']['overall']
                    comparison_data.append({
                        'Karşılaştırma': 'Genel',
                        'İnsan_Başarı': overall['human_accuracy'],
                        'Model_Başarı': overall['model_accuracy'],
                        'Fark': overall['difference'],
                        'İyileştirme_Yüzdesi': overall['improvement_percentage']
                    })

                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    comp_df.to_excel(writer, sheet_name='Model_İnsan_Karşılaştırma', index=False)

        self.logger.info(f"✅ Excel raporu kaydedildi: {output_path}")

    def generate_markdown_report(self):
        """Markdown formatında tez raporu oluştur"""
        self.logger.info("📝 Markdown raporu oluşturuluyor...")

        report_content = f"""
        # Dermatoloji Uzmanlık Tezi - Veri Analizi Raporu

        **Tarih:** {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}  
        **Araştırmacı:** arcankc  
        **Analiz Tipi:** Yapay Zeka vs İnsan Uzman Performans Karşılaştırması  
        **Anlamlılık Düzeyi:** p < {CONFIG['analysis']['significance_level']}

        ## 📋 Analiz Özeti

        ### Veri Seti Bilgileri
        - **Toplam Katılımcı:** {len(self.dp.participant_data) if self.dp.participant_data is not None else 'N/A'}
        - **Toplam Soru Sayısı:** 80
        - **Analiz Edilen Sınıf Sayısı:** {len(CONFIG['classes'])}
        - **Model Genel Başarı:** {self.dp.model_quiz_data['model_correct'].mean():.3f} ({self.dp.model_quiz_data['model_correct'].mean() * 100:.1f}%) 
          (Model verileri mevcut ise)

        ### Veri Temizleme İşlemleri
        - İlk {CONFIG['data_cleaning']['skip_first_n_rows']} satır analiz dışı bırakıldı
        - Aynı katılımcıdan birden fazla denemede en yüksek skor tutuldu
        - Deneyim kriteri: position="resident" ve experience="<1" olan veriler çıkarıldı

        ## 📊 Ana Bulgular

        ### Grup Karşılaştırması
        """

        # Grup karşılaştırma sonuçlarını ekle
        if 'group_comparisons' in self.analyzer.results:
            if 'uzman_vs_resident' in self.analyzer.results['group_comparisons']:
                result = self.analyzer.results['group_comparisons']['uzman_vs_resident']

                report_content += f"""
        **Uzman vs Resident Karşılaştırması:**
        - **Test:** {result['test_name']}
        - **Uzman Başarı Oranı:** {result['uzman_mean']:.3f} ± {result['uzman_std']:.3f} (n={result['uzman_n']})
        - **Resident Başarı Oranı:** {result['resident_mean']:.3f} ± {result['resident_std']:.3f} (n={result['resident_n']})
        - **p-değeri:** {result['p_value']:.4f}
        - **İstatistiksel Anlamlılık:** {'Anlamlı' if result['significant'] else 'Anlamlı Değil'}
        - **Etki Boyutu (Cohen's d):** {result['cohens_d']:.3f} ({result['effect_size_interpretation']})
        """

        # Model vs İnsan karşılaştırması
        if 'model_vs_human' in self.analyzer.results:
            if 'overall' in self.analyzer.results['model_vs_human']:
                overall = self.analyzer.results['model_vs_human']['overall']

                report_content += f"""
        ### Model vs İnsan Performansı
        - **İnsan Genel Başarısı:** {overall['human_accuracy']:.3f} ({overall['human_accuracy'] * 100:.1f}%)
        - **Model Genel Başarısı:** {overall['model_accuracy']:.3f} ({overall['model_accuracy'] * 100:.1f}%)
        - **Performans Farkı:** {overall['difference']:.3f}
        - **İyileştirme Oranı:** {overall['improvement_percentage']:.1f}%
        """

        # Sınıf bazlı performans
        report_content += """
        ### Sınıf Bazlı Performans

        | Hastalık Sınıfı | Model Başarısı | Uzman Başarısı | Resident Başarısı |
        |------------------|----------------|----------------|-------------------|
        """

        if self.dp.merged_data is not None:
            for class_code, class_name in CONFIG['classes'].items():
                class_data = self.dp.merged_data[self.dp.merged_data['true_class'] == class_code]

                if len(class_data) > 0:
                    model_acc = class_data['model_correct'].mean()
                    uzman_data = class_data[class_data['group'] == 'Uzman']
                    resident_data = class_data[class_data['group'] == 'Resident']

                    uzman_acc = uzman_data['is_correct'].mean() if len(uzman_data) > 0 else 0
                    resident_acc = resident_data['is_correct'].mean() if len(resident_data) > 0 else 0

                    report_content += f"| {class_name} | {model_acc:.3f} | {uzman_acc:.3f} | {resident_acc:.3f} |\n"

        report_content += """
        ## 🎯 Sonuç ve Öneriler

        ### Ana Bulgular Özeti
        1. **Uzman vs Resident Performansı:** [Bulgulara göre yorumlanacak]
        2. **AI Model Performansı:** [Bulgulara göre yorumlanacak]
        3. **Sınıf Bazlı Zorluklar:** [En zor/kolay sınıflar belirtilecek]

        ### Klinik Öneriler
        1. **Eğitim Programları:** Resident eğitiminde AI destekli sistemlerin kullanımı
        2. **Tanı Desteği:** Belirli hastalık sınıflarında AI desteğinin önemi
        3. **Kalite Kontrol:** Tanı doğruluğunu artırmak için öneriler

        ### Araştırma Sınırlılıkları
        1. Örneklem büyüklüğü dengesizliği (Resident > Uzman)
        2. Quiz formatının klinik pratiği tam yansıtmaması
        3. AI model sınırlamaları

        ### Gelecek Çalışmalar
        1. Daha geniş örneklem ile tekrar
        2. Gerçek klinik vaka analizleri
        3. Longitudinal performans takibi

        ---

        **Rapor Oluşturma Tarihi:** {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}  
        **GitHub:** https://github.com/arcankc/Statistics
        """

        # Markdown dosyasını kaydet
        output_path = self.output_dir / 'dermatoloji_tez_raporu.md'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"✅ Markdown raporu kaydedildi: {output_path}")

    def create_thesis_tables(self):
        """Tez için hazır LaTeX tabloları oluştur"""
        self.logger.info("📋 Tez tabloları oluşturuluyor...")

        # Tez için hazır tablolar
        output_path = self.output_dir / 'tez_tablolari.txt'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("DERMATOLOJI TEZİ - HAZIR TABLOLAR\n")
            f.write("=" * 50 + "\n\n")

            # Tablo 1: Katılımcı Demografikleri
            f.write("TABLO 1: Katılımcı Demografik Özellikleri\n")
            f.write("-" * 40 + "\n")

            if self.dp.participant_data is not None:
                group_summary = self.dp.participant_data.groupby('group').agg({
                    'participant_id': 'count',
                    'success_rate': ['mean', 'std']
                }).round(3)

                f.write(str(group_summary))
                f.write("\n\n")

            # Tablo 2: İstatistiksel Test Sonuçları
            f.write("TABLO 2: İstatistiksel Test Sonuçları\n")
            f.write("-" * 40 + "\n")

            if 'group_comparisons' in self.analyzer.results:
                if 'uzman_vs_resident' in self.analyzer.results['group_comparisons']:
                    result = self.analyzer.results['group_comparisons']['uzman_vs_resident']

                    f.write(f"Test: {result['test_name']}\n")
                    f.write(f"Uzman: {result['uzman_mean']:.3f} ± {result['uzman_std']:.3f}\n")
                    f.write(f"Resident: {result['resident_mean']:.3f} ± {result['resident_std']:.3f}\n")
                    f.write(f"p-değeri: {result['p_value']:.4f}\n")
                    f.write(f"Cohen's d: {result['cohens_d']:.3f}\n")
                    f.write("\n")

            # Tablo 3: Sınıf Bazlı Performans
            f.write("TABLO 3: Hastalık Sınıfı Bazlı Performans Karşılaştırması\n")
            f.write("-" * 60 + "\n")
            f.write("Sınıf\t\tModel\tUzman\tResident\tToplam Soru\n")
            f.write("-" * 60 + "\n")

            if self.dp.merged_data is not None:
                for class_code, class_name in CONFIG['classes'].items():
                    class_data = self.dp.merged_data[self.dp.merged_data['true_class'] == class_code]

                    if len(class_data) > 0:
                        model_acc = class_data['model_correct'].mean()
                        uzman_data = class_data[class_data['group'] == 'Uzman']
                        resident_data = class_data[class_data['group'] == 'Resident']

                        uzman_acc = uzman_data['is_correct'].mean() if len(uzman_data) > 0 else 0
                        resident_acc = resident_data['is_correct'].mean() if len(resident_data) > 0 else 0

                        f.write(
                            f"{class_name[:15]:<15}\t{model_acc:.3f}\t{uzman_acc:.3f}\t{resident_acc:.3f}\t{len(class_data)}\n")

        self.logger.info(f"✅ Tez tabloları kaydedildi: {output_path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def bootstrap_analysis(data1, data2, n_bootstrap=1000, confidence_level=0.95):
    """Bootstrap analizi ile güven aralıkları hesapla"""
    np.random.seed(CONFIG['analysis']['random_state'])

    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample1 = np.random.choice(data1, len(data1), replace=True)
        sample2 = np.random.choice(data2, len(data2), replace=True)

        # Calculate difference in means
        diff = np.mean(sample1) - np.mean(sample2)
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_diffs, (alpha / 2) * 100)
    upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

    return {
        'mean_difference': np.mean(bootstrap_diffs),
        'confidence_interval': (lower, upper),
        'confidence_level': confidence_level,
        'bootstrap_samples': n_bootstrap
    }


def calculate_power_analysis(effect_size, n1, n2, alpha=0.05):
    """İstatistiksel güç analizi hesapla"""
    try:
        from scipy.stats import norm

        # Cohen's d için güç analizi
        pooled_n = (n1 * n2) / (n1 + n2)
        ncp = effect_size * np.sqrt(pooled_n / 2)  # Non-centrality parameter

        # Critical value for two-tailed test
        critical_value = norm.ppf(1 - alpha / 2)

        # Power calculation
        power = 1 - norm.cdf(critical_value - ncp) + norm.cdf(-critical_value - ncp)

        return {
            'power': power,
            'effect_size': effect_size,
            'sample_size_1': n1,
            'sample_size_2': n2,
            'alpha': alpha
        }
    except Exception as e:
        logging.getLogger(__name__).warning(f"Güç analizi hesaplanamadı: {e}")
        return None


def apply_multiple_comparison_correction(p_values, method='bonferroni'):
    """Çoklu karşılaştırma düzeltmesi uygula"""
    if method == 'bonferroni':
        adjusted_alpha = CONFIG['analysis']['significance_level'] / len(p_values)
        adjusted_p_values = [p * len(p_values) for p in p_values]
    elif method == 'fdr':
        # False Discovery Rate (Benjamini-Hochberg)
        sorted_p = np.sort(p_values)
        sorted_indices = np.argsort(p_values)
        m = len(p_values)

        adjusted_p_values = []
        for i, p in enumerate(sorted_p):
            adjusted_p = p * m / (i + 1)
            adjusted_p_values.append(min(adjusted_p, 1.0))

        # Restore original order
        final_adjusted = [0] * len(p_values)
        for i, idx in enumerate(sorted_indices):
            final_adjusted[idx] = adjusted_p_values[i]

        adjusted_p_values = final_adjusted
        adjusted_alpha = CONFIG['analysis']['significance_level']

    return {
        'method': method,
        'original_p_values': p_values,
        'adjusted_p_values': adjusted_p_values,
        'adjusted_alpha': adjusted_alpha,
        'significant_after_correction': [p < adjusted_alpha for p in adjusted_p_values]
    }


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Ana analiz fonksiyonu"""
    logger = setup_logging()

    try:
        logger.info("🚀 Dermatoloji tez analizi başlatılıyor...")

        # 1. Veri İşleme
        logger.info("📊 1. VERİ İŞLEME AŞAMASI")
        logger.info("-" * 30)

        data_processor = DermatologyDataProcessor()

        if not data_processor.load_data():
            logger.error("❌ Veri yüklenemedi!")
            return False

        if not data_processor.clean_participant_data():
            logger.error("❌ Veri temizlenemedi!")
            return False

        if not data_processor.prepare_model_data():
            logger.error("❌ Model verileri hazırlanamadı!")
            return False

        if not data_processor.merge_data():
            logger.error("❌ Veriler birleştirilemedi!")
            return False

        # 2. İstatistiksel Analiz
        logger.info("\n📊 2. İSTATİSTİKSEL ANALİZ AŞAMASI")
        logger.info("-" * 35)

        analyzer = DermatologyStatAnalyzer(data_processor)

        analyzer.descriptive_statistics()
        analyzer.group_comparisons()
        analyzer.model_vs_human_analysis()
        analyzer.correlation_analysis()
        analyzer.advanced_statistical_tests()

        # 3. Görselleştirme
        logger.info("\n🎨 3. GÖRSELLEŞTİRME AŞAMASI")
        logger.info("-" * 30)

        visualizer = DermatologyVisualizer(data_processor, analyzer)

        visualizer.create_main_comparison_chart()
        visualizer.create_participant_distribution_charts()
        visualizer.create_performance_boxplots()
        visualizer.create_confusion_matrices()
        visualizer.create_correlation_heatmap()
        visualizer.create_statistical_results_table()
        visualizer.create_roc_analysis()
        visualizer.create_time_analysis()

        # 4. Rapor Oluşturma
        logger.info("\n📋 4. RAPOR OLUŞTURMA AŞAMASI")
        logger.info("-" * 30)

        report_generator = DermatologyReportGenerator(data_processor, analyzer, visualizer)

        report_generator.generate_json_report()
        report_generator.generate_excel_report()
        report_generator.generate_markdown_report()
        report_generator.create_thesis_tables()

        # 5. Özet Bilgiler
        logger.info("\n📈 ANALİZ ÖZETİ")
        logger.info("-" * 20)

        # Ana bulgular
        if 'group_comparisons' in analyzer.results:
            if 'uzman_vs_resident' in analyzer.results['group_comparisons']:
                result = analyzer.results['group_comparisons']['uzman_vs_resident']
                logger.info(f"👨‍⚕️ Uzman Başarı: {result['uzman_mean']:.3f} ({result['uzman_mean'] * 100:.1f}%)")
                logger.info(
                    f"👨‍🎓 Resident Başarı: {result['resident_mean']:.3f} ({result['resident_mean'] * 100:.1f}%)")
                logger.info(f"📊 p-değeri: {result['p_value']:.4f}")
                logger.info(f"🎯 Anlamlılık: {'Anlamlı' if result['significant'] else 'Anlamlı Değil'}")

        if 'model_vs_human' in analyzer.results:
            if 'overall' in analyzer.results['model_vs_human']:
                overall = analyzer.results['model_vs_human']['overall']
                logger.info(f"🤖 Model Başarı: {overall['model_accuracy']:.3f} ({overall['model_accuracy'] * 100:.1f}%)")
                logger.info(f"👥 İnsan Başarı: {overall['human_accuracy']:.3f} ({overall['human_accuracy'] * 100:.1f}%)")
                logger.info(f"📈 Fark: {overall['difference']:.3f}")

        logger.info(f"\n📁 Tüm çıktılar hazır: {CONFIG['paths']['output_dir']}")
        logger.info("🎓 Dermatoloji uzmanlık teziniz için analizler tamamlandı!")

        return True

    except Exception as e:
        logger.error(f"❌ Analiz sırasında hata oluştu: {e}")
        logger.error(f"Hata detayı: {traceback.format_exc()}")
        return False


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("🚀 Dermatoloji Uzmanlık Tezi - Kapsamlı Veri Analizi Sistemi")
    print(f"👤 Geliştirici: arcankc (GitHub: https://github.com/arcankc)")
    print(f"📅 Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print(f"🏥 Amaç: AI Model vs İnsan Uzman Performans Karşılaştırması")
    print("=" * 70)

    # Ana analizi çalıştır
    success = main()

    if success:
        print("\n🎉 TÜM ANALİZLER BAŞARIYLA TAMAMLANDI!")
        print("=" * 50)
        print("📊 Dermatoloji teziniz için hazır:")
        print("   • Yüksek kaliteli grafikler (300 DPI)")
        print("   • İstatistiksel test sonuçları")
        print("   • Kapsamlı analiz raporları")
        print("   • JSON ve Excel veri formatları")
        print("   • Markdown tez raporu")
        print("   • LaTeX tabloları")
        print("   • ROC eğrileri ve confusion matrices")

        print("\n🎯 Sonraki Adımlar:")
        print("   1. Grafikleri tez formatına uygun şekilde düzenleyin")
        print("   2. İstatistiksel anlamlılık sonuçlarını yorumlayın")
        print("   3. Model performansının klinik etkileri üzerine tartışın")
        print("   4. Deneyim süresi bulgularını eğitim önerileriyle ilişkilendirin")
        print("   5. Sınırlılıklar ve gelecek çalışmalar bölümünü yazın")

        print("\n📞 Destek:")
        print("   GitHub: https://github.com/arcankc/Statistics")
        print("   🔬 Dermatoloji AI araştırmaları için optimize edilmiştir")

    else:
        print("\n💥 Analiz başarısız oldu!")
        print("🔧 Log dosyasını kontrol edin ve gerekli düzeltmeleri yapın.")
        print(f"📁 Log konumu: {CONFIG['paths']['output_dir']}")

    print(f"\n🏆 Tez yazımında başarılar dileriz!")
