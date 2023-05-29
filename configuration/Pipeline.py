# Import

import pandas as pd
import numpy as np
import re
from scipy.stats import norm
from sklearn.impute import KNNImputer


class Pipeline():
    def __init__(self, raw_df: pd.DataFrame):
        self.clean_z_test_cols = {
            'height': 0.05,
            'body_weight': 0.01,
        }
        self.categoraical_cols = ['Urine Ketone', 'Urine WBC',
                                  'Urine Color', 'Urine Sugar', 'Urine Leukocytes', 'Peritonitis/abdominal guarding', 'Migration of pain',
                                  'Tenderness in right lower quadrant', 'Rebound tenderness',
                                  'Cough tenderness', 'Nausea/vomiting', 'Anorexia', 'Dysuria', 'Stool']

        self.categ_dict = {'Negative': 1,
                           'Trace': 2, 'Positive': 2,
                           'Positive +1': 3, 'Positive +2': 4,
                           'Positive +3': 5, 'Positive +4': 6,
                           'Positive +5': 7}
        self.urine_color_dict = {'Colorless': 1,
                                 'Light yellow': 2,
                                 'Yellow': 3,
                                 'Deep yellow': 4,
                                 'Orange': 5,
                                 'Amber': 6,
                                 'Brown': 7,
                                 'Red': 8}
        self.raw_df = raw_df

    def clean_outliers_Z_test(self, df: pd.DataFrame, x, column, alpha):
        data = df.loc[df[column].isna()
                      == False, column].values
        mean = np.mean(data)
        std = np.std(data)

        z_threshold = norm.ppf(1 - (alpha/2))

        z_scores = np.abs((data - mean) / std)

        outliers = np.where(z_scores > z_threshold)
        non_outliers = np.where(z_scores <= z_threshold)

        if len(outliers[0]) != 0:
            min_outliers = min(data[outliers])
            max_outliers = max(data[outliers])
            min_non_outliers = min(data[non_outliers])
            max_non_outliers = max(data[non_outliers])
            if (x[column] <= max_non_outliers) and (x[column] >= min_non_outliers):
                return x[column]
            else:
                return None

        else:
            return x[column]

    def data_cleaning(self, df: pd.DataFrame):
        for col, alpha in self.clean_z_test_cols.items():
            df[col] = df.apply(
                lambda x: self.clean_outliers_Z_test(df, x, col, alpha), axis=1)
        return df

    def imputation(self, df: pd.DataFrame):
        # Only numerical cols
        df_numeric = df.select_dtypes(include=[np.number])
        # KNN Imuter
        imputer = KNNImputer(n_neighbors=3)
        imputed = imputer.fit_transform(df_numeric)
        knn_df_imputed = pd.DataFrame(imputed, columns=df_numeric.columns)
        # Digit
        for col in self.categoraical_cols:
            knn_df_imputed[col] = knn_df_imputed[col].apply(
                lambda x: int(round(x)))
        return knn_df_imputed

    def extract_2_condition(self, row, keywords_1, keywords_2):
        if row['Conditions']:
            text = str(row['Conditions']).lower()
            match_1 = re.search("|".join(keywords_1), text)
            match_2 = re.search("|".join(keywords_2), text)
            if match_1:
                return 1
            elif match_2:
                return 2
            else:
                return None
        else:
            return None

    def extract_3_condition(self, row, keywords_1, keywords_2, keywords_3):
        if row['Conditions']:
            text = str(row['Conditions']).lower()
            match_1 = re.search("|".join(keywords_1), text)
            match_2 = re.search("|".join(keywords_2), text)
            match_3 = re.search("|".join(keywords_3), text)
            if match_1:
                return 1
            elif match_2:
                return 2
            elif match_3:
                return 3
            else:
                return None
        else:
            return None

    def feature_engineering(self, df: pd.DataFrame):
        # Prefix ไม่มีอาการ
        not_or_no = '(ไม่|ไม่มี)'
        # Prefix มีอาการ; ต้องไม่มีคำว่า (ไม่มี or ไม่)นำหน้าอาการ
        non_no_or_not = '(?<!ไม่)'
        # Verb
        hurt = '(ปวด|ปว[ก-ฮ]?|เจ็บ|เจ็[ก-ฮ]?)'
        hurt_en = '(p[ai|ia]n|hurt)'

        # 1. Peritonitis/abdominal guarding
        # Adjective
        noy = '(น้อย|น้อ[ก-ฮ]?)'
        s = '[\s|\S]'

        # Noun
        abdomen = '(ท้อง|ท้อ[ก-ฮ]?)'
        abdomen_en = '(abdominal|abdomen)'
        localized = f'(ขวาล่าง|ขวาล่าง?|ขวา|ขวา?|rlq)'
        epigastrium = '(ลิ้นปี่|ลิ้นปี|ลิ้[ก-ฮ]?ปี่)'

        # Preposition
        position = '(ข้าง|ด้าน|ตรง|บริเวณ|ที่)'

        # ไม่มีอาการ ปวดท้อง
        no_perit_abdominal = [f'{not_or_no}(อาการ)?{hurt}{abdomen}']
        # มีอาการ ปวดท้องทั่วๆ, กลางท้อง, รอบสะดือ, ลิ้นปี่
        generalized = [
            f'{non_no_or_not}(มี)?(อาการ)?{hurt}(ทั้ง|ทั่ว(ๆ)?|กลาง){abdomen}',
            f'{non_no_or_not}(มี)?(อาการ)?{hurt}(ทั้ง|ทั่ว(ๆ)?|กลาง).*{abdomen}',
            f'{non_no_or_not}(มี)?(อาการ)?{hurt}(สดือ|{epigastrium})',
        ]
        # มีอาการ ปวดท้อง (ด้าน|ข้าง) ขวา, ขวาล่าง, ท้องน้อย, rlq
        localized = [
            f'{non_no_or_not}(มี)?(อาการ)?{hurt}{abdomen}{localized}',
            f'{hurt}{abdomen}{localized}',
            f'{hurt}{localized}{position}',
        ]
        df['Peritonitis/abdominal guarding'] = df.apply(lambda x: self.extract_3_condition(
            x, no_perit_abdominal, generalized, localized), axis=1)

        # 2. Migration of pain
        having = '(มีการ|มี)'
        no_not = '(no|not)'
        migration = '(เปลี่ยน|ย้า[ก-ฮ]?|เคลื่อน|ขยับ|ย้า[ก-ฮ]?มา|มา)'
        migration_en = '(migrate|transfer|movement|change|move)'
        epigastrium = '(ลิ้นปี่|ลิ้นปี|ลิ้[ก-ฮ]?ปี่)'
        epigastrium_en = 'epigastrium'
        right_lower_quandarant = '(ขวาล่าง|ขวา)'
        right_lower_quandarant_en = '(rlq|right lower quadrant|right lower|lower right|right l[o?]w[or|ro] q[ua|au]drant)'

        no_migration = [
            f'ไม่{having}?(อาการ)?{migration}{hurt}',
            f'ไม่มี{hurt}.*{abdomen}{migration}.*({epigastrium}|{right_lower_quandarant})',
            f'ไม่{having}?(อาการ)?{hurt}.*{abdomen}{migration}',
            f'{hurt}.*{abdomen}ไม่{having}{migration}',
        ]

        migration = [
            f'{migration}{hurt}',
            f'(มี)?(อาการ)?{migration}{hurt}',
            f'(มี)?(อาการ)?{migration}{hurt}{s}({epigastrium}|{epigastrium_en}|{right_lower_quandarant}|{right_lower_quandarant_en})',
            f'(มี)?(อาการ)?{migration}{position}{hurt}',
        ]

        df['Migration of pain'] = df.apply(
            lambda x: self.extract_2_condition(x, no_migration, migration), axis=1)

        # 3. RLQ
        # white space or non-space
        s = '[\s|\S]'
        negative = '(no|not|negative|neg)'

        rlq = [
            f'{non_no_or_not}{hurt}{abdomen}{right_lower_quandarant}',
            f'{non_no_or_not}{hurt}{abdomen}.*{position}.*{right_lower_quandarant}',
            f'{non_no_or_not}(มี)?(อาการ)?{hurt}{abdomen}{right_lower_quandarant}',
            f'{non_no_or_not}(มี)?(อาการ)?{hurt}{abdomen}.*{right_lower_quandarant}',
            f'{right_lower_quandarant_en}',
        ]

        no_rlq = [
            f'ไม่{hurt}.*{abdomen}',
            f'ไม่มี(อาการ)?{hurt}.*{abdomen}',
            f'{hurt}.*{abdomen}(ไม่มาก|เบา|ปกติ|ธรรมดา)',
            f'{negative}{s}{right_lower_quandarant_en}(tender|tendernes[ss|s])',
        ]

        df['Tenderness in right lower quadrant'] = df.apply(
            lambda x: self.extract_2_condition(x, no_rlq, rlq), axis=1)
        # 4. Rebound Tenderness
        guarding = '(เกร็ง|เกร็ง?)'

        # Conjunction
        when = '(ตอน|เมื่อ|เวลา|ขณะ)'
        # Verb
        hurt = '(ปวด|ปว[ก-ฮ]?|เจ็บ|เจ็[ก-ฮ]?)'
        rebound_en = '(reboud|rebound|rebounding|rebounded)'
        guarding_en = '(guard|guarding)'
        tender = '(tender|tenderness)'
        press = '(กด|ก[ก-ฮ])'
        tender_conditions = [
            f'{non_no_or_not}(อาการ)?{hurt}{when}?{press}{abdomen}',
            f'กด.*{abdomen}',
            f'{non_no_or_not}(อาการ)?{hurt}{abdomen}(มาก|ปานกลาง|น้อย)',
            f'{non_no_or_not}(อาการ)?{hurt}{when}?(ขยับ|เคลื่อน)',
            f'{non_no_or_not}(อาการ)?{abdomen}{guarding}',
            f'{non_no_or_not}(อาการ)?{guarding}{abdomen}',
            f'{guarding_en}',
            f'{rebound_en}',
            f'{guarding_en}{s}{tender}',
            f'{rebound_en}{s}{tender}',
        ]

        no_tender_conditions = [
            f'ไม่มี(อาการ)?{hurt}.*{press}.*{abdomen}',
            f'ไม่มี(อาการ)?{hurt}{when}?{press}.*{abdomen}',
            f'{press}.*{abdomen}.*ไม่มี(อาการ)?{hurt}',
            f'ไม่มี(อาการ)?{hurt}.*(ขยับ|เคลื่อน)',
            f'(no|not|negative|neg|soft){s}{guarding_en}',
            f'(no|not|negative|neg|soft){s}{rebound_en}',
            f'(no|not|negative|neg|soft){s}{tender}',
            f'(no|not|negative|neg|soft){s}{rebound_en}{s}{tender}'
        ]

        df['Rebound tenderness'] = df.apply(lambda x: self.extract_2_condition(
            x, no_tender_conditions, tender_conditions), axis=1)

        # 5. Cough tenderness
        cough = '(\Wไอ\W|ไอ)'
        tender = '(tender|tenderne(ss|s))'
        s = '[\s|\S]'

        cough_condition = [
            f'{non_no_or_not}มี{cough}',
            f'{non_no_or_not}มี(อาการ)?{cough}',
            f'{non_no_or_not}{cough}',
            f'cough',
            f'cough{s}{tender}'
        ]

        no_cough_condition = [
            f'ไม่มี?(อาการ)?{cough}',
            f'ไม่มี(อาการ)?{cough}',
            f'(no|not|negative|neg){s}{cough}',
            f'(no|not|negative|neg){s}{cough}{s}{tender}'
        ]
        df['Cough tenderness'] = df.apply(lambda x: self.extract_2_condition(
            x, no_cough_condition, cough_condition), axis=1)

        # 6. Nausea / Vomitting
        nausea = '(คลื่นไส้|\Wคลื่นไส้\W|คลืนไส|คลื่นไส|คลื่นไส|คลื่นไส่|คลื่[ก-ฮ]?ไส้)'
        nausea_en = '(nausea|n[ua|au]s[ae|ea]|nause(a)?|nauseated|nauseous)'
        nausea_synonym = '(nausea|queasiness|upset stomach|upset stomach|stomach unease|stomach sickness)'

        vomit = '(อาเจียน|อ(า)?เจีย[ก-ฮ]?|\Wอาเจียน\W|อ้วก|อ้ว[ก-ฮ]?)'
        vomit_en = '(vomit|vomi(t)?|vomited|vomitus|vomiting)'
        vomit_synonym = '(vomit|vomitting|throw up|regurgitate|disgorge|disgorge|barf|upchuck|puke)'

        nausea_vomit_conditions = [
            f'มี(อาการ)?({nausea}|{vomit})',
            f'{non_no_or_not}({nausea}|{vomit})',
            f'(?<!ไม่)มี({nausea}|{vomit})',
            f'\W({nausea}|{vomit})\W',
            f'{nausea_en}',
            f'{nausea_synonym}',
            f'{vomit_en}',
            f'{vomit_synonym}',
        ]

        no_nausea_vomit_conditions = [
            f'ไม่มี(อาการ)?({nausea}|{vomit})',
            f'{negative}{s}{nausea_en}',
            f'{negative}{s}{nausea_synonym}',
            f'{negative}{s}{s}{vomit_en}',
            f'{negative}{s}{s}{vomit_synonym}',
        ]

        df['Nausea/vomiting'] = df.apply(lambda x: self.extract_2_condition(
            x, no_nausea_vomit_conditions, nausea_vomit_conditions), axis=1)

        # 7. Anorexia
        # Verb
        eat = '(กิน|กิน?|ทาน|ทาน?)'
        bored = '(เบื่อ|เบือ|เบื่(อ)?|เบื่อหน่าย)'
        wont = '(ไม่อยาก|ไมอยาก|ไม่อยา(ก)?)'
        # Noun
        food = '(อาหาร|อ(า)?หา(ร)?|ขาว|ข่าว|ข้าว?|ข้าว)'
        # Adjective
        negative_th = '(ไม่ได้|ไม่ค่อยได้|ได้น้อย|น้อย|น้อ(ย)?|น้อ[ก-ฮ]?|ไม่มาก)'
        non_negative = '(?<![น้อย?|ไม่มาก])'

        # synonym
        anorexia_synonym = '(anorexia|eating disorder|loss of appetite|appetite loss|anorectic)'

        anorexia = [
            f'({bored}|{wont}){food}',
            f'({bored}|{wont}).*{food}',
            f'(?<!ไม่)มี(อาการ)?({bored}|{wont}){food}',
            f'(?<!ไม่)มี(อาการ)?({bored}|{wont}).*{food}',
            f'{eat}(?<!ยา)(อาหาร|ข้าว){negative_th}',
            f'{eat}(อาหาร|ข้าว){negative_th}',
            f'(?<![no?|not]){s}{anorexia_synonym}',
            f'มี(อาการ)?{s}{anorexia_synonym}'
        ]
        no_anorexia = [
            f'ไม่มี(อาการ)?{bored}{food}',
            f'(อาการ)?{bored}{food}ไม่มี',
            f'{eat}(?<!ไม่)ได้',
            f'{eat}(?<!ยา).*{food}(?<!ไม่)ได้',
            f'{eat}{food}(?<!ไม่)ได้',
            f'{eat}ปกติ',
            f'{eat}{food}ปกติ',
            f'{eat}{food}?(?<!ไม่)ได้(?<![น้อย?|ไม่มาก])',
            f'{negative}{s}{anorexia_synonym}',
        ]

        df['Anorexia'] = df.apply(lambda x: self.extract_2_condition(
            x, no_anorexia, anorexia), axis=1)

        # 8. Dysuria
        # Noun
        urine = '(ปัสสาวะ|ปัสสาวะ?|ฉี|ฉี่|เยียว?|เยี่ยว?)'

        # Verb
        hurt = '(ปวด|ปว[ก-ฮ]?|เจ็บ|เจ็[ก-ฮ]?)'

        # Adjective
        burning = '(ร้อน|ร้อน?|ปวดร้อน|ปวดร้อน?|แสบร้อน|แสบร้อน?)'
        stinging = '(แสบ|แสบ?|ปวดแสบ)'
        itching = '(คัน|แสบคัน|คันแสบ)'
        stuck = '(ขัด|ติดขัด)'

        # Conjunction
        when = '(ตอน|เมื่อ|เวลา|ขณะ)'

        dysuria = '(dysuria|dysur[ia|ai]|dysuria?)'
        # synonym
        dysuria_synonym = '(dysuria|urination painful|micturition|burning on urination|discomfort on urination|urinary pain|bladder pain|urethral discomfort)'

        dysuria_condition = [
            f'{hurt}.*{when}.*{urine}',
            f'(?<!ไม่)มี(อาการ)?{urine}({burning}|{stinging}|{itching}|{stuck}|{hurt})',
            f'มี(อาการ)?{urine}({burning}|{stinging}|{itching}|{stuck}|{hurt})',
            f'{urine}(?<!ไม่)มี(อาการ)?({burning}|{stinging}|{itching}|{stuck}|{hurt})',
            f'{urine}มี(อาการ)?({burning}|{stinging}|{itching}|{stuck}|{hurt})',
            f'{non_no_or_not}{urine}({burning}|{stinging}|{itching}|{stuck}|{hurt})',
            f'(?<![no?|not]){s}{dysuria}',
            f'{dysuria_synonym}'
        ]
        no_dysuria_condition = [
            f'{urine}(ปกติ|ออก|ดี)',
            f'{urine}ไม่({burning}|{stinging}|{itching}|{stuck}|{hurt})',
            f'(ไม่|ไม่มี)(อาการ)?{urine}({burning}|{stinging}|{itching}|{stuck}|{hurt})'
        ]
        df['Dysuria'] = df.apply(
            lambda x: self.extract_2_condition(x, no_dysuria_condition, dysuria_condition), axis=1)

        # 9. Stool
        # Noun
        poop = '(อุจจาระ|อุจจาระ?)'

        # Verb
        defecate = '(ถาย|ถ่าย?|ขับถาย|ขับถ่าย|ขับถ่าย?)'

        # Adjective
        obstipation = f'({abdomen}ผูก|{abdomen}ผูก?)'
        diarrhea = f'({abdomen}เสีย|{abdomen}เสีย?)'

        stool_normal = [
            f'(อาการ)?{defecate}ปกติ',
            f'(อาการ)?{defecate}.*ปกติ',
            f'(ไม่มี?|ไม่?)(อาการ)?{defecate}เหลว',
            f'(ไม่มี?|ไม่?)(อาการ)?{abdomen}({obstipation}|{diarrhea})',
            f'stool.*normal',
            f'normal.*stool',
            f'defecate.*normal',
            f'normal.*defecate',
        ]
        stool_obstipation = [
            f'(?<![ไม่มี?|ไม่])(อาการ)?{obstipation}',
            f'{poop}แข็ง',
            f'{defecate}ลำบาก',
            f'(obstipation|constipation)',
        ]
        stool_diarrhea = [
            f'(?<![ไม่มี?|ไม่])(อาการ)?{diarrhea}',
            f'{poop}เหลว',
            f'(?<![ไม่มี?|ไม่])(อาการ)?{defecate}เหลว',
            f'(diarrhea|diarrhoea)'
        ]
        df['Stool'] = df.apply(lambda x: self.extract_3_condition(
            x, stool_normal, stool_obstipation, stool_diarrhea), axis=1)
        return df

    def calculate_alvarado_score(self, row):
        score = 0
        if row['Migration of pain'] == 2:
            score += 1
        if row['Anorexia'] == 2 or row['Urine Ketone'] != 'Negative':
            score += 1
        if row['Nausea/vomiting'] == 2:
            score += 1
        if row['Tenderness in right lower quadrant'] == 2:
            score += 2
        if row['Rebound tenderness'] == 2:
            score += 1
        if row['body_temperature'] >= 37.3:
            score += 1
        if row['WBC'] > 10:
            score += 2
        if row['Neutrophil'] > 75:
            score += 1
        return score

    def calculate_pediatric_score(self, row):
        score = 0
        if row['Migration of pain'] == 2:
            score += 1
        if row['Anorexia'] == 1 or row['Urine Ketone'] != 'Negative':
            score += 1
        if row['Nausea/vomiting'] == 2:
            score += 1
        if row['Tenderness in right lower quadrant'] == 2:
            score += 2
        if row['Rebound tenderness'] == 2 or row['Cough tenderness'] == 2:
            score += 2
        if row['body_temperature'] >= 38:
            score += 1
        if row['WBC'] > 10:
            score += 1
        if row['Neutrophil'] > 75:
            score += 1
        return score

    def re_calculate_score(self, df: pd.DataFrame):
        df['Alvarado Score (AS)'] = df.apply(
            lambda x: self.calculate_alvarado_score(x), axis=1)
        df['Pediatric appendicitis score (PAS)'] = df.apply(
            lambda x: self.calculate_pediatric_score(x), axis=1)

        return df

    def encoding(self, df: pd.DataFrame):
        # Binarization (0-1)
        df['sex'] = df['sex'].map(
            {'male': 0, 'female': 1})
        df['alcohol'] = df['alcohol'].map(
            {'no': 0, 'yes': 1})
        df['exercise'] = df['exercise'].map(
            {'no': 0, 'yes': 1})
        df['smoking'] = df['smoking'].map(
            {'no': 0, 'yes': 1})
        # Ordinal Scale
        df['Urine WBC'] = df['Urine WBC'].map(self.categ_dict)
        df['Urine RBC'] = df['Urine RBC'].map(self.categ_dict)
        df['Leukocytes'] = df['Leukocytes'].map(self.categ_dict)
        df['Urine Ketone'] = df['Urine Ketone'].map(self.categ_dict)
        df['Urine Color'] = df['Urine Color'].map(self.urine_color_dict)
        df['Urine Sugar'] = df['Urine Sugar'].map(self.categ_dict)
        df['Urine Leukocytes'] = df['Urine Leukocytes'].map(self.categ_dict)
        return df

    def preprocessing(self, df: pd.DataFrame):
        df = self.data_cleaning(df)
        df = self.feature_engineering(df)
        df = self.encoding(df)
        df = self.imputation(df)
        df = self.re_calculate_score(df)
        return df
