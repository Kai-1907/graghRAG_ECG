import pandas as pd
import ast

# 1. 讀取資料
path = 'data/ptbxl/'
db_df = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
scp_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)

# 2. 將字串格式的字典轉為真正的 Python 字典
db_df.scp_codes = db_df.scp_codes.apply(ast.literal_eval)

# 3. 建立診斷碼到超類別 (Superclass) 的映射表
# 只篩選 diagnostic = 1 的臨床診斷碼
diag_scp_df = scp_df[scp_df.diagnostic == 1]

def get_superclass(scp_dict):
    classes = set()
    for code in scp_dict.keys():
        if code in diag_scp_df.index:
            classes.add(diag_scp_df.loc[code].diagnostic_class)
    return list(classes)

# 4. 進行分類標註
db_df['superclass'] = db_df.scp_codes.apply(get_superclass)

# 5. 根據結案書要求，過濾出五大類別的樣本各 2 筆作為初步測試
target_categories = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
test_set = []

for cat in target_categories:
    samples = db_df[db_df.superclass.apply(lambda x: cat in x)].head(2)
    for idx, row in samples.iterrows():
        test_set.append({
            "ecg_id": idx,
            "category": cat,
            "scp_codes": row['scp_codes'],
            "report": row['report']
        })

# 6. 儲存成你的 RAGAS 評估基礎資料
pd.DataFrame(test_set).to_json('samples_for_ragas.json', orient='records', force_ascii=False)
print("成功！已根據結案書標準篩選出五大類別樣本，存於 samples_for_ragas.json")