# סקירת המחברת המאוחדת של הפרויקט

מסמך טכני המסביר את מבנה המחברת `eda_model_dashboard.ipynb` — המחברת המאוחדת של הפרויקט. המסמך מציג את שלושת חלקי המחברת ואת האחריות של כל חבר צוות עליהם, מבלי להיכנס לפירוט תא-אחר-תא של החלק של Or. פירוט תא-אחר-תא של חלק Or (הדאשבורד) מופיע במסמך נפרד בשם `CELL_BY_CELL.md`.

---

## מבנה כללי של המחברת

המחברת המאוחדת מכילה 136 תאים ומתחלקת לשלושה חלקים ברצף:

1. **Part 1 — EDA ו-Data Preparation (שי)** — טעינת הדאטה של הלמ"ס, טיפול בקודי non-response, בניית ה-target, feature engineering, בדיקת multicollinearity, ויצירת ה-dataset הסופי למודל.
2. **Part 2 — Model: GMM (Quick version) (עובד)** — קריאת ה-dataset של שי, בניית pipeline המשלב GMM עם מסווגים supervised (Logistic Regression, Random Forest, Gradient Boosting), GridSearchCV, הערכה על test set, וניתוח feature importance עם SHAP.
3. **Part 3 — Interactive Dashboard (Or)** — טעינת ה-pipeline של עובד, בניית קובץ HTML עצמאי המציג את תוצרי שלושת החלקים בשמונה טאבים אינטראקטיביים, עם שאלון פיננסי במסך הכניסה.

בראש המחברת יש שני תאים משותפים לפני Part 1: תא ה-imports הראשי (מייבא את כל החבילות של שלושת החלקים ומזהה אוטומטית האם ההרצה מקומית או ב-Colab), ותא בדיקה שמוודא שקבצי הקלט (`data_24.xlsx` ו-`H20241501codebook.xlsx`) קיימים ב-cwd; ב-Colab התא מפעיל דיאלוג העלאה במקום להיכשל ב-`FileNotFoundError`.

---

## Part 1 — EDA ו-Data Preparation (שי)

**תפקיד:** לקחת את קובץ הסקר החברתי 2024 של הלמ"ס במצבו הגולמי, לזהות בעיות איכות, לבצע הכנה עדינה, לבנות את ה-target ואת מטריצת הפיצ'רים למודל, ולהעביר את התוצר לעובד.

### 1. עקרונות עבודה (Working Principles)

שי מקדים את החלק שלו בהצהרת חוזה: הקובץ הגולמי אינו נגוע ידנית; קודי non-response של הלמ"ס (`888888` = "לא יודע/סירב", `999999` = "לא רלוונטי/דילוג") מומרים ל-NaN; חסרים מתועדים ואינם נמחקים בשקט; לא מתבצע scaling בשלב הזה (זה עניין של ה-pipeline של המודל); כל משתנה נגזר ניתן לעקיבה חזרה ל-codebook.

### 2. טעינת הדאטה + בדיקות מבנה ראשוניות

טעינה של `data_24.xlsx` (6,907 שורות, 324 עמודות). בניית `structure_summary` — טבלה עם dtype, ספירת ערכים ייחודיים ואחוז חסרים לכל עמודה. בדיקת duplicates לפי `SerialNumber`.

### 3. טיפול בקודי non-response

החלפה של `888888` ו-`999999` ל-NaN בכל הטבלה, ובניית `missing_pct` — סדרה של אחוז חסרים לאחר ההחלפה. אחוזי חסרים גבוהים במשתנים מסוימים הם צפויים, כי הסקר משתמש בקודים אלה כפילטרים אמיתיים בין קבוצות משיבים (למשל, שאלות על שכר עבודה אינן רלוונטיות למי שאינו עובד).

### 4. טעינת ה-codebook והצלבתו עם הדאטה

טעינה של `H20241501codebook.xlsx` (הקודבוק הרשמי של הלמ"ס) לאחר preprocessing שהוסף במסגרת המיזוג (השלב הזה מוסבר בהמשך, בסעיף הטיפול במיזוג). לאחר הטעינה נבנה `data_dictionary` — join של שם עמודה, אחוז חסרים, שם המשתנה בסקר, תיאור השאלה מהקודבוק, והחלטת ניקוי ראשונית (`Drop candidate` / `Keep`). המשתנים המסומנים ל-drop הם ברובם משתנים עם רוב מוחלט של חסרים לאחר החלפת הקודים.

### 5. EDA לפי קבוצות

הגדרה של `eda_groups` — dict של שלוש קבוצות משתנים (Demographic profile, Financial profile, Social profile) לצורך EDA ממוקד. הפונקציה `plot_frequency_group(data, variables, title, codebook)` בונה עבור כל משתנה בקבוצה תרשים עמודות של התפלגות התשובות, עם תוויות באנגלית שנשלפות מהקודבוק. הפעולה חוזרת בלולאה על כל אחת משלוש הקבוצות.

### 6. Target: Financial Vulnerability

בניית `financial_risk_score` — סכום של אינדיקטורים למצוקה כלכלית מתוך שמונה שאלות של הסקר (כיסוי הוצאות, ויתור על טיפול רפואי, ויתור על תרופות, ויתור על טיפול שיניים ועוד). הדרישה היא לפחות חמישה אינדיקטורים תקינים (לא-NaN) מתוך שמונה כדי להימנע משורות עם רוב חסרים; מי שלא עמד בסף מוסר. הציון בסקאלה 0-7 (סכום הדגלים), וההתפלגות מוטה חיובית — רוב המשיבים בערך 0 או 1.

### 7. Predictor Variables — Feature Engineering

בניית מטריצת פיצ'רים ב-`model_df`, בחלוקה תמטית לשמונה תת-קבוצות:

- **7.2.1 Demographic:** גיל, מגדר, מצב משפחתי, מספר ילדים, שנת עלייה, דת, שירות צבאי/לאומי, רישיון נהיגה, מוצא לפי אב ולפי אם.
- **7.2.2 Household:** סוג משק בית, מספר מפרנסים.
- **7.2.3 Health:** משתנה בינארי `health_problem` הנגזר מ-`BeayaBriut`.
- **7.2.4 Education:** שנות לימוד + מדד `hebrew_proficiency` שנבנה מ-Cronbach's alpha על שלושה פריטי היגד בעברית (דיבור, קריאה, כתיבה) שנמצאו מתואמים מאוד.
- **7.2.5 Employment:** מצב תעסוקה, מספר מקומות עבודה, מנהל/עובד, מעמד עבודה, שביעות רצון מעבודה, רמת הכנסת עבודה, היקף משרה. חסרים מבניים (למשל, שאלות על שכר עבור מי שאינו עובד) מטופלים במיפוי מפורש.
- **7.2.6 Social participation:** התנדבות, תרומה כספית, סכום תרומה.
- **7.2.7 Financial management + behavior:** מי אחראי על הכסף במשק הבית, שימוש באפליקציית מעקב, הוראות קבע, השקעות בנק, השקעות אחרות, מדד ידע השקעות. גם כאן מבוצע חישוב Cronbach's alpha לפני צירוף פריטי התנהגות כמדד אחד.
- **7.2.8 Financial literacy:** `FinancialLiteracyScore` — סכום התשובות הנכונות בשאלות אוריינות פיננסית (אינפלציה, ריבית, תשואה ריאלית).

התוצאה: `model_data` — DataFrame נקי עם ה-target ועם כ-40 פיצ'רים.

### 8. Multicollinearity Check

חישוב מטריצת Phi-K (מקדם קורלציה שעובד גם על משתנים קטגוריאליים) על כל הפיצ'רים ב-`model_data`. משתנים מסוימים עם קורלציה גבוהה מדי לפיצ'ר אחר מוסרים בסוף הצעד. `model_data` הסופי מכיל את הפיצ'רים שנשארו אחרי הצעד הזה + ה-target.

### 9. Final Dataset Summary

תדפיס של הצורה הסופית של `model_data` (מספר שורות, מספר עמודות, שם ה-target).

### Handoff לסוף Part 1

התאים האחרונים של Part 1 שומרים שני קבצים לצורך המשך הזרימה:

- `model data.csv` — ה-`model_data` הנקי עם ה-target, לצורך טעינה על ידי Part 2.
- `shai_state.pkl` — dict הכולל את `raw`, `structure_summary`, `data_dictionary`, `eda_groups`, `codebook`, `model_data`, לצורך תמיכה בזרם Part-3-only ב-Colab (מתואר בסעיף בסוף המסמך).

### שינויים שנעשו בחלק של שי במסגרת המיזוג

הקוד של שי לא נגע בעצמו. כל השינויים מרוכזים בתאים חדשים שנוספו סביבו:

- **תא preprocessing של הקודבוק:** הקובץ הגולמי מהלמ"ס מגיע עם 12 שורות מטא ושמות עמודות בעברית (`שם שדה/ נוסח שאלה`, `שם משתנה`). הקוד של שי מצפה לשמות אנגליים (`field_name`, `variavle_name`, `code`, `code_title`). התא החדש מזהה אם הקובץ עדיין בפורמט הגולמי, מדלג על 12 שורות מטא, ומחליף את שמות העמודות לאנגלית. הפעולה idempotent — הרצה נוספת אינה משנה קובץ כבר מעובד.
- **תרגום ערכי `code_title` לאנגלית:** תא ה-preprocessing כולל גם מילון תרגום `_HE2EN` הממיר את הערכים הכי שכיחים בגרפי EDA לאנגלית (`Male/Female`, `Married/Single`, `Very satisfied/Satisfied`, `Up to 2,000 NIS`, `Yes/No`, `Often/Rarely/Never`). התוויות בגרפים של `plot_frequency_group` יופיעו באנגלית בזכות זה.
- **גשר `missing_report = missing_after`:** בקוד של שי יש שימוש במשתנה `missing_report` שלא הוגדר בשום מקום — הרצה של הקוד נופלת עם `NameError`. תא גשר קצר מגדיר את המשתנה בעזרת `missing_after` שכן קיים.

פירוט מדויק של כל תוספת מופיע בקובץ `changes_after_merge.xlsx`.

---

## Part 2 — Model: GMM (Quick version) (עובד)

**תפקיד:** לקחת את `model data.csv` של שי, לבנות pipeline המשלב Gaussian Mixture Model להפקת פיצ'רים חדשים על גבי מסווגים supervised, למצוא את הקומבינציה הטובה ביותר עם GridSearchCV, להעריך את המודל על test set, ולנתח feature importance עם SHAP. התוצאה מועברת ל-Part 3.

### 0. Datas — טעינה ובניית ה-target הבינארי

טעינה של `model data.csv`. הבעיה המקורית של שי היא עם target בסקאלה 0-7 (regression). עובד מטרין אותה לבעיית classification בינארית: `AtRisk = 1` אם `financial_risk_score >= 3`, אחרת `0`. הסף 3 מוגדר על סמך התפלגות ה-target ומאזן בין רגישות (recall) לדיוק.

לאחר מכן: הגדרת רשימת `features` (37 פיצ'רים), חלוקה ל-`X` ו-`y`, ופיצול ל-train ו-test עם stratification.

### 1. Baseline Model Evaluation

הערכה של שלושה מסווגים "dumb" ללא GMM, כדי לקבוע רף השוואה: Logistic Regression, Random Forest, Gradient Boosting. הטבלה שנוצרת (`cv_results`) מציגה Accuracy, Precision, Recall, F1 ו-ROC AUC לכל מודל, ומשמשת אחר כך בטאב Model Performance בדאשבורד.

### 2. Supervised Classification Using GMM-Derived Features

הלב של Part 2. עובד בונה שלוש מחלקות מותאמות:

- **`GMMFeatureGenerator(BaseEstimator, TransformerMixin)`** — טרנספורמר שמאמן GMM על ה-X ומייצר עמודות פיצ'רים חדשות מ-`predict_proba` (הסתברויות ה-soft cluster לכל דגימה). העמודות נוספות ל-X המקורי, כך שהמודל הבא בצינור מקבל גם את הפיצ'רים המקוריים וגם את הפיצ'רים שמייצג ה-GMM.
- **`ThresholdClassifier(BaseEstimator, ClassifierMixin)`** — עטיפה סביב מסווג פרובבליסטי המאפשרת להשתמש בסף החלטה שאינו 0.5. בשימוש בסף 0.3 כדי להטות לכיוון recall גבוה יותר על חשבון precision — הגיוני עבור בעיה של סיכון פיננסי, שם false negative יקר יותר.
- **`Pipeline`** — צינור sklearn של ארבעה שלבים: `SimpleImputer(strategy="median")` -> `StandardScaler` -> `gmm` (או `passthrough`) -> `model` (ה-ThresholdClassifier עוטף את המסווג).

### GridSearchCV

`param_grid` בנוי כרשימה של שישה תת-מרחבים, כל אחד מגדיר קומבינציה של: GMM דלוק (עם `n_components` = 5 או 7 ו-`covariance_type = "full"`) או `passthrough`, מסווג (LogReg, RF או GB), והיפר-פרמטרים של המסווג עצמו. GridSearchCV חוזר על כל הקומבינציות ב-5-fold CV ובוחר את זו עם ה-F1 הגבוה ביותר. הבחירה בסוף היא ה-`best_estimator_`.

### 3. Summary of Final Model Evaluation With and Without GMM

עובד מציג טבלה משווה בין המודל הטוב ביותר עם GMM לבין הטוב ביותר בלי GMM. השוואה זו עונה על השאלה האם ה-Gaussian Mixture אכן מוסיף אינפורמציה שהמסווגים לא מוציאים לבד.

### 4. Final Test Evaluation

הרצה של המודל הנבחר (`best_model`) על ה-test set ומדידת המדדים הסופיים.

### Handoff לסוף Part 2

- `xgb_model.pkl` — ה-`best_model` נשמר בעזרת `joblib.dump`. Part 3 קורא אותו וטוען לתוך המשתנה `model`.
- `ovad_state.pkl` — dict הכולל את `features` (רשימת הפיצ'רים), `cv_results` (טבלת ההשוואה), ו-`saved_figures` (מאגר גרפי matplotlib שנלכדים אוטומטית — ראה סעיף 5-7 להלן). לצורך תמיכה בזרם Part-3-only.

### 5. Hyperparameter Sensitivity Analysis

לאחר בחירת ה-best model, עובד רץ סבב sensitivity על ההיפר-פרמטרים המרכזיים: `n_estimators` (75-120), `learning_rate` (0.01-0.9), `max_depth` (4-9), ו-`threshold` (0.2-0.6). לכל פרמטר מוצג גרף של Accuracy, Recall ו-F1 כפונקציה של הערך, עם קו אנכי המסמן את הערך של ה-best model. הגרפים נלכדים אוטומטית ב-`saved_figures` ומוצגים בטאב Model Performance של הדאשבורד.

### 6. Model Performance Evaluation on Test Set

הצגה גרפית של הביצועים של ה-best model על ה-test set: confusion matrix, ROC curve, precision-recall curve. גם אלה נלכדים ל-`saved_figures`.

### 7. SHAP Model Interpretation

חישוב SHAP values על ה-best model והצגתם דרך beeswarm plot, feature importance, ו-dependence plots. הגרפים נלכדים ל-`saved_figures` ומוצגים בטאב Model Interpretation של הדאשבורד.

### מנגנון לכידת הגרפים

עובד מתקין monkey-patch על `plt.show` (בהתחלה של Part 2) — כל קריאה ל-`plt.show()` מוסיפה את ה-figure ל-`saved_figures = {}` ואז קוראת ל-show המקורי. כך כל גרף שנוצר בהמשך Part 2 נכנס אוטומטית למאגר, בלי שצריך לגעת בקוד של כל תא בנפרד. Part 3 של Or קורא את `saved_figures` בהמשך כדי להטמיע את הגרפים בדאשבורד.

### שינויים שנעשו בחלק של עובד במסגרת המיזוג

- **`drive.mount(...)`** — התא הראשון של עובד מריץ `from google.colab import drive; drive.mount('/content/drive')`. בסביבה מקומית זה קורס עם `ImportError`. במסגרת המיזוג התא נעטף ב-`try/except (ImportError, ModuleNotFoundError): pass`. ב-Colab ההתנהגות זהה למקור.
- **`path` בטעינת `model data.csv`** — הקוד המקורי טוען מנתיב Drive של עובד: `path = "/content/drive/MyDrive/PROJECT/data/"`. הערך שונה ל-`""` (מחרוזת ריקה), כך שה-`pd.read_csv("model data.csv")` קוראת מ-cwd — שם קיים הקובץ שנשמר בסוף Part 1 של שי.
- **תאי handoff לפני ואחרי המודל** — נוספו תאי שמירה ל-`xgb_model.pkl` וגם ל-`ovad_state.pkl`.

פירוט מדויק של כל שינוי מופיע בקובץ `changes_after_merge.xlsx`.

---

## Part 3 — Interactive Dashboard (Or)

**תפקיד:** ליצור קובץ HTML עצמאי אחד (`index.html`) המציג את תוצרי שלושת החלקים בשמונה טאבים אינטראקטיביים. הקובץ עצמאי לחלוטין — Plotly.js מוטמע (כ-3.5 MB), כל הגרפים embedded, כל הטבלאות embedded, המפה כ-iframe srcdoc, וענן המילים כ-base64. אין תלות בשרת ייצור; GitHub Pages פשוט מגיש קובץ סטטי.

הזרימה של Part 3:

1. טעינת המודל של עובד (`xgb_model.pkl`) ובדיקה האם נדרש לשחזר את מצב הזיכרון של Parts 1 ו-2 מקבצי pkl (במצב Part-3-only).
2. טעינה מחדש של הדאטה הגולמי + הגדרת מיפויי קידוד לתוויות + קואורדינטות של תת-מחוזות למפה.
3. הגדרת השאלון (10 שאלות: 5 ידע + 5 התנהגות).
4. בניית שני גרפי Plotly לטאב Young Generation, ענן מילים פיננסי, ומפת Folium של תת-מחוזות.
5. הרכבת מחרוזות HTML לחמישה טאבים ראשיים (Overview, Data Quality, Descriptives, Model Performance, Model Interpretation).
6. הרכבת קובץ `index.html` יחיד עם CSS, JavaScript, ותוכן הטאבים.
7. הרמת שרת מקומי + פתיחה בדפדפן (מקומית), או הורדה של `index.html` (ב-Colab).

**פירוט תא-אחר-תא של Part 3, כולל הסבר על JavaScript של השאלון, מנגנון החלפת הטאבים, כפתור Back to quiz, סימולטור ריבית דריבית, ופונקציות הבנייה של הגרפים — מופיע במסמך נפרד בשם [`CELL_BY_CELL.md`](./CELL_BY_CELL.md).**

---

## הזרימה הבין-חלקית (Cross-part Flow)

הזרימה של תוצרי החלקים בין Parts 1 -> 2 -> 3 מתבצעת בשני מסלולים במקביל:

### 1. בזיכרון (במצב Restart & Run All)

כשמריצים את כל המחברת ברצף (Restart & Run All), משתני זיכרון גלובליים עוברים בין החלקים בלי צורך בטעינה מהדיסק:

- **מ-Part 1 ל-Parts 2 ו-3:** `raw`, `structure_summary`, `data_dictionary`, `eda_groups`, `codebook`, `model_data`.
- **מ-Part 2 ל-Part 3:** `features`, `cv_results`, `saved_figures`, `best_model`.

### 2. בקבצי handoff על הדיסק (למצב Part-3-only)

כשלא ניתן להריץ את הכל ברצף (למשל, סשן Colab שקרס ומתחילים מ-Part 3), Part 3 יודע לטעון את המצב שנשמר בסוף Parts 1 ו-2:

- **`model data.csv`** — הדאטה הנקי של שי (המקושר ל-Part 2).
- **`xgb_model.pkl`** — ה-pipeline המאומן של עובד.
- **`shai_state.pkl`** — dict של כל תוצרי הזיכרון של שי, לצורך שחזור המצב ב-Part 3.
- **`ovad_state.pkl`** — dict של כל תוצרי הזיכרון של עובד, לצורך שחזור המצב ב-Part 3.

התא הראשון של Part 3 בודק אם המשתנים ב-globals קיימים. אם כן — לא נעשה דבר. אם לא, קוראים את קבצי ה-pkl ו-`globals().update(...)` מזריק אותם למחברת כאילו Parts 1 ו-2 באמת רצו. אם קבצי ה-pkl חסרים — נזרקת שגיאת `FileNotFoundError` ברורה עם הוראה להריץ את שאר החלקים או להביא את הקבצים.

---

## סביבת ההרצה

המחברת נבנתה כך שתרוץ בשתי סביבות בלי שינוי:

- **מקומית ב-Jupyter (Windows / Linux / macOS)** — כל הדרוש: קובץ `data_24.xlsx` ו-`H20241501codebook.xlsx` באותה תיקייה כמו המחברת. הרצה ב-Restart & Run All מייצרת את `index.html` ופותחת אותו בדפדפן דרך שרת מקומי על 127.0.0.1.
- **ב-Google Colab** — התא הראשון מזהה את הסביבה ומפעיל דיאלוג להעלאת קבצי הקלט אם הם חסרים. `google.colab.drive.mount(...)` שהיה בקוד המקורי של עובד עוטף ב-`try/except` כדי שלא יקרוס כאשר אינו רלוונטי. בסוף ההרצה, `index.html` יורד אוטומטית לדפדפן של המשתמש דרך `google.colab.files.download`.
