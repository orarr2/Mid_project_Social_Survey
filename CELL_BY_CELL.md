# מדריך תא-אחר-תא של המחברת המאוחדת - חלק Or (הדאשבורד)

מסמך הסבר לפגישת הזום. עובר על **החלק שלי בלבד** במחברת המאוחדת `eda_model_dashboard.ipynb` - כלומר Part 3 - Interactive Dashboard, ומסביר מה כל תא עושה, למה, ומה חשוב לומר עליו כדי שלא ייחשב לקופסה שחורה.

המחברת המאוחדת מורכבת מ-3 חלקים: Part 1 (שי - EDA וניקוי), Part 2 (עובד - מודל GMM+XGBoost), **Part 3 (אני - דאשבורד)**. המסמך הזה מתחיל בתא הראשון של Part 3 - כל מה שרץ לפניו הוא הכנת דאטה על ידי שי ואימון מודל על ידי עובד, והתוצרים שלהם עוברים אליי בשני אופנים במקביל: (1) משתני זיכרון גלובליים (`model_data`, `features`, `saved_figures`, `codebook`, `structure_summary`, `eda_groups`, `cv_results`) שקיימים אחרי Restart & Run All, וגם (2) קבצי handoff על הדיסק (`model data.csv`, `xgb_model.pkl`, `shai_state.pkl`, `ovad_state.pkl`) - כדי שאפשר יהיה להריץ רק את Part 3 בסשן חדש, למשל ב-Google Colab, בלי לחזור על שי ועובד.

Part 3 שלי בנוי מ-**11 תאי קוד** ועוד כמה תאי markdown של כותרות. הזרימה: טעינה של תוצרי החלקים הקודמים -> בניית 3 גרפי Plotly (Young Generation) + ענן מילים + מפת Folium -> הרכבת HTML לכל טאב -> כתיבה של `index.html` יחיד -> הרמת שרת מקומי שפותח אותו בדפדפן (או Colab download).

---

## תא 1 - קוד - Handoff מ-Part 2: טעינת המודל + Colab bridge

**מה זה עושה:** התא הראשון של Part 3 שלי. שני דברים במקביל:

1. **בדיקה אם צריך "לשחזר" את מצב הזיכרון של Parts 1+2:** במצב Restart & Run All כל המשתנים כבר קיימים בזיכרון ולא צריך לעשות דבר. אבל במצב Part-3-only (סשן חדש, למשל Colab שקרס באמצע) הזיכרון ריק - במקרה כזה קוראים את קבצי ה-pickle שנשמרו בסוף שי ובסוף עובד, ופורקים אותם ל-`globals()`.

2. **טעינת המודל המאומן:** `model = joblib.load("xgb_model.pkl")`.

**קטעי קוד מרכזיים:**

```python
_stale = any(_need(v) for v in
             ["model_data", "features", "saved_figures", "codebook",
              "structure_summary", "data_dictionary", "eda_groups", "raw"])

if _stale:
    _missing = [f for f in _PART3_REQUIRED if not os.path.exists(f)]
    if _missing and IN_COLAB:
        from google.colab import files
        _uploaded = files.upload()   # מבקש מהמשתמש להעלות pkl-ים
    ...
    _shai_state = joblib.load("shai_state.pkl")
    _ovad_state = joblib.load("ovad_state.pkl")
    globals().update(_shai_state)   # מזריק raw, structure_summary, codebook, model_data...
    globals().update(_ovad_state)   # מזריק features, cv_results, saved_figures

model = joblib.load("xgb_model.pkl")
```

**דגשים חשובים לפגישה:**

- **`globals().update(...)` הוא הטריק:** dict נטען מ-pickle מכיל את השמות המקוריים של המשתנים, וההזרקה ישירות ל-`globals()` הופכת אותם למשתנים גלובליים במחברת - כאילו Parts 1+2 באמת רצו. זו הטכניקה שמאפשרת "לזייף" את המצב אחרי הפעלה חדשה.
- **`IN_COLAB`** הוגדר עוד בתא ה-imports הראשון של המחברת המלאה (בראש - לפני החלקים) על ידי `try/except google.colab`. אני מסתמך עליו כאן, לא מגלה את הפלטפורמה מחדש.
- **הבחירה של joblib ולא pickle רגיל** נחוצה כי `saved_figures` הוא dict של אובייקטי `matplotlib.figure.Figure` - joblib מסתדר איתם טוב יותר.

---

## תא 2 - קוד - בדיקת סביבה + imports של Part 3

**מה זה עושה:** בדיקה חוזרת שהחבילות שהחלק שלי צריך זמינות (`plotly`, `folium`, `wordcloud`, `PIL`, `openpyxl`, `matplotlib`) ומייבא אותן. יש כאן חפיפה מסוימת עם תא ה-imports הכללי בראש המחברת - התא הזה קיים כדי שכשמריצים רק את Part 3 מתחילתו הוא יעבוד גם אם תא ה-imports הכללי לא רץ.

**קטע קוד מרכזי:**

```python
REQUIRED = ["numpy", "pandas", "plotly", "openpyxl", "folium", "wordcloud",
            "PIL", "matplotlib"]
OPTIONAL = ["pyarrow"]
```

**דגש חשוב:** `plotly` הוא המנוע של הגרפים האינטראקטיביים, `folium` בונה את המפה, `wordcloud` יוצר את ענן המילים, `PIL` נדרש ל-`wordcloud` בסביבתו, `pyarrow` אופציונלי - רק אם רוצים cache מהיר של הדאטה בפורמט parquet במקום לקרוא את ה-Excel כל פעם מחדש.

---

## תא 3 - קוד - טעינת דאטה + מיפויי קודים + קואורדינטות תת-מחוזות

**מה זה עושה:** שלושה דברים:

1. **טעינת הדאטה** מקובץ `data_24.xlsx`. ריצה ראשונה קוראת את ה-Excel (איטי, כ-דקה) וכותבת `data_24_cache.parquet`. כל ריצה הבאה טוענת מה-parquet תוך פחות משנייה.
2. **הגדרת מילוני קידוד -> תווית:** `AGE_LABELS`, `INCOME_LABELS`, `SAT_LABELS`, `LEFTOVER_LABELS`, `OPTIMISM_LABELS`.
3. **הגדרת `NAFA`** - 16 קודי תת-מחוזות של הלמ"ס עם קואורדינטות (lat, lon). זה מה שמזין את מפת Folium בטאב Geography.

**דגש חשוב:** המילונים משמשים **רק לתוויות ציר בגרפים**. הערך הגולמי בדאטה נשאר `1` או `2` או `888888` - הוא לא מוחלף. שי מטפל בהחלפת הקודים ל-NaN בחלק שלו על המחברת - אני משאיר את הדאטה כמו שהוא בקוד של אור כדי לא לחדור לתחום שלו.

---

## תא 4 - קוד - הגדרת השאלון הפיננסי (10 שאלות)

**מה זה עושה:** מגדיר את השאלון בתור מבנה נתונים בשם `QUIZ` - list of dicts. יש **5 שאלות ידע** (`kind: "knowledge"` עם שדה `correct`) ו-**5 שאלות התנהגות** (`kind: "behavior"` עם שדה `survey_var` שמצביע על משתנה בסקר, ו-`popYes` שנוצר בזמן טעינת המחברת).

**קטע קוד מרכזי:**

```python
QUIZ = [
    {"id": "Q1", "kind": "knowledge",
     "text": "What is a typical annual management fee (dmei nihul)...",
     "options": ["0.05% to 0.3%", "0.5% to 1%", ...], "correct": 1},
    ...
    {"id": "Q3", "kind": "behavior", "survey_var": "HisachonPensyoni",
     "yes_code": 1,
     "text": "Do you currently have a pension savings account?",
     "options": ["Yes", "No"], "popYes": 63.8},
    ...
]

# popYes נבנה מהדאטה עצמו - לא ערך קשיח:
for q in QUIZ:
    if q.get("kind") == "behavior":
        col = q["survey_var"]
        q["popYes"] = float(((df[col] == q["yes_code"]).sum() / len(df)) * 100)
```

**דגשים חשובים לפגישה:**

- **המשמעות של החלוקה 5+5:** שאלות הידע מייצרות את **ציון המשתמש** (Gauge חצי-עגול בטאב Your Profile), שאלות ההתנהגות מייצרות את **גרף ההשוואה** (מה אחוז ה"כן" של המשתמש מול 6,907 המשיבים).
- **`popYes` נחשב בזמן טעינת המחברת פעם אחת**. הוא נכתב לתוך HTML כחלק מ-`QUIZ` JSON. בדפדפן זו כבר סתם קריאה מטבלה - אין קריאת רשת, אין פייתון בסביבת ה-runtime.
- **חמש שאלות ההתנהגות ממופות ישירות למשתני הסקר** - `HisachonPensyoni`, `HatavaKHishtalmut_wp`, `InternetShilemMatbeaDig`, `HashkaotBank`, `InternetDohPensia`. כשמדברים על הדאשבורד בפגישה: זו נקודת ההשוואה 1:1 של המשתמש מול המדגם.
- **המבנה של `QUIZ` הוא JSON serializable במלואו**, וזה חשוב כי בהמשך אני מכניס אותו לתוך `HTML_TEMPLATE` דרך `json.dumps` (בתא 10).

---

## תא 5 - קוד - Helpers משותפים (`FIGS`, `_clean`, `register`, `age_order`)

**מה זה עושה:** מגדיר את התשתית המשותפת של גרפי Plotly. אחרי המיזוג עם שי ועובד, נשארו בי רק 2 גרפי Plotly (Young Generation) - שאר הגרפים הוסרו כי הטאבים שלהם הוחלפו בטאבים חדשים שמציגים תוצרים של שי ועובד. עדיין נחוצים ה-helpers כי הגרפים הנותרים משתמשים בהם.

**קטעי קוד מרכזיים:**

```python
FIGS = {}                     # dict מרכזי - המפה של key -> Plotly Figure

def _clean(fig, height=430):
    fig.update_layout(height=height, margin=dict(t=70, r=30, b=70, l=60),
                      font=dict(family="Segoe UI, Arial", size=13),
                      title=dict(font=dict(size=17), x=0.5, xanchor="center"),
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showgrid=False, zeroline=False, ...)
    fig.update_yaxes(showgrid=False, zeroline=False, ...)
    return fig

def register(key, fig):
    FIGS[key] = fig         # שמירה למאגר שיתעכשמע ל-HTML
    fig.show()              # הצגה inline במחברת עצמה
    return fig

age_order = [AGE_LABELS[k] for k in sorted(AGE_LABELS)]   # ["20-24","25-29",...,"75+"]
```

**דגש חשוב:** ההפרדה בין "בניית הגרף" ל-"רישום הגרף" זו בחירה מכוונת. `register` הוא ה"חוזה" - כל מי שמסיים לבנות Figure חייב לקרוא לו כדי שיופיע (1) ב-`FIGS` שמנוצל בבניית ה-HTML, (2) inline במחברת עצמה למי שרץ אותה ב-Jupyter. זה מונע מצב שבו מישהו בונה גרף ושוכח לחבר אותו לדאשבורד.

---

## תא 6 - קוד - טאב Young Generation (2 גרפים)

**מה זה עושה:** בונה 2 גרפי Plotly עבור הטאב Young Generation. במקור היו כאן 4 גרפים; הסרתי את `fig-young-radar` ואת `fig-young-coverage` לפי בקשה שהעלית באמצע הבנייה.

**הגרפים שנשארו:**

1. **fig-young-digital** - קו לפי גיל של אימוץ שירותים דיגיטליים פיננסיים (Crypto, בדיקת אשראי אונליין, בדיקת פנסיה אונליין). הצעירים דווקא מובילים ב-crypto ובבדיקת אשראי.
2. **fig-young-satisfaction** - עמודות של רמת שביעות רצון מתכנון פרישה, רק בקבוצת 20-29. מדגיש כמה מהם "בכלל לא מרוצים" - זה ה-wake-up call החינוכי של הטאב.

**דגש חשוב:** הטאב Young Generation נשאר לא רק בגרפים - יש בו גם **סימולטור ריבית דריבית אינטראקטיבי** (הקוד שלו יושב ב-HTML בתא 10). המשתמש מגלה שהפקדת 500 ש"ח בחודש מגיל 25 עד 67 ב-6% ריבית מגיעה לכ-1.1 מיליון ש"ח. זו הנקודה החינוכית של הטאב לצעירים.

---

## תא 7 - קוד - ענן מילים פיננסי (Word Cloud)

**מה זה עושה:** מייצר תמונת PNG של ענן מילים שכולה מונחים פיננסיים, כשגודל כל מילה פרופורציונלי לספירה בפועל בדאטה של הסקר. התמונה נשמרת בזיכרון כ-base64 ומוטמעת בתוך ה-HTML הסופי.

**קטע קוד מרכזי:**

```python
def yes_count(var):
    return int((df[var] == 1).sum())

FIN_WEIGHTS = {
    "Pension":         yes_count("HisachonPensyoni"),
    "Study Fund":      yes_count("HatavaKHishtalmut_wp"),
    "Bitcoin":         yes_count("InternetShilemMatbeaDig"),
    "Loan":            yes_count("HalvaaBank"),
    "Investments":     yes_count("HashkaotBank"),
    ...   # 32 מונחים סה"כ
}

wc = WordCloud(width=1200, height=520, background_color="white",
               colormap="viridis", ...).generate_from_frequencies(FIN_WEIGHTS)

buf = io.BytesIO()
wc.to_image().save(buf, format="PNG")
WORDCLOUD_B64 = base64.b64encode(buf.getvalue()).decode("ascii")
```

**דגשים חשובים לפגישה:**

- **לא ענן מילים גנרי מהאינטרנט** - כל שקלול מבוסס על ספירה אמיתית מהסקר. אם 1,093 אנשים ענו שיש להם פנסיה, המילה `Pension` מקבלת משקל 1,093.
- **base64 embedding** - התמונה נכנסת לתוך ה-HTML כ-`<img src="data:image/png;base64,...">`, כך שאין קובץ PNG חיצוני. `index.html` שלם ועצמאי.
- אחרי המיזוג, הענן חזר להיות מוצג בטאב Overview.

---

## תא 8 - קוד - מפת Folium של ישראל

**מה זה עושה:** יוצר מפה אינטראקטיבית של ישראל (Folium שבנוי על Leaflet) עם 16 עיגולים - אחד לכל תת-מחוז (Nafa). גודל כל עיגול פרופורציונלי למספר המשיבים באותו אזור (סקאלת שורש - כדי שאזורים קטנים לא ייבלעו על ידי תל אביב).

**קטע קוד מרכזי:**

```python
m = folium.Map(location=[31.85, 35.10], zoom_start=8,
               tiles="cartodbpositron", min_zoom=7, max_zoom=12, max_bounds=True)

for code_val, meta in NAFA.items():
    n = int(nafa_counts.get(code_val, 0))
    radius = 6 + 22 * (n / max_c) ** 0.5      # scaling sqrt
    folium.CircleMarker(
        location=[meta["lat"], meta["lon"]], radius=radius,
        color="#1f5fc4", fill=True, fill_color="#2c7be5", fill_opacity=0.55,
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=folium.Tooltip(f"<b>{meta['name']}</b>: {n:,} respondents ...", sticky=True)
    ).add_to(m)

MAP_SRCDOC = m.get_root().render()   # HTML מלא של המפה כמחרוזת אחת
```

**דגשים חשובים לפגישה:**

- **אין `fit_bounds`** - קורא ל-`fit_bounds` ב-Folium 0.20 גורם ל-over-zoom (הוא נכנס למקסימום זום כדי לכסות את כל הגבולות; לישראל שהיא קטנה, זה חתך של רחוב). הבחירה של `zoom_start=8` מיצירה מסגרת נכונה.
- **`MAP_SRCDOC` הוא string של HTML שלם עם JS ו-CSS משלו**. הוא מוזרק בהמשך ל-`<iframe srcdoc>` - כך Leaflet רץ בסביבה מבודדת ולא מתנגש עם ה-CSS/JS של הדאשבורד.
- אחרי המיזוג, המפה חזרה להיות מוצגת בטאב Geography החדש.

---

## תא 9 - קוד - הרכבת תוכן HTML לכל טאב

**מה זה עושה:** התא הכי גדול והכי חשוב שלי. לוקח את התוצרים של שי ועובד + הגרפים והתמונות של הטאבים החדשים, ומרכיב מחרוזות HTML לכל אחד מ-5 הטאבים הראשיים: `OVERVIEW_HTML`, `DATA_QUALITY_HTML`, `DESCRIPTIVES_HTML`, `MODEL_PERF_HTML`, `MODEL_INTERP_HTML`. כל אחת מהמחרוזות נכנסת בהמשך למקום ההתאמה שלה ב-`HTML_TEMPLATE` (תא 10).

**מבנה TAB TAB:**

### `OVERVIEW_HTML` - טאב Overview

- **כרטיסי KPI:** מספר משיבים (6,907), מספר משתנים גולמיים, מספר predictors שנכנסו למודל, שנת הסקר (2024).
- **ענן מילים** - `<img src="data:image/png;base64,{WORDCLOUD_B64}">` - הוחזר אחרי המיזוג.
- **Working Principles של שי** - 5 עקרונות שיסודו הפרויקט: כלב הקובץ הגולמי נשאר ללא נגיעה; קודי `888888` ו-`999999` מומרים ל-NaN; לא מוחקים שורות רבות בהיחבא; לא סקיילינג בשלב זה; שקיפות מלאה מול ה-codebook.
- **Structure summary (10 שורות ראשונות)** - טבלה מתא של שי עם דוגמאות עמודות + dtype + מספר ערכים ייחודיים + אחוז חסרים.
- **Raw data head(10)** - 10 שורות ראשונות של `df` אחרי ניקוי בסיסי של שי.
- **Features selected for the model (Ovad)** - "chips" של השמות של כל הפיצ'רים שעובד בחר.

### `DATA_QUALITY_HTML` - טאב Data Quality

- **Missing values report per column** - **רשת רב-עמודתית של אריחים קומפקטיים**, אריח לכל משתנה, מציג את שם המשתנה ואחוז החסרים בו. צביעה לפי חומרה: אדום ≥50%, ענבר ≥20%, ירוק אחרת. בונה 324 אריחים - הרבה יותר קריא מטבלה של 324 שורות.
- **Data dictionary** (10 שורות ראשונות) - הטבלה של שי שמרכזת: שם משתנה + התיאור מה-codebook של הלמ"ס + החלטת ניקוי.
- **Drop candidates** (10 שורות ראשונות) - עמודות ששי סימן להסרה בגלל % חסרים גבוה או חוסר רלוונטיות.
- **CBS codebook** (30 שורות ראשונות) - הצגת ה-codebook עצמו אחרי preprocessing שהוספתי (12 שורות מטא הוסרו, שמות עמודות תורגמו לאנגלית, ותוויות המשתנים המרכזיים תורגמו לאנגלית).

### `DESCRIPTIVES_HTML` - טאב Descriptives

- לכל אחת מ-3 הקבוצות ב-`eda_groups` של שי (Demographic profile, Financial profile, Social profile) - מראה **התפלגות ה-value_counts** של כל משתנה בקבוצה עם עמודות אופקיות "mini bar plot" ב-HTML.
- **התוויות מגיעות מהcodebook** דרך פונקציית helper `_lookup_label(var, code)` שאני בונה בתא הזה - היא מחזירה את `code_title` הרלוונטי, אחרי שהתא של preprocessing של ה-codebook תרגם את הערכים העבריים לאנגלית.
- בסוף - **Distribution of `financial_risk_score`** - התפלגות ה-target המחושב של שי (סקאלה 0-7 של תסמיני סיכון פיננסי).

### `MODEL_PERF_HTML` - טאב Model Performance

- **Baseline comparison table** - טבלה שעובד בונה: מודלים שנוסו, מדדים (Accuracy, Precision, Recall, F1, ROC AUC).
- **6 גרפי ביצועים ראשונים** מ-`saved_figures` (נוצרים ב-Part 2 של עובד ונלכדים אוטומטית דרך monkey-patch של `plt.show`) - מוטמעים כתמונות base64.

### `MODEL_INTERP_HTML` - טאב Model Interpretation

- **6 גרפי SHAP הבאים** מ-`saved_figures` - Beeswarm, feature importance, dependence plots.

**קטעי קוד מרכזיים:**

```python
def _fig_to_img(fig, alt="figure", max_width_px=1000):
    """ממיר matplotlib Figure ל-<img> base64 - מפה שי/עובד ל-HTML"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" ...>'

def _df_to_html(df, max_rows=None, classes="datatable"):
    """טבלה עם cap אופציונלי + הודעה 'Showing first X of Y rows'"""
    if max_rows is not None and len(df) > max_rows:
        note = f'<div class="table-note">Showing first {max_rows} of {len(df):,} rows</div>'
        return note + df.head(max_rows).to_html(...)
    return df.to_html(...)

def _lookup_label(_var, _code):
    """שולף label מה-codebook (אחרי תרגום לאנגלית) עם fallback לקוד הגולמי"""
    _match = codebook.loc[(codebook["variavle_name"] == _var) &
                          (codebook["code"] == _code), "code_title"]
    return str(_match.iloc[0]).strip() if len(_match) else str(_code)
```

**דגשים חשובים לפגישה:**

- **הפרדה מלאה בין TAB-בנייה ל-TAB-הצגה**. התא הזה בונה מחרוזות HTML בלבד. הדבקתן במסמך HTML מלא קורית בתא הבא. זו הפרדה שהופכת את הדאשבורד לקל לתחזוקה.
- **קפיצות זיכרון** - כל מטריצת Phi-K של שי (מטריצת קורלציות רב-סוגית שדורשת כמה GB) לא מוצגת כאן - הוצאתי אותה כדי לא לפוצץ את ה-kernel. הערה מודפסת בטאב Descriptives.
- **`_lookup_label` הוא הלב של תיקון "התוויות בעברית"**. הבעיה: הקוד של שי לא נגע - הוא מייצר גרפים עם `.value_counts()` וזה מציג קודים מספריים 1, 2, 3. הפתרון: תא preprocessing של ה-codebook תרגם את `code_title` לאנגלית (בזיכרון גם ובקובץ ה-Excel), ואז `_lookup_label(var, code)` מחזיר את התווית באנגלית לכל שילוב.

---

## תא 10 - קוד - HTML_TEMPLATE + הרכבת index.html

**מה זה עושה:** התא **הגדול והקריטי ביותר** של הפרויקט שלי. בונה קובץ HTML עצמאי אחד (`index.html`) שכולל:

- כל ה-CSS (~800 שורות)
- Plotly.js מלא מוטמע (~3.5 MB) כדי שהאתר יעבוד אופליין
- כל ה-JavaScript של השאלון, הדאשבורד, הטאבים, המפה, הסימולטור, ה-Gauge, ה-profile chart
- את כל תוצרי הטאבים (`OVERVIEW_HTML`, `DATA_QUALITY_HTML`, ...) שהתא הקודם בנה
- את המפה שהתא של Folium בנה (MAP_SRCDOC)
- את השאלון (QUIZ) כ-JSON

**מנגנון ה-Template:** התא בונה `HTML_TEMPLATE` כמחרוזת רגילה של פייתון עם **placeholders** - `__PLOTLYJS__`, `__FIGS__`, `__QUIZ__`, `__MAP_SRCDOC__`, `__OVERVIEW__`, `__DATA_QUALITY__`, `__DESCRIPTIVES__`, `__MODEL_PERF__`, `__MODEL_INTERP__`. בסוף - `HTML_TEMPLATE.replace(...)` מוחלף כל placeholder בערך הרלוונטי.

**קטע קוד מרכזי - שרשרת ההחלפה:**

```python
def js_safe(s):
    """מנטרל '</script>' בתוך JSON strings שנמצאים בתוך <script> block."""
    return s.replace("</script>", "<\\/script>").replace("</Script>", "<\\/Script>")

figs_payload = {name: json.loads(fig.to_json()) for name, fig in FIGS.items()}

html = (HTML_TEMPLATE
        .replace("__PLOTLYJS__", get_plotlyjs())
        .replace("__OVERVIEW__", OVERVIEW_HTML)
        .replace("__FIGS__", js_safe(json.dumps(figs_payload)))
        .replace("__QUIZ__", js_safe(json.dumps(QUIZ)))
        .replace("__MAP_SRCDOC__", js_safe(json.dumps(MAP_SRCDOC)))
        .replace("__DATA_QUALITY__", DATA_QUALITY_HTML)
        .replace("__DESCRIPTIVES__", DESCRIPTIVES_HTML)
        .replace("__MODEL_PERF__", MODEL_PERF_HTML)
        .replace("__MODEL_INTERP__", MODEL_INTERP_HTML))

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)
```

**דגש חשוב על `js_safe`:** בעיה טיפוסית ב-HTML embedding של JSON: אם מחרוזת JSON מכילה `</script>`, הדפדפן חושב שסקריפט הסתיים ושובר את כל הדף. `js_safe` מחליף `</script>` ל-`<\/script>` (חוקי ב-JSON, לא סוגר את ה-script). זה סוג של בעיה שקל לפספס - שאלה קלאסית של מרצה: "מה קורה אם ה-`code_title` הזה מכיל `</script>`?". התשובה שלי: `js_safe` דואג לזה.

### הפונקציות המרכזיות ב-JavaScript של index.html

בנוסף לתוכן, מסמך ה-HTML מכיל **בלוק סקריפט אחד גדול** עם כל הלוגיקה. הפונקציות המרכזיות שלו:

#### `buildQuiz()` - בונה את השאלון

```js
function buildQuiz() {
  var box = el("questions");
  QUIZ.forEach(function (q, i) {
    var d = document.createElement("div"); d.className = "question";
    var t = document.createElement("div"); t.className = "qtext";
    t.textContent = (i + 1) + ". " + q.text; d.appendChild(t);
    q.options.forEach(function (opt, oi) {
      var lab = document.createElement("label"); lab.className = "opt";
      var inp = document.createElement("input");
      inp.type = "radio"; inp.name = q.id; inp.value = oi;
      inp.addEventListener("change", updateProgress);
      lab.appendChild(inp);
      var sp = document.createElement("span"); sp.textContent = opt; lab.appendChild(sp);
      d.appendChild(lab);
    });
    box.appendChild(d);
  });
  updateProgress();
}
```

מסתובב על מערך `QUIZ` (10 שאלות), יוצר `<div>` לכל שאלה, ולכל אחת מוסיף `<label>` עם `<input type="radio">` לכל אפשרות. הרדיו-בטן משתמש ב-`name=q.id` (Q1, Q2, ...) כדי שרק אחד יבחר בכל שאלה.

#### `submitQuiz()` - שער הכניסה לדאשבורד

```js
function submitQuiz() {
  var a = currentAnswers(), missing = [];
  QUIZ.forEach(function (q, i) { if (a[q.id] === null) missing.push(i + 1); });
  if (missing.length) {
    err.innerHTML = "Cannot enter the dashboard yet. Please answer question " +
                     missing.join(", ") + ".";
    return;
  }
  var score = 0, ktot = 0;
  QUIZ.forEach(function (q) {
    if (q.kind === "knowledge") { ktot++; if (a[q.id] === q.correct) score++; }
  });
  window.userScore = score;
  el("quiz-screen").style.display = "none";
  el("dashboard").style.display = "block";
  el("topbar-back").style.display = "inline-flex";
  buildKnowledgeTable(a); buildMap();
  rendered = {};  // איפוס מטמון הרינדור
  showTab("overview");
}
```

מוודא שכל 10 השאלות נענו (אם לא, מציג שגיאה עם רשימת השאלות החסרות ולא מכניס לדאשבורד). אחרי, סופר כמה שאלות ידע ענה נכון (ה-score לגייג'), מסתיר את השאלון, מציג את הדאשבורד ואת כפתור Back-to-quiz בטופ, בונה את טבלת השאלות של הפרופיל, בונה את המפה, ומראה את טאב Overview.

**דגש חשוב:** כפי שרואים כאן, **אף גרף לא מרונדר לפני שהמשתמש עונה על כל 10 השאלות**. השאלון הוא Gate של ממש. Plotly.newPlot לא נקרא ל-Overview עד ל-`showTab("overview")` שנקרא בשורה האחרונה של `submitQuiz`.

#### `showTab(name)` - החלפת טאב + Lazy Rendering

```js
function showTab(name) {
  document.querySelectorAll(".tabbtn").forEach(function (b) {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach(function (p) {
    p.style.display = (p.id === "panel-" + name) ? "block" : "none";
  });
  (TAB_FIGS[name] || []).forEach(function (fid) {
    if (!rendered[fid]) {
      if (fid === "fig-profile") { buildProfileChart(); }
      else if (fid === "fig-gauge") { buildGauge(); }
      else {
        var f = FIGS[fid];
        Plotly.newPlot(fid, f.data, f.layout, { responsive: true, displaylogo: false });
      }
      rendered[fid] = true;
    } else {
      try { Plotly.Plots.resize(fid); } catch (e) {}
    }
  });
  if (name === "geography") { buildMap(); setTimeout(refreshMap, 80); }
  if (name === "young") { setTimeout(initSimulator, 80); }
}
```

**עובד ב-3 שלבים:**

1. **מסמן את הכפתור הפעיל** (`.tabbtn.active`) ומסתיר/מציג את הפאנלים המתאימים (`display: block/none`).
2. **Lazy rendering:** רק אם `rendered[fid]` הוא false, קורא ל-`Plotly.newPlot`. אם כבר רונדר קודם, קורא ל-`Plotly.Plots.resize` כדי שיתאים לרוחב הנוכחי של הפאנל. זו אופטימיזציית ביצועים חשובה - Plotly.newPlot איטי; לא רוצים לרנדר את כל 12 הגרפים בהתחלה, רק לפי צורך.
3. **טיפול מיוחד לטאבים לא-Plotly:**
   - `geography` -> מפעיל את `buildMap()` שמכניס `<iframe srcdoc="MAP_SRCDOC">` ל-`#fig-map`, ואז אחרי 80ms קורא ל-`refreshMap()` שנוגע ב-Leaflet כדי שיתאים לרוחב.
   - `young` -> קורא ל-`initSimulator()` שקושר את הסליידרים של סימולטור ריבית דריבית לפונקציית ה-render שלו.

**דגש חשוב:** למה 80ms? כי CSS `display: block` הוא סינכרוני אבל layout reflow לוקח לדפדפן כמה מילישניות. Leaflet ו-Plotly קוראים את `getBoundingClientRect()` בזמן שהם מתחילים, ואם עדיין הכל בגודל 0 - הם מרנדרים דבר קטן ולא מתקנים. `setTimeout(..., 80)` נותן לדפדפן זמן לחשב את הגודל.

#### `buildMap()` + `refreshMap()` - הזרקת המפה + נדנוד Leaflet

```js
function buildMap() {
  var box = el("fig-map");
  if (box && !box.querySelector("iframe")) {
    var iframe = document.createElement("iframe");
    iframe.setAttribute("srcdoc", MAP_SRCDOC);
    iframe.setAttribute("width", "100%");
    iframe.setAttribute("height", "560");
    box.appendChild(iframe);
  }
}

function refreshMap() {
  var iframe = document.querySelector('#fig-map iframe');
  if (!iframe || !iframe.contentWindow) return;
  var win = iframe.contentWindow;
  var mapKey = Object.keys(win).find(function (k) { return k.indexOf('map_') === 0; });
  if (mapKey && win[mapKey] && win[mapKey].invalidateSize) {
    win[mapKey].invalidateSize();   // Leaflet: מודיע שהמסגרת שינתה גודל
  }
}
```

**נקודה עדינה:** Folium מייצר משתנה גלובלי בתוך ה-iframe בשם `map_<hash>`. אני מוצא אותו דרך `Object.keys(win).find(k => k.startsWith('map_'))` ואז קורא ל-`invalidateSize()` שלו - Leaflet API סטנדרטי שאומר "המסגרת שלי שינתה גודל, חשב מחדש". בלי זה, אם המפה נטענה בזמן שהפאנל היה `display:none` (רוחב 0), היא תיראה מצומקת.

#### `buildGauge()` - Gauge חצי-עגול לציון

בונה SVG של Gauge חצי-עגול עם 3 קשתות (אדום/כתום/ירוק לפי טווח הציון), עם מחט שמצביעה על `userScore/5 * 100` אחוז. כתוב בעיקר ב-SVG paths ידניים כי Plotly לא מספק Gauge לחצי-עיגול מיושר.

#### `buildProfileChart()` - השוואה שלך מול האוכלוסייה

בונה גרף Plotly של עמודות אופקיות: לכל שאלת התנהגות (Q3, Q4, Q6, Q7, Q9), עמודה אחת מציגה את התשובה של המשתמש (Yes=100, No=0), עמודה שנייה מציגה את `popYes` של האוכלוסייה. המשתמש רואה מיידית: "אני מקדים את הממוצע בפנסיה, אבל אחריו בקרן השתלמות".

#### `buildKnowledgeTable(a)` - הטבלה המחודשת של הפרופיל

```js
function buildKnowledgeTable(a) {
  var tbody = el("ktable").querySelector("tbody");
  tbody.innerHTML = "";
  QUIZ.forEach(function (q) {
    var isSurvey = (q.kind === "behavior");
    var badge = document.createElement("span");
    badge.className = "qbadge " + (isSurvey ? "survey" : "knowl");
    badge.textContent = isSurvey ? "From social survey" : "Knowledge check";
    ...
    if (isSurvey) {
      tdRef.textContent = "Yes in population: " + q.popYes.toFixed(1) + "%";
      tdRes.textContent = "-";
    } else {
      tdRef.textContent = q.options[q.correct];
      var ok = a[q.id] === q.correct;
      tdRes.textContent = ok ? "Correct" : "Wrong";
      tdRes.className = ok ? "ok" : "bad";
    }
  });
}
```

מציגה את **כל 10 השאלות** (במקור הציגה רק שאלות ידע). לכל שורה יש תג צבע: **ירוק "From social survey"** לשאלות התנהגות שיש להן משנה תוקף כי הן נמצאות בסקר הרשמי של הלמ"ס, ו**כחול "Knowledge check"** לשאלות ידע שאני הוספתי כדי לבחון אוריינות פיננסית. עבור שאלות התנהגות מוצג `popYes` במקום "תשובה נכונה"; עבור שאלות ידע מוצגים "Correct/Wrong".

#### `renderSim()` + `initSimulator()` - סימולטור ריבית דריבית

```js
function renderSim() {
  var startAge = +el("sim-start-age").value;
  var monthly  = +el("sim-monthly").value;
  var rate     = +el("sim-rate").value / 100;
  var years    = 67 - startAge;
  var r        = rate / 12;
  var months   = years * 12;
  var future   = monthly * (Math.pow(1 + r, months) - 1) / r;   // FV של אנואיטי
  el("sim-result").textContent = fmtNIS(future);
  // ... עוד קריאה לגרף השוואה של "התחלת השקעה מוקדם מול מאוחר"
}

function initSimulator() {
  ["sim-start-age", "sim-monthly", "sim-rate"].forEach(function (id) {
    el(id).addEventListener("input", renderSim);
  });
  renderSim();
}
```

הנוסחה הקלאסית של Future Value של אנואיטי (הפקדה חוזרת) - `M * ((1+r)^n - 1) / r`. הסימולטור משמש את הרעיון החינוכי של הטאב: "התחל להפריש מוקדם - הריבית דריבית עושה את העבודה".

#### `backToQuiz()` - חזרה למסך השאלון

```js
function backToQuiz() {
  document.querySelectorAll("#questions input[type=radio]").forEach(function (r) {
    r.checked = false;   // איפוס כל הבחירות
  });
  var err = el("quiz-error");
  if (err) { err.style.display = "none"; }
  updateProgress();
  el("dashboard").style.display = "none";
  el("quiz-screen").style.display = "block";
  el("topbar-back").style.display = "none";
  window.scrollTo(0, 0);
}
```

מאפס את כל הרדיו-בטנס, מנקה שגיאה, מעדכן מונה "Answered 0/10", מסתיר את הדאשבורד, מציג את מסך השאלון בחזרה, מסתיר את כפתור ה-Back מהטופ. **הכפתור עצמו** בטופ נראה כך:

```html
<button id="topbar-back" class="home-btn" style="display:none" onclick="backToQuiz()">
  <svg>...</svg> Back to quiz
</button>
```

`style="display:none"` בטעינה - עולה רק אחרי submitQuiz מוצלח.

---

## תא 11 - קוד - שרת HTTP מקומי (Colab-safe)

**מה זה עושה:** התנהגות שונה לפי סביבה:

- **מקומית (Jupyter):** מרים שרת פייתון פשוט (`http.server`) על 127.0.0.1 בפורט 8890 (ואם תפוס - 8891, 8892 וכו'), ופותח את `index.html` בדפדפן דרך הפורט הזה. השרת רץ ברקע ב-thread daemon.
- **Colab:** אין דפדפן וגם אין גישה ל-127.0.0.1 מהצד של המשתמש. במקום זאת קורא ל-`google.colab.files.download(OUT_HTML)` שמדליק הורדת קובץ בדפדפן של המשתמש.

**קטע קוד מרכזי:**

```python
if IN_COLAB:
    print(f"Dashboard ready: {OUT_HTML}")
    from google.colab import files
    files.download(OUT_HTML)
else:
    _LOCAL_SERVER, _LOCAL_PORT = start_local_server()
    if _LOCAL_PORT:
        LOCAL_URL = f"http://127.0.0.1:{_LOCAL_PORT}/index.html"
        print(f"Local dashboard is live: {LOCAL_URL}")
        webbrowser.open(LOCAL_URL)
```

**דגשים חשובים לפגישה:**

- **למה בכלל שרת מקומי ולא סתם `file://index.html`?** קבצי HTML שנפתחים בפרוטוקול `file://` חוסמים לפעמים אלמנטים כמו `iframe srcdoc` (בגלל Same-Origin Policy) או fonts חיצוניים. `http://127.0.0.1` פותר את זה ומחקה בדיוק את סביבת GitHub Pages.
- **הקובץ שנפתח בשרת המקומי הוא בדיוק אותו קובץ** שמתפרסם ב-GitHub Pages - סביבת פיתוח = סביבת ייצור. אין הפתעות בפריסה.
- **Thread daemon:** ה-`threading.Thread(target=httpd.serve_forever, daemon=True)` דואג שהשרת ייסגר כשה-kernel של הנוטבוק ייסגר. אין תהליכי zombie.

---

## הקשר Cross-part: איך הוצאתי מ-Shai ומ-Ovad

השלמה חשובה שמופיעה לרוחב כל התאים למעלה: אני מסתמך על תוצרים של Parts 1+2 בשני מסלולים:

1. **In-memory (במצב Restart & Run All הסטנדרטי):** משתני זיכרון גלובליים - `raw`, `structure_summary`, `data_dictionary`, `eda_groups`, `codebook`, `model_data`, `features`, `cv_results`, `saved_figures`, `best_model`.

2. **קבצי handoff על הדיסק (למקרה של Part-3-only):**
   - `model data.csv` - הדאטה הנקי של שי (מאותה קבוצת פיצ'רים שהזין את המודל של עובד)
   - `xgb_model.pkl` - ה-pipeline המאומן של עובד (best_estimator_ מה-GridSearch)
   - `shai_state.pkl` - dict של כל תוצרי הזיכרון של שי (`raw`, `structure_summary`, `data_dictionary`, `eda_groups`, `codebook`, `model_data`)
   - `ovad_state.pkl` - dict של כל תוצרי הזיכרון של עובד (`features`, `cv_results`, `saved_figures`)

בסביבת Colab, תא ה-Handoff הראשון שלי (תא 1 למעלה) בודק אם ה-globals ריקים; אם כן, מציע לך להעלות את 3 הקבצים ופורק אותם ל-`globals()`.

---

## סיכום קצר לפגישה

- **המחברת המאוחדת מורכבת מ-3 חלקים** שרצים ברצף: EDA של שי, מודל של עובד, דאשבורד שלי. הזרימה עוברת גם בזיכרון (משתנים גלובליים) וגם בקבצי handoff שאני הוספתי (`shai_state.pkl`, `ovad_state.pkl`, `xgb_model.pkl`) - כדי לתמוך גם ב-Part-3-only בסביבת Colab.
- **הדאשבורד הוא קובץ HTML אחד עצמאי** (`index.html`) - Plotly.js מוטמע (~3.5 MB), כל הגרפים embedded, כל הטבלאות embedded, המפה כ-iframe srcdoc, ענן המילים כ-base64. אין שרת ייצור. GitHub Pages פשוט מגיש קובץ סטטי.
- **השאלון חוסם את הכניסה לדאשבורד** - אף גרף לא מרונדר עד ל-`submitQuiz()` מוצלח. `Plotly.newPlot` הוא lazy per-tab (רק כשהטאב נפתח) ומקטמן על `Plotly.Plots.resize` כשחוזרים אליו.
- **8 טאבים סופיים:** Overview (עם word cloud), Data Quality, Descriptives, Model Performance, Model Interpretation, Young Generation, **Geography** (הוחזרה מפת Folium אחרי בקשה מפורשת), Your Profile.
- **טאב Your Profile מציג את כל 10 השאלות** עם תגי צבע ("From social survey" ירוק לשאלות התנהגות שיש להן משנה תוקף כי הן בסקר הרשמי, "Knowledge check" כחול לשאלות ידע שהוספתי).
- **תמיכה מלאה ב-Colab:** תא upload prompt לקבצי הקלט, `try/except` על drive.mount, download אוטומטי של index.html במקום שרת מקומי.
