# מדריך תא-אחר-תא של המחברת המאוחדת — חלק Or (הדאשבורד)

מסמך טכני המתאר את החלק של Or במחברת `eda_model_dashboard.ipynb` — Part 3 - Interactive Dashboard. כל תא במסמך מלווה בהסבר תמציתי, בקטעי קוד רלוונטיים ובנקודות טכניות מרכזיות. המסמך אינו מכסה את תאי Part 1 (שי) ו-Part 2 (עובד); הם נדונים רק בהקשר של תלויות בזרימת הדאטה.

המחברת המאוחדת מורכבת משלושה חלקים ברצף: Part 1 (שי — EDA וניקוי), Part 2 (עובד — מודל GMM+XGBoost), Part 3 (Or — דאשבורד). Part 3 מסתמך על תוצרים של הקודמים בשני מסלולים מקבילים:

1. **בזיכרון:** משתנים גלובליים כמו `raw`, `structure_summary`, `data_dictionary`, `eda_groups`, `codebook`, `model_data`, `features`, `cv_results`, `saved_figures` — קיימים לאחר Restart & Run All של המחברת.
2. **בקבצי handoff על הדיסק:** `model data.csv`, `xgb_model.pkl`, `shai_state.pkl`, `ovad_state.pkl` — מאפשרים הרצה של Part 3 בלבד בסשן חדש, למשל ב-Google Colab.

Part 3 בנוי מ-11 תאי קוד ותאי markdown של כותרות. הזרימה בו: טעינת תוצרים מהחלקים הקודמים -> בניית שני גרפי Plotly (Young Generation) + ענן מילים + מפת Folium -> הרכבת HTML לכל טאב -> כתיבה של `index.html` יחיד -> הרמת שרת מקומי שפותח את הקובץ בדפדפן, או הורדה אוטומטית ב-Colab.

---

## תא 1 — קוד — Handoff מ-Part 2: טעינת המודל + Colab bridge

**מה התא עושה:** התא הראשון של Part 3. שני תפקידים משולבים בו:

1. **בדיקה האם דרוש שחזור של מצב הזיכרון של Parts 1 ו-2.** במצב Restart & Run All כל המשתנים כבר קיימים בזיכרון; במצב Part-3-only, למשל בסביבת Colab לאחר קריסת סשן, הזיכרון ריק. במקרה כזה התא קורא את קבצי ה-pickle שנשמרו בסוף Part 1 ובסוף Part 2, ומזריק את תוכנם ל-`globals()`.
2. **טעינת המודל המאומן:** `model = joblib.load("xgb_model.pkl")`.

**קטע קוד מרכזי:**

```python
_stale = any(_need(v) for v in
             ["model_data", "features", "saved_figures", "codebook",
              "structure_summary", "data_dictionary", "eda_groups", "raw"])

if _stale:
    _missing = [f for f in _PART3_REQUIRED if not os.path.exists(f)]
    if _missing and IN_COLAB:
        from google.colab import files
        _uploaded = files.upload()   # דיאלוג העלאה של קבצי pkl
    ...
    _shai_state = joblib.load("shai_state.pkl")
    _ovad_state = joblib.load("ovad_state.pkl")
    globals().update(_shai_state)   # מזריק raw, structure_summary, codebook, model_data...
    globals().update(_ovad_state)   # מזריק features, cv_results, saved_figures

model = joblib.load("xgb_model.pkl")
```

**נקודות טכניות:**

- **`globals().update(...)` הוא המנגנון המרכזי:** dict שנטען מ-pickle מכיל את השמות המקוריים של המשתנים, וההזרקה שלו ל-`globals()` הופכת אותם למשתנים גלובליים במחברת — כאילו Parts 1 ו-2 באמת רצו. זה מה שמאפשר שחזור מלא לאחר סשן חדש.
- **`IN_COLAB`** מוגדר בתא ה-imports הראשון של המחברת המלאה (בראש, לפני החלוקה לחלקים) באמצעות `try/except google.colab`. Part 3 מסתמך על המשתנה הזה ואינו מגלה את הפלטפורמה מחדש.
- **הבחירה ב-`joblib` ולא ב-`pickle` רגיל** נדרשת מפני ש-`saved_figures` הוא dict של אובייקטי `matplotlib.figure.Figure`. joblib מתמודד עם אובייקטים כאלה בצורה יציבה יותר.

---

## תא 2 — קוד — בדיקת סביבה + imports של Part 3

**מה התא עושה:** בודק שהחבילות שנחוצות ל-Part 3 (`plotly`, `folium`, `wordcloud`, `PIL`, `openpyxl`, `matplotlib`) קיימות בסביבה, ומייבא אותן. יש חפיפה מסוימת עם תא ה-imports הכללי בראש המחברת; התא הזה קיים כדי להבטיח שאם מריצים רק את Part 3 מתחילתו, החבילות ייבדקו ויתייבאו גם אז.

**קטע קוד מרכזי:**

```python
REQUIRED = ["numpy", "pandas", "plotly", "openpyxl", "folium", "wordcloud",
            "PIL", "matplotlib"]
OPTIONAL = ["pyarrow"]
```

**נקודות טכניות:**

- `plotly` — מנוע הגרפים האינטראקטיביים.
- `folium` — בונה את המפה של תת-מחוזות (Nafa).
- `wordcloud` — יוצר את ענן המילים.
- `PIL` — נדרש על ידי `wordcloud` לעיבוד תמונה.
- `pyarrow` — אופציונלי, לשמירה וטעינה של הדאטה בפורמט parquet במקום קריאת ה-Excel כל פעם מחדש.

---

## תא 3 — קוד — טעינת דאטה + מיפויי קודים + קואורדינטות תת-מחוזות

**מה התא עושה:** שלוש פעולות:

1. **טעינת הדאטה** מקובץ `data_24.xlsx`. הרצה ראשונה קוראת את ה-Excel (איטי, כדקה) וכותבת cache בפורמט parquet לקובץ `data_24_cache.parquet`. הרצות עוקבות טוענות מ-parquet בפחות משנייה.
2. **הגדרת מילוני קידוד -> תווית לגרפים:** `AGE_LABELS`, `INCOME_LABELS`, `SAT_LABELS`, `LEFTOVER_LABELS`, `OPTIMISM_LABELS`.
3. **הגדרת `NAFA`** — 16 קודי תת-מחוזות של הלמ"ס עם קואורדינטות (lat, lon). המילון הזה מזין את מפת ה-Folium בטאב Geography.

**נקודות טכניות:**

- המילונים משמשים אך ורק לתוויות ציר. הערך הגולמי בדאטה נשאר `1`, `2` או `888888` — לא מוחלף בזיכרון. הטיפול בקודים השמורים מתבצע ב-Part 1 של שי; Part 3 שומר על הדאטה כפי שהוא כדי לא לחדור לתחום שלו.

---

## תא 4 — קוד — הגדרת השאלון הפיננסי (10 שאלות)

**מה התא עושה:** מגדיר את השאלון כרשימה של dictionaries בשם `QUIZ`. במבנה הנתונים יש חמש שאלות ידע (`kind: "knowledge"` עם שדה `correct`) וחמש שאלות התנהגות (`kind: "behavior"` עם שדה `survey_var` המצביע על משתנה בסקר, ועם שדה `popYes` שמחושב בזמן טעינת המחברת).

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

# popYes מחושב מהדאטה הגולמי — לא ערך קשיח:
for q in QUIZ:
    if q.get("kind") == "behavior":
        col = q["survey_var"]
        q["popYes"] = float(((df[col] == q["yes_code"]).sum() / len(df)) * 100)
```

**נקודות טכניות:**

- **החלוקה 5 + 5 היא מהותית:** שאלות הידע מייצרות את הציון של המשתמש (מוצג ב-Gauge חצי-עגול בטאב Your Profile), ושאלות ההתנהגות מייצרות את גרף ההשוואה (מה אחוז ה"כן" של המשתמש מול 6,907 המשיבים בסקר).
- **`popYes` מחושב פעם אחת בזמן טעינת המחברת** ונכתב ל-HTML כחלק ממערך `QUIZ` בפורמט JSON. בדפדפן זו כבר קריאה מטבלה, בלי קריאה לרשת ובלי פייתון בזמן ריצה.
- **חמש שאלות ההתנהגות ממופות ישירות למשתני הסקר:** `HisachonPensyoni`, `HatavaKHishtalmut_wp`, `InternetShilemMatbeaDig`, `HashkaotBank`, `InternetDohPensia`. זו נקודת ההשוואה 1:1 של המשתמש מול המדגם.
- **המבנה של `QUIZ` הוא JSON serializable במלואו.** בהמשך הוא מוכנס לתוך `HTML_TEMPLATE` דרך `json.dumps` (בתא 10).

---

## תא 5 — קוד — Helpers משותפים (`FIGS`, `_clean`, `register`, `age_order`)

**מה התא עושה:** מגדיר את התשתית המשותפת לגרפי Plotly ב-Part 3. לאחר המיזוג עם החלקים של שי ועובד, נותרו רק שני גרפי Plotly בחלק של Or (בטאב Young Generation); שאר הגרפים הוסרו מפני שהטאבים שלהם הוחלפו בטאבים חדשים המציגים תוצרים של Parts 1 ו-2. עם זאת, ה-helpers עדיין נחוצים לגרפים שנותרו.

**קטעי קוד מרכזיים:**

```python
FIGS = {}                     # dict מרכזי - מיפוי key -> Plotly Figure

def _clean(fig, height=430):
    fig.update_layout(height=height, margin=dict(t=70, r=30, b=70, l=60),
                      font=dict(family="Segoe UI, Arial", size=13),
                      title=dict(font=dict(size=17), x=0.5, xanchor="center"),
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showgrid=False, zeroline=False, ...)
    fig.update_yaxes(showgrid=False, zeroline=False, ...)
    return fig

def register(key, fig):
    FIGS[key] = fig         # רישום למאגר שממנו נבנה ה-HTML
    fig.show()              # הצגה inline במחברת עצמה
    return fig

age_order = [AGE_LABELS[k] for k in sorted(AGE_LABELS)]   # ["20-24","25-29",...,"75+"]
```

**נקודות טכניות:**

- ההפרדה בין בניית הגרף לרישום הגרף היא בחירה מכוונת. `register` מהווה חוזה: כל גרף שנוצר חייב לעבור דרכו כדי להופיע גם ב-`FIGS` (שממנו ההרכבה של ה-HTML שולפת) וגם inline במחברת עבור מי שרץ אותה ב-Jupyter. המנגנון מונע מצב שבו גרף נבנה אך לא מחובר לדאשבורד.

---

## תא 6 — קוד — טאב Young Generation (2 גרפים)

**מה התא עושה:** בונה שני גרפי Plotly עבור הטאב Young Generation.

**הגרפים:**

1. **fig-young-digital** — קו של אחוז אימוץ שירותים דיגיטליים פיננסיים לפי גיל (Crypto, בדיקת אשראי אונליין, בדיקת פנסיה אונליין). קבוצת הצעירים מובילה ב-Crypto ובבדיקת אשראי.
2. **fig-young-satisfaction** — עמודות של רמת שביעות רצון מתכנון פרישה, בקבוצת 20-29 בלבד.

**נקודות טכניות:**

- טאב Young Generation מכיל גם **סימולטור ריבית דריבית אינטראקטיבי**, שהלוגיקה שלו יושבת בקוד ה-JavaScript שבתא 10. הסימולטור מציג שהפקדה של 500 ש"ח בחודש בין גיל 25 לגיל 67 בריבית 6% מגיעה לכ-1.1 מיליון ש"ח בסוף התקופה.

---

## תא 7 — קוד — ענן מילים פיננסי (Word Cloud)

**מה התא עושה:** מייצר תמונת PNG של ענן מילים של מונחים פיננסיים, כשגודל כל מילה פרופורציונלי לספירה שלה בפועל בדאטה של הסקר. התמונה נשמרת בזיכרון כמחרוזת base64 ומוטמעת בתוך ה-HTML הסופי.

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
    ...   # 32 מונחים בסך הכל
}

wc = WordCloud(width=1200, height=520, background_color="white",
               colormap="viridis", ...).generate_from_frequencies(FIN_WEIGHTS)

buf = io.BytesIO()
wc.to_image().save(buf, format="PNG")
WORDCLOUD_B64 = base64.b64encode(buf.getvalue()).decode("ascii")
```

**נקודות טכניות:**

- הענן אינו רשימה גנרית של מונחים; כל שקלול מבוסס על ספירה בפועל בדאטה. אם 1,093 משיבים ענו שיש להם פנסיה, המילה `Pension` מקבלת משקל 1,093.
- **טכניקת ה-base64 embedding:** התמונה מוזרקת ל-HTML כ-`<img src="data:image/png;base64,...">`. אין קובץ PNG חיצוני; `index.html` נשאר שלם ועצמאי.
- הענן מוצג בתחתית הטאב Overview, לאחר יתר תוכן הטאב.

---

## תא 8 — קוד — מפת Folium של ישראל

**מה התא עושה:** יוצר מפה אינטראקטיבית של ישראל (Folium מבוסס Leaflet) עם 16 עיגולים — אחד לכל תת-מחוז (Nafa). גודל העיגול פרופורציונלי למספר המשיבים באזור, לפי סקאלת שורש כדי שאזורים קטנים לא יטבעו לצד תל אביב.

**קטע קוד מרכזי:**

```python
m = folium.Map(location=[31.85, 35.10], zoom_start=8,
               tiles="cartodbpositron", min_zoom=7, max_zoom=12, max_bounds=True)

for code_val, meta in NAFA.items():
    n = int(nafa_counts.get(code_val, 0))
    radius = 6 + 22 * (n / max_c) ** 0.5      # sqrt scaling
    folium.CircleMarker(
        location=[meta["lat"], meta["lon"]], radius=radius,
        color="#1f5fc4", fill=True, fill_color="#2c7be5", fill_opacity=0.55,
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=folium.Tooltip(f"<b>{meta['name']}</b>: {n:,} respondents ...", sticky=True)
    ).add_to(m)

MAP_SRCDOC = m.get_root().render()   # HTML מלא של המפה כמחרוזת יחידה
```

**נקודות טכניות:**

- **אין קריאה ל-`fit_bounds`.** ב-Folium 0.20, `fit_bounds` נכנס לזום מקסימלי כדי לכסות בדיוק את גבולות המרקרים; עבור ישראל שהיא קטנה, התוצאה היא זום ברמת רחוב. הבחירה של `zoom_start=8` מייצרת מסגרת נכונה.
- **`MAP_SRCDOC` הוא מחרוזת HTML שלמה** הכוללת CSS ו-JS משל עצמה. היא מוזרקת בהמשך לתוך `<iframe srcdoc>` כך ש-Leaflet רץ בסביבה מבודדת ואינו מתנגש עם ה-CSS וה-JS של הדאשבורד.
- המפה מוצגת בטאב עצמאי בשם Geography.

---

## תא 9 — קוד — הרכבת תוכן HTML לכל טאב

**מה התא עושה:** התא המרכזי של Part 3. אוסף את התוצרים של שי ועובד יחד עם הגרפים והתמונות שנבנו בתאים הקודמים, ומרכיב מחרוזות HTML לכל אחד מחמשת הטאבים הראשיים: `OVERVIEW_HTML`, `DATA_QUALITY_HTML`, `DESCRIPTIVES_HTML`, `MODEL_PERF_HTML`, `MODEL_INTERP_HTML`. כל מחרוזת נכנסת בהמשך למקומה ב-`HTML_TEMPLATE` (בתא 10).

**מבנה כל טאב:**

### `OVERVIEW_HTML` — טאב Overview

- ארבעה כרטיסי KPI: מספר משיבים (6,907), מספר משתנים גולמיים, מספר predictors שנכנסו למודל, שנת הסקר (2024).
- Working Principles של שי — חמישה עקרונות שיסוד הפרויקט: הקובץ הגולמי נשאר ללא שינוי; קודי `888888` ו-`999999` מומרים ל-NaN; אין מחיקה שקטה של שורות רבות; אין scaling בשלב הזה; שקיפות מלאה מול ה-codebook.
- Structure summary (עשר שורות ראשונות) — טבלה מ-Part 1 של שי המציגה שם עמודה, dtype, מספר ערכים ייחודיים ואחוז חסרים.
- Raw data head(10) — עשר שורות ראשונות של `df` לאחר ניקוי בסיסי של שי.
- Features selected for the model — chips של שמות הפיצ'רים שעובד בחר.
- ענן מילים — `<img src="data:image/png;base64,{WORDCLOUD_B64}">` בתחתית הטאב.

### `DATA_QUALITY_HTML` — טאב Data Quality

- Missing values report per column — רשת רב-עמודתית של אריחים קומפקטיים. כל אריח מציג שם משתנה ואחוז חסרים; צביעה לפי חומרה: אדום עבור 50% ומעלה, ענבר עבור 20% ומעלה, ירוק אחרת. הצורה הרב-עמודתית מתאימה ל-324 משתנים בפריסה קריאה בהרבה מטבלה של 324 שורות.
- Data dictionary (עשר שורות ראשונות) — טבלה של שי המרכזת שם משתנה, תיאור מה-codebook של הלמ"ס, והחלטת ניקוי.
- Drop candidates (עשר שורות ראשונות) — עמודות ששי סימן להסרה בגלל אחוז חסרים גבוה או חוסר רלוונטיות.
- CBS codebook (30 שורות ראשונות) — הצגת ה-codebook לאחר preprocessing: 12 שורות מטא הוסרו, שמות עמודות תורגמו לאנגלית, ותוויות המשתנים המרכזיים תורגמו לאנגלית.

### `DESCRIPTIVES_HTML` — טאב Descriptives

- לכל אחת משלוש הקבוצות ב-`eda_groups` של שי (Demographic profile, Financial profile, Social profile) — התפלגות ה-value_counts של המשתנים בקבוצה, מוצגת כ-mini bar plot ב-HTML.
- התוויות של הערכים המספריים מגיעות מהcodebook דרך פונקציית helper בשם `_lookup_label(var, code)` שמוגדרת בתא זה. הפונקציה מחזירה את `code_title` הרלוונטי מהקודבוק, שכבר תורגם לאנגלית בתא ה-preprocessing של הקודבוק (למשל `Male/Female` במקום `1/2`).
- בסוף הטאב — התפלגות של `financial_risk_score`, ה-target המחושב של שי (סקאלה 0-7 של תסמיני סיכון פיננסי).

### `MODEL_PERF_HTML` — טאב Model Performance

- Baseline comparison table — טבלה של עובד המשווה מודלים לפי מדדים (Accuracy, Precision, Recall, F1, ROC AUC).
- שישה גרפי ביצועים ראשונים מתוך `saved_figures`. המשתנה `saved_figures` נבנה ב-Part 2 של עובד דרך monkey-patch של `plt.show` שלוכד כל figure שנוצר. הגרפים מוטמעים כתמונות base64.

### `MODEL_INTERP_HTML` — טאב Model Interpretation

- שישה גרפי SHAP הבאים מתוך `saved_figures` — Beeswarm, feature importance, dependence plots.

**קטעי קוד מרכזיים:**

```python
def _fig_to_img(fig, alt="figure", max_width_px=1000):
    """ממיר Figure של matplotlib ל-<img> base64. שימוש עיקרי בגרפי שי ועובד."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" ...>'

def _df_to_html(df, max_rows=None, classes="datatable"):
    """המרת DataFrame ל-HTML עם cap אופציונלי + כיתוב 'Showing first X of Y rows'."""
    if max_rows is not None and len(df) > max_rows:
        note = f'<div class="table-note">Showing first {max_rows} of {len(df):,} rows</div>'
        return note + df.head(max_rows).to_html(...)
    return df.to_html(...)

def _lookup_label(_var, _code):
    """שולף label מה-codebook (לאחר תרגום לאנגלית) עם fallback לקוד הגולמי."""
    _match = codebook.loc[(codebook["variavle_name"] == _var) &
                          (codebook["code"] == _code), "code_title"]
    return str(_match.iloc[0]).strip() if len(_match) else str(_code)
```

**נקודות טכניות:**

- **הפרדה מלאה בין בניית התוכן להצגתו.** התא הזה בונה מחרוזות HTML בלבד. ההרכבה של מסמך ה-HTML המלא מתבצעת בתא הבא. ההפרדה שומרת על תחזוקיות: שינוי בתוכן טאב אינו דורש נגיעה בתבנית ה-HTML.
- **קפיצות זיכרון:** מטריצת Phi-K של שי (קורלציות רב-סוגית) מצריכה כמה GB ואינה מוצגת כאן; היא הוסרה כדי לא לפוצץ את ה-kernel, וכיתוב הבהרה מופיע בטאב Descriptives.
- **`_lookup_label` הוא הגורם המרכזי בטיפול בתוויות שהיו בעברית.** הקוד של שי משתמש ב-`.value_counts()` על משתני הסקר, שמחזירה קודים מספריים (1, 2, 3). ה-preprocessing של הקודבוק (הוסף במחברת המאוחדת) תרגם את `code_title` לאנגלית גם בזיכרון וגם בקובץ ה-Excel; `_lookup_label(var, code)` מחזיר את התווית המתורגמת לכל שילוב של משתנה וקוד. הקוד המקורי של שי לא נגע.

---

## תא 10 — קוד — HTML_TEMPLATE + הרכבת index.html

**מה התא עושה:** התא הגדול והקריטי ביותר של Part 3. בונה קובץ HTML עצמאי אחד (`index.html`) שכולל:

- כל ה-CSS של הדאשבורד (כ-800 שורות).
- ספריית Plotly.js המלאה מוטמעת (כ-3.5 MB) כדי שהאתר יעבוד אופליין.
- כל ה-JavaScript של השאלון, הדאשבורד, הטאבים, המפה, הסימולטור, ה-Gauge, וגרף הפרופיל.
- את כל תוצרי הטאבים (`OVERVIEW_HTML`, `DATA_QUALITY_HTML`, ...) שהתא הקודם בנה.
- את המפה שהתא של Folium בנה (`MAP_SRCDOC`).
- את השאלון (`QUIZ`) כמחרוזת JSON.

**מנגנון ה-Template:** התא בונה את `HTML_TEMPLATE` כמחרוזת Python רגילה עם placeholders — `__PLOTLYJS__`, `__FIGS__`, `__QUIZ__`, `__MAP_SRCDOC__`, `__OVERVIEW__`, `__DATA_QUALITY__`, `__DESCRIPTIVES__`, `__MODEL_PERF__`, `__MODEL_INTERP__`. בסוף התא, שרשרת של `.replace(...)` מחליפה כל placeholder בערך הרלוונטי.

**קטע קוד מרכזי — שרשרת ההחלפה:**

```python
def js_safe(s):
    """מנטרל '</script>' בתוך JSON strings שמוטמעים בתוך <script> block."""
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

**נקודה חשובה על `js_safe`:** מלכודת ידועה ב-HTML embedding של JSON — אם מחרוזת JSON מכילה `</script>`, הדפדפן מפרש זאת כסיום של בלוק הסקריפט וכל הדף נשבר. `js_safe` מחליף `</script>` ב-`<\/script>` (חוקי ב-JSON, לא סוגר את ה-script). זו בעיה שקל לפספס.

### הפונקציות המרכזיות ב-JavaScript של index.html

בנוסף לתוכן, מסמך ה-HTML מכיל בלוק סקריפט אחד גדול עם כל הלוגיקה. הפונקציות המרכזיות שלו:

#### `buildQuiz()` — בניית השאלון

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

הפונקציה עוברת על מערך `QUIZ` (10 שאלות), יוצרת `<div>` לכל שאלה, ולכל אחת מוסיפה `<label>` עם `<input type="radio">` לכל אפשרות. שם ה-radio (`name=q.id`) מבטיח שרק אפשרות אחת נבחרת בכל שאלה.

#### `submitQuiz()` — שער הכניסה לדאשבורד

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

הפונקציה בודקת שכל עשר השאלות נענו. אם לא, מציגה שגיאה עם רשימת השאלות החסרות ואינה מאפשרת כניסה לדאשבורד. אם כן, סופרת כמה שאלות ידע נענו נכון (הציון של Gauge), מסתירה את השאלון, מציגה את הדאשבורד ואת כפתור ה-Back בטופ, בונה את טבלת השאלות של הפרופיל, בונה את המפה, ומראה את טאב Overview.

הערה חשובה: אף גרף אינו מרונדר לפני `submitQuiz` מוצלח. `Plotly.newPlot` נקרא לראשונה בתוך `showTab("overview")` שבשורה האחרונה של הפונקציה.

#### `showTab(name)` — החלפת טאב + Lazy Rendering

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

הפונקציה פועלת בשלושה שלבים:

1. **סימון הכפתור הפעיל** (`.tabbtn.active`) והצגה/הסתרה של הפאנלים המתאימים (`display: block/none`).
2. **Lazy rendering:** רק אם `rendered[fid]` הוא false, נקראת `Plotly.newPlot`. אם הגרף כבר רונדר, נקראת `Plotly.Plots.resize` כדי שיתאים לרוחב הנוכחי של הפאנל. זו אופטימיזציה חשובה: `Plotly.newPlot` איטי, אז אין צורך לרנדר את כל הגרפים מיד עם טעינת הדף.
3. **טיפול מיוחד בטאבים שאינם Plotly טהור:**
   - `geography` — קורא ל-`buildMap()` שמכניס `<iframe srcdoc="MAP_SRCDOC">` לתוך `#fig-map`, ואז לאחר 80ms קורא ל-`refreshMap()` שמעדכן את Leaflet לרוחב החדש.
   - `young` — קורא ל-`initSimulator()` שקושר את הסליידרים של סימולטור ריבית דריבית לפונקציית ה-render שלו.

הסיבה ל-80ms: CSS `display: block` הוא סינכרוני, אך layout reflow של הדפדפן לוקח כמה מילישניות. Leaflet ו-Plotly קוראים ל-`getBoundingClientRect()` בזמן אתחול; אם עדיין הכל בגודל 0, הם מרנדרים תוכן קטן ואינם מתקנים אחר כך. `setTimeout(..., 80)` נותן לדפדפן זמן לחשב את הגודל.

#### `buildMap()` + `refreshMap()` — הזרקת המפה ועדכון Leaflet

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
    win[mapKey].invalidateSize();   // Leaflet: מודיע שגודל המסגרת השתנה
  }
}
```

נקודה עדינה: Folium מייצר משתנה גלובלי בתוך ה-iframe בשם `map_<hash>`. המשתנה נמצא באמצעות `Object.keys(win).find(k => k.startsWith('map_'))`, ואז נקראת עליו `invalidateSize()` — API של Leaflet שמורה לספרייה לחשב מחדש את הגודל. בלי הקריאה הזו, אם המפה נטענה בזמן שהפאנל היה במצב `display:none` (רוחב 0), היא תראה מצומקת.

#### `buildGauge()` — Gauge חצי-עגול לציון

בונה SVG של Gauge חצי-עגול עם שלוש קשתות (אדום/כתום/ירוק לפי טווח הציון) ומחט המצביעה על `userScore/5 * 100` אחוז. השימוש ב-SVG paths ידניים נובע מכך ש-Plotly אינו מספק Gauge חצי-עגול מיושר בברירת המחדל.

#### `buildProfileChart()` — השוואת המשתמש לאוכלוסייה

בונה גרף Plotly של עמודות אופקיות. לכל שאלת התנהגות (Q3, Q4, Q6, Q7, Q9), עמודה אחת מציגה את התשובה של המשתמש (Yes=100, No=0), ועמודה שנייה מציגה את `popYes` של האוכלוסייה. המשתמש רואה מיידית באילו התנהגויות הוא מקדים את הממוצע ובאילו מפגר אחריו.

#### `buildKnowledgeTable(a)` — הטבלה של טאב Your Profile

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

הטבלה מציגה את כל עשר השאלות. לכל שורה תג צבע: `From social survey` (ירוק) לשאלות התנהגות הנמצאות בסקר הרשמי של הלמ"ס, ו-`Knowledge check` (כחול) לשאלות ידע. עבור שאלות התנהגות מוצג `popYes` (אחוז ה"כן" באוכלוסייה) במקום "תשובה נכונה"; עבור שאלות ידע מוצגים "Correct" או "Wrong".

#### `renderSim()` + `initSimulator()` — סימולטור ריבית דריבית

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
}

function initSimulator() {
  ["sim-start-age", "sim-monthly", "sim-rate"].forEach(function (id) {
    el(id).addEventListener("input", renderSim);
  });
  renderSim();
}
```

הנוסחה של Future Value של אנואיטי (הפקדה חוזרת): `M * ((1+r)^n - 1) / r`. הסימולטור ממחיש את היתרון של הפקדה מוקדמת: הריבית דריבית מכפילה את התוצאה עבור מי שמתחיל מוקדם.

#### `backToQuiz()` — חזרה למסך השאלון

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

הפונקציה מאפסת את כל ה-radio buttons, מנקה שגיאה קודמת, מעדכנת את מונה ההתקדמות ל-"Answered 0/10", מסתירה את הדאשבורד, מציגה שוב את מסך השאלון, ומסתירה את כפתור ה-Back מהטופ. הכפתור עצמו בטופ בנוי כך:

```html
<button id="topbar-back" class="home-btn" style="display:none" onclick="backToQuiz()">
  <svg>...</svg> Back to quiz
</button>
```

הכפתור מוסתר בטעינה (`style="display:none"`) ומופיע רק אחרי submitQuiz מוצלח.

---

## תא 11 — קוד — שרת HTTP מקומי (Colab-safe)

**מה התא עושה:** מגיב באופן שונה לפי סביבת ההרצה:

- **מקומית (Jupyter):** מרים שרת פייתון פשוט (`http.server`) על 127.0.0.1 בפורט 8890 (ואם תפוס — 8891, 8892 וכן הלאה), ופותח את `index.html` בדפדפן דרך הפורט הזה. השרת רץ ברקע ב-thread daemon.
- **Colab:** אין דפדפן ואין גישה ל-127.0.0.1 מהצד של המשתמש. במקום זאת התא קורא ל-`google.colab.files.download(OUT_HTML)`, שמפעיל הורדה של הקובץ לדפדפן של המשתמש.

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

**נקודות טכניות:**

- **הצורך בשרת מקומי:** קבצי HTML הנפתחים ב-`file://` חוסמים לעיתים אלמנטים כמו `iframe srcdoc` (מדיניות Same-Origin) או פונטים חיצוניים. `http://127.0.0.1` פותר זאת ומחקה את סביבת GitHub Pages.
- **סביבת הפיתוח זהה לסביבת הייצור:** הקובץ שנפתח בשרת המקומי הוא בדיוק אותו קובץ המתפרסם ב-GitHub Pages. אין הפתעות בפריסה.
- **thread daemon:** ה-`threading.Thread(target=httpd.serve_forever, daemon=True)` מבטיח שהשרת ייסגר עם סגירת ה-kernel של המחברת, ולא יישאר תהליך zombie.

---

## הקשר Cross-part: התלויות ב-Shai וב-Ovad

Part 3 מסתמך על תוצרים של Parts 1 ו-2 בשני מסלולים מקבילים:

1. **In-memory (במצב Restart & Run All):** משתני זיכרון גלובליים — `raw`, `structure_summary`, `data_dictionary`, `eda_groups`, `codebook`, `model_data`, `features`, `cv_results`, `saved_figures`, `best_model`.

2. **קבצי handoff על הדיסק (למצב Part-3-only):**
   - `model data.csv` — הדאטה הנקי של שי (אותו set של פיצ'רים שהזין את המודל של עובד).
   - `xgb_model.pkl` — ה-pipeline המאומן של עובד (`best_estimator_` מ-GridSearchCV).
   - `shai_state.pkl` — dict של כל תוצרי הזיכרון של שי (`raw`, `structure_summary`, `data_dictionary`, `eda_groups`, `codebook`, `model_data`).
   - `ovad_state.pkl` — dict של כל תוצרי הזיכרון של עובד (`features`, `cv_results`, `saved_figures`).

בסביבת Colab, תא ה-Handoff הראשון של Part 3 (תא 1 לעיל) בודק אם ה-globals ריקים; במקרה כזה הוא מציע להעלות את שלושת הקבצים ופורק אותם ל-`globals()`.

---

## סיכום

- המחברת המאוחדת מורכבת משלושה חלקים הרצים ברצף: EDA של שי, מודל של עובד, דאשבורד של Or. הזרימה עוברת גם בזיכרון (משתנים גלובליים) וגם בקבצי handoff (`shai_state.pkl`, `ovad_state.pkl`, `xgb_model.pkl`) לתמיכה גם ב-Part-3-only בסביבת Colab.
- הדאשבורד הוא קובץ HTML יחיד ועצמאי (`index.html`) — Plotly.js מוטמע (כ-3.5 MB), כל הגרפים embedded, כל הטבלאות embedded, המפה כ-iframe srcdoc, ענן המילים כ-base64. אין שרת ייצור. GitHub Pages מגיש קובץ סטטי.
- השאלון חוסם את הכניסה לדאשבורד — אף גרף אינו מרונדר לפני `submitQuiz()` מוצלח. `Plotly.newPlot` מופעל בעצלתיים per-tab (רק כשהטאב נפתח), ו-`Plotly.Plots.resize` מטפל בחזרה לטאב שכבר רונדר.
- שמונה טאבים בסופו של דבר: Overview (עם word cloud בתחתית), Data Quality, Descriptives, Model Performance, Model Interpretation, Young Generation, Geography (מפת Folium), Your Profile.
- טאב Your Profile מציג את כל עשר השאלות עם תגי צבע: `From social survey` (ירוק) לשאלות התנהגות הנמצאות בסקר הרשמי, ו-`Knowledge check` (כחול) לשאלות ידע.
- תמיכה מלאה ב-Colab: תא upload prompt לקבצי הקלט, `try/except` על `drive.mount`, הורדה אוטומטית של `index.html` במקום שרת מקומי.
