<div dir="rtl" align="right">

# מדריך תא-אחר-תא של המחברת המאוחדת - חלק Or (הדאשבורד)

מסמך טכני מפורט המתאר את החלק של Or במחברת `eda_model_dashboard.ipynb` - Part 3 - Interactive Dashboard. כל תא מלווה בהסבר תמציתי, בקטעי קוד מרכזיים, ובנקודות טכניות מעמיקות שאינן מסתמכות על ידע קודם. המסמך אינו מכסה את תאי Part 1 (שי) ו-Part 2 (עובד); הם מתוארים במסמך `NOTEBOOK_WALKTHROUGH_he.md`.

---

## הקדמה - הקשר Cross-part

המחברת המאוחדת מורכבת משלושה חלקים ברצף: Part 1 (שי - EDA וניקוי), Part 2 (עובד - מודל GMM+XGBoost), Part 3 (Or - דאשבורד). Part 3 מסתמך על תוצרים של הקודמים בשני מסלולים מקבילים:

1. **בזיכרון:** משתנים גלובליים כמו `raw`, `structure_summary`, `data_dictionary`, `eda_groups`, `codebook`, `model_data`, `features`, `cv_results`, `saved_figures` - קיימים לאחר Restart & Run All של המחברת.
2. **בקבצי handoff על הדיסק:** `model data.csv`, `xgb_model.pkl`, `shai_state.pkl`, `ovad_state.pkl` - מאפשרים הרצה של Part 3 בלבד בסשן חדש, למשל ב-Google Colab.

Part 3 בנוי מ-11 תאי קוד ותאי markdown של כותרות. הזרימה בו: טעינת תוצרים מהחלקים הקודמים -> בניית שני גרפי Plotly (Young Generation) + ענן מילים + מפת Folium -> הרכבת HTML לכל טאב -> כתיבה של `index.html` יחיד -> הרמת שרת מקומי שפותח את הקובץ בדפדפן, או הורדה אוטומטית ב-Colab.

### מונחי מפתח למי שלא מכיר

- **`pickle` / `joblib`:** פורמט serialization של פייתון. שומר אובייקטים בזיכרון (dict, DataFrame, מודל sklearn, Figure של matplotlib) לקובץ בינארי שאפשר לטעון בחזרה ולקבל את אותו אובייקט בדיוק. `joblib` הוא wrapper של pickle שממוטב לאובייקטים גדולים ומטפל טוב יותר בגרפי matplotlib ובמודלי sklearn.
- **`globals()`:** מחזיר את מרחב השמות הגלובלי של המודול הנוכחי (במחברת - המחברת עצמה). קריאה ל-`globals().update({'x': 42})` מגדירה משתנה גלובלי `x` באותה הצורה שהיינו כותבים `x = 42` בשורה בקוד. זו טכניקה שמאפשרת "להזריק" מצב שלם למחברת אחרי טעינה מ-pickle.
- **`monkey-patching`:** דריסה של פונקציה קיימת בפונקציה חדשה, בזמן ריצה. שינוי לא בקוד המקור של הספרייה אלא באובייקט של הספרייה שיש בזיכרון. מסוכן ולא מומלץ ברוב המקרים, אך שימושי כאשר רוצים "לגנוב" את היציאה של פונקציה כדי לתעד אותה בלי לגעת בקוד שקורא לה.
- **`iframe srcdoc`:** מאפיין של תג `<iframe>` בדפדפן שמאפשר להזין ל-iframe מחרוזת HTML שלמה במקום URL. הדפדפן מרנדר את המחרוזת בתוך iframe כאילו הייתה דף נפרד - כולל CSS, JavaScript, ומה שלא יהיה. משמש כאן להטמעה של מפת Folium.
- **`base64 embedding`:** קידוד של קובץ בינארי (תמונה, למשל) כמחרוזת ASCII (אותיות, ספרות, `+`, `/`). המחרוזת יכולה להיכנס ישירות ל-HTML: `<img src="data:image/png;base64,iVBOR...">`. הדפדפן מפרש את המחרוזת ומציג את התמונה בלי לפנות לקובץ חיצוני.

---

## תא 1 - קוד - Handoff מ-Part 2: טעינת המודל + Colab bridge

**מה התא עושה:** התא הראשון של Part 3. שני תפקידים משולבים:

1. **בדיקה האם דרוש שחזור של מצב הזיכרון של Parts 1 ו-2.** במצב Restart & Run All כל המשתנים כבר קיימים בזיכרון. במצב Part-3-only (למשל בסביבת Colab לאחר קריסת סשן), הזיכרון ריק. במקרה כזה, התא קורא את קבצי ה-pickle שנשמרו בסוף Part 1 ובסוף Part 2, ומזריק את תוכנם ל-`globals()`.
2. **טעינת המודל המאומן** מ-`xgb_model.pkl` למשתנה `model`.

**קטע קוד מלא:**

```python
import joblib
import os

_PART3_REQUIRED = ["xgb_model.pkl", "shai_state.pkl", "ovad_state.pkl"]

def _need(v):
    return v not in globals()

_stale = any(_need(v) for v in
             ["model_data", "features", "saved_figures", "codebook",
              "structure_summary", "data_dictionary", "eda_groups", "raw"])

if _stale:
    _missing = [f for f in _PART3_REQUIRED if not os.path.exists(f)]
    if _missing:
        raise FileNotFoundError(
            f"Cannot run Part 3 alone without: {_missing}. "
            "Run Parts 1 and 2 first, or place the handoff files next to the notebook."
        )
    _shai_state = joblib.load("shai_state.pkl")
    _ovad_state = joblib.load("ovad_state.pkl")
    globals().update(_shai_state)
    globals().update(_ovad_state)
    print(f"Restored: {list(_shai_state.keys()) + list(_ovad_state.keys())}")

model = joblib.load("xgb_model.pkl")
print(f"Loaded xgb_model.pkl -> {type(model).__name__}")
```

**נקודות טכניות מפורטות:**

- **מהו `_stale`?** לא בהכרח שהמצב "מיושן" (עם ערכים ישנים), אלא ש**המצב חסר**. אם אחד מהמשתנים החיוניים אינו קיים ב-globals - סימן שלא רצו Parts 1 ו-2 בסשן הזה. במצב כזה, יש לטעון את המצב מהקובץ. בחירת שם המשתנה `_stale` מטעה קצת, אבל היא מבטאת את הרעיון "לא מעודכן ביחס למה שצריך".
- **המנגנון של `globals().update(...)`:** קבצי pickle שומרים dict שמפתחותיו הם שמות משתנים ו-ערכיהם הם האובייקטים עצמם. כשקוראים אותם ומזריקים אותם ל-`globals()`, זה שקול להצהרה `raw = ...; structure_summary = ...; codebook = ...` וכן הלאה בשורה במחברת. אחרי הפעולה, שאר תאי Part 3 יכולים להשתמש במשתנים האלה כאילו הם הוגדרו במחברת במפורש. ההוכחה: תאים 128, 133 וכו' מתייחסים ל-`raw`, `codebook`, `features` בלי להצהיר עליהם - כי הם נטענו ל-globals כאן.
- **למה `joblib.load` ולא `pickle.load`?** ה-dict של `ovad_state` מכיל את `saved_figures` - dict של אובייקטי `matplotlib.figure.Figure`. `pickle` הסטנדרטי לפעמים נופל על אובייקטי matplotlib עם קשרים למנועי backend. `joblib` מטפל בזה יותר יציב, וגם מבצע דחיסה טובה יותר לאובייקטים מספריים גדולים כמו arrays של numpy שנמצאים בפנים.
- **`IN_COLAB`** מוגדר בתא ה-imports הראשון של המחברת המלאה (בראש, לפני החלוקה לחלקים) באמצעות `try/except google.colab`. אם ה-import מצליח - סימן שאנחנו ב-Colab (הוא מייצא את המודול הזה בסביבת ה-runtime שלו); אחרת - מקומית.
- **מה קורה כשקובץ חסר?** התא זורק `FileNotFoundError` עם הודעה ברורה. המשתמש מבין שעליו להריץ את Parts 1 ו-2 קודם, או להביא את קבצי ה-pkl לאותה תיקייה של המחברת.

---

## תא 2 - קוד - imports ייחודיים ל-Part 3

**מה התא עושה:** מייבא רק את החבילות שהחלק שלי צריך ולא נטענו כבר בתא ה-imports הראשי בראש המחברת. אחרי צמצום המצגת, התא הזה מכיל שבע שורות בלבד.

**חבילות ותפקידן:**

- **`webbrowser`** - פותח את הדאשבורד בדפדפן ברירת המחדל אחרי שהוא נבנה.
- **`plotly.express` (px) + `plotly.graph_objects` (go)** - שני API שונים של Plotly. px לגרפים סטנדרטיים מהירים, go לשליטה מלאה על מבנה הגרף.
- **`folium`** - עוטף Python מעל `Leaflet.js`, ספריית מפות ב-JavaScript. מייצר HTML מלא של מפה שאפשר להטמיע ב-iframe.
- **`wordcloud`** - יוצר תמונת PNG של ענן מילים לפי משקלים.
- **`px.defaults.template = "plotly_white"`** - מגדיר עיצוב ברירת מחדל לבן ונקי לכל גרפי plotly.express.

**מה כבר קיים מתא ה-imports הראשי ולכן לא חוזר כאן:** `numpy`, `pandas`, `os`, `io`, `json`, `base64`, `warnings`, `matplotlib`, `joblib`, בדיקת גרסאות של החבילות, וזיהוי סביבת `IN_COLAB`.

---

## תא 3 - קוד - טעינת דאטה + מיפויי קודים + קואורדינטות תת-מחוזות

**מה התא עושה:** שלוש פעולות:

1. **טעינת הדאטה** מקובץ `data_24.xlsx` בקריאה חד-פעמית.
2. **הגדרת מילוני קידוד -> תווית לגרפים** (`AGE_LABELS`, `INCOME_LABELS`, `SAT_LABELS`, `LEFTOVER_LABELS`, `OPTIMISM_LABELS`).
3. **הגדרת `NAFA`** - 16 קודי תת-מחוזות של הלמ"ס עם קואורדינטות (lat, lon). זה מזין את מפת Folium בטאב Geography.

**קטע קוד:**

```python
DATA_PATH = "data_24.xlsx"
df = pd.read_excel(DATA_PATH)
print(f"Social Survey 2024 loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
```

**נקודות טכניות מפורטות:**

- **קריאה חד-פעמית של Excel**. הרצה של המחברת ב-Restart & Run All לוקחת כ-10-15 שניות רק לתא הזה. המחברת נועדה להצגה, לא לפיתוח חוזר, אז אין טעם ב-cache על הדיסק.
- **המילונים משמשים אך ורק לתוויות ציר.** הערך הגולמי בדאטה נשאר `1`, `2` או `888888` - לא מוחלף. הטיפול בקודים השמורים מתבצע ב-Part 1 של שי; Part 3 שומר על הדאטה כפי שהוא כדי לא לחדור לתחום שלו.
- **NAFA - 16 תת-מחוזות בישראל.** קוד `nafa` מופיע בדאטה של הלמ"ס לכל משיב; המיפוי בקובץ מקשר קוד לשם באנגלית + קואורדינטות. הקואורדינטות אינן מדויקות של מרכז הישוב; הן קרובות למרכז הגיאוגרפי של האזור. Folium בונה את המפה על סמך המפה הזאת.

---

## תא 4 - קוד - הגדרת השאלון הפיננסי (10 שאלות)

**מה התא עושה:** מגדיר את השאלון כרשימה של dictionaries בשם `QUIZ`. חמש שאלות ידע (`kind: "knowledge"` עם שדה `correct`) וחמש שאלות התנהגות (`kind: "behavior"` עם שדה `survey_var` המצביע על משתנה בסקר, ועם שדה `popYes` שמחושב בזמן טעינת המחברת).

**קטע קוד מלא של שאלה מכל סוג:**

```python
QUIZ = [
    # שאלת ידע - יש תשובה נכונה
    {"id": "Q1", "kind": "knowledge",
     "text": "What is a typical annual management fee (dmei nihul) on a Study Fund...",
     "options": ["0.05% to 0.3%", "0.5% to 1%", "2% to 3%", "5% to 10%"],
     "correct": 1},   # אינדקס של האפשרות הנכונה (0-based)

    # שאלת התנהגות - אין תשובה נכונה, יש השוואה לאוכלוסייה
    {"id": "Q3", "kind": "behavior",
     "survey_var": "HisachonPensyoni",   # שם המשתנה בסקר של הלמ"ס
     "yes_code": 1,                       # הקוד שמייצג "כן" ב-CBS
     "text": "Do you currently have a pension savings account?",
     "options": ["Yes", "No"],
     "popYes": 63.8},  # אחוז המשיבים שענו כן (יחושב בהמשך)
    ...
]

# popYes מחושב מהדאטה - לא ערך קשיח:
for q in QUIZ:
    if q.get("kind") == "behavior":
        col = q["survey_var"]
        # בין המשיבים שיש להם ערך תקין (לא 888888/999999):
        valid = df[col].isin([1, 2])
        if valid.sum() > 0:
            q["popYes"] = float(((df.loc[valid, col] == q["yes_code"]).sum() /
                                 valid.sum()) * 100)
```

**נקודות טכניות מפורטות:**

- **החלוקה 5+5 היא מהותית:**
  - שאלות הידע מייצרות את **ציון המשתמש**. ה-Gauge בטאב Your Profile מציג את אחוז השאלות הנכונות (score/5).
  - שאלות ההתנהגות מייצרות את **גרף ההשוואה** בטאב Your Profile. לכל שאלה יש עמודה שמראה את התשובה של המשתמש (Yes=100, No=0) לצד עמודה שמראה את `popYes` של האוכלוסייה. המשתמש רואה מיידית באילו התנהגויות הוא מקדים את הממוצע ובאילו מפגר.
- **חישוב `popYes` בזמן טעינת המחברת, פעם אחת.** למה זה חשוב? כי אחרי החישוב, `QUIZ` נכתב ל-HTML כחלק ממחרוזת JSON. בדפדפן זה כבר סתם קריאה של מערך - אין קריאה לרשת, אין הרצת פייתון בזמן שהמשתמש מסתכל.
- **חמש שאלות ההתנהגות ממופות ישירות למשתני הסקר:** `HisachonPensyoni`, `HatavaKHishtalmut_wp`, `InternetShilemMatbeaDig`, `HashkaotBank`, `InternetDohPensia`. שם המשתנה בסקר הוא בתעתיק ולא בשם ידידותי לקורא, כדי לשמור על עקיבה מלאה לקובץ המקורי של הלמ"ס.
- **המבנה של `QUIZ` הוא JSON serializable במלואו** (רק ערכים primitive: string, int, float, list). זה חשוב כי בתא 10 המבנה מוכנס ל-HTML דרך `json.dumps(QUIZ)`. אילו היו שדות מסוג `datetime` או `numpy.array`, ה-serialization הייתה נכשלת.

---

## תא 5 - קוד - Helpers משותפים (`FIGS`, `_clean`, `age_order`)

**מה התא עושה:** מגדיר את התשתית המשותפת לגרפי Plotly ב-Part 3. אחרי המיזוג עם החלקים של שי ועובד, נותרו רק שני גרפי Plotly ב-Part 3 (בטאב Young Generation). התא מכיל שלושה דברים בלבד.

**קטע קוד מלא:**

```python
FIGS = {}   # dict מרכזי - מיפוי key -> Plotly Figure

def _clean(fig, height=430):
    """Remove gridlines, set consistent typography and margins."""
    fig.update_layout(
        height=height, margin=dict(t=70, r=30, b=70, l=60),
        font=dict(family="Segoe UI, Arial", size=13),
        title=dict(font=dict(size=17), x=0.5, xanchor="center"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True,
                     linecolor="#d1d5db", ticks="outside", tickcolor="#d1d5db")
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True,
                     linecolor="#d1d5db", ticks="outside", tickcolor="#d1d5db")
    return fig

age_order = [AGE_LABELS[k] for k in sorted(AGE_LABELS)]
```

**נקודות טכניות מפורטות:**

- **`FIGS` הוא dict מרכזי.** הגרפים נכתבים לתוכו ישירות בתא של Young Generation (למשל `FIGS["fig-young-digital"] = fig`). אין פונקציית עטיפה - עבור שני גרפים בלבד, השורה הישירה קצרה יותר ופחות מעורפלת.
- **מה `_clean` עושה בפועל?** ברירת המחדל של Plotly מציגה רשתות (gridlines) מרובעות, כותרות ממוקמות שמאל, וריצוד ברקע. `_clean` מסיר את כל הגרידים והרקעים, מציב את הכותרות במרכז, וקובע פונט אחיד. זו החלטה אסתטית שגורמת לגרפים להיראות אחידים על פני הטאבים.
- **`age_order`** - משמש ב-`reindex` בתא הבא כדי לשמור על סדר גילאים כרונולוגי במקום מיון אלפביתי של pandas.

---

## תא 6 - קוד - טאב Young Generation (2 גרפים)

**מה התא עושה:** בונה שני גרפי Plotly עבור טאב Young Generation.

**הגרפים:**

1. **fig-young-digital** - קו לפי גיל של אחוז אימוץ שירותים דיגיטליים פיננסיים. מציג שלושה משתני sav-survey לאורך `age_order`: תשלום בקריפטו (`InternetShilemMatbeaDig`), בדיקת אשראי אונליין (`InternetDohAshrai`), ובדיקת פנסיה אונליין (`InternetDohPensia`). קבוצת הצעירים מובילה ב-crypto ובבדיקת אשראי.
2. **fig-young-satisfaction** - עמודות של רמת שביעות רצון מתכנון פרישה, בקבוצת 20-29 בלבד. מציג את התפלגות `MerutzeTichnunFinPrisha` - סקאלה 1 (מרוצה מאוד) עד 4 (לא מרוצה בכלל), עם קטגוריה נוספת של "Don't know" עבור קוד 888888.

**נקודות טכניות:**

- הטאב Young Generation מכיל גם **סימולטור ריבית דריבית אינטראקטיבי**, שהלוגיקה שלו יושבת בקוד ה-JavaScript של תא 10. הסימולטור מציג שהפקדה של 500 ש"ח בחודש בין גיל 25 לגיל 67 בריבית 6% מגיעה לכ-1.21 מיליון ש"ח בסוף התקופה.
- **`_clean(fig)`** מוחל על כל גרף לפני הכנסתו ל-`FIGS`. אחריו: `fig.show()` להצגה inline במחברת, ואז `FIGS["fig-young-digital"] = fig`.

---

## תא 7 - קוד - ענן מילים פיננסי (Word Cloud)

**מה התא עושה:** מייצר תמונת PNG של ענן מילים של מונחים פיננסיים, כשגודל כל מילה פרופורציונלי לספירה שלה בפועל בדאטה של הסקר. התמונה נשמרת בזיכרון כמחרוזת base64 ומוטמעת בתוך ה-HTML הסופי.

**קטע קוד מלא:**

```python
def yes_count(var):
    """מספר המשיבים שענו 'כן' (קוד 1) על משתנה מסוים."""
    return int((df[var] == 1).sum())

def nonmissing_count(var):
    """מספר המשיבים שיש להם תשובה תקינה (לא 888888, לא 999999)."""
    return int(len(df) - (df[var].isin([888888, 999999])).sum())

FIN_WEIGHTS = {
    "Pension":          yes_count("HisachonPensyoni"),      # ~4,400
    "Study Fund":       yes_count("HatavaKHishtalmut_wp"),  # ~2,500
    "Keren Hishtalmut": yes_count("HatavaKHishtalmut_wp"),  # sinonimos
    "Bitcoin":          yes_count("InternetShilemMatbeaDig"), # ~165
    "Loan":             yes_count("HalvaaBank"),             # ~1,900
    "Investments":      yes_count("HashkaotBank"),           # ~3,500
    ...   # 32 מונחים בסך הכל
}

wc = WordCloud(width=1200, height=520, background_color="white",
               colormap="viridis", prefer_horizontal=0.85,
               relative_scaling=0.55, min_font_size=14,
               max_words=len(FIN_WEIGHTS)).generate_from_frequencies(FIN_WEIGHTS)

# המרה של התמונה למחרוזת base64
buf = io.BytesIO()               # buffer בזיכרון (לא קובץ)
wc.to_image().save(buf, format="PNG")   # שמירת PNG לתוך ה-buffer
WORDCLOUD_B64 = base64.b64encode(buf.getvalue()).decode("ascii")
```

**נקודות טכניות מפורטות:**

- **הענן אינו רשימה גנרית של מונחים.** כל שקלול מבוסס על ספירה בפועל בדאטה. אם 1,093 משיבים ענו שיש להם פנסיה, המילה `Pension` מקבלת משקל 1,093. `WordCloud` ממיר את המשקלים לגודל פונט לפי סקאלה לוגריתמית (`relative_scaling=0.55`) כדי שמונחים נדירים לא ייעלמו וגם המונחים הכי שכיחים לא ישתלטו.
- **base64 embedding - מה זה בעצם עושה?** תמונת PNG היא רצף של בייטים (בערך 20-30 KB). ב-HTML לא ניתן להטמיע ישירות בייטים בינאריים. base64 ממיר כל 3 בייטים ל-4 תווים ASCII מקבוצה של 64 תווים (A-Z, a-z, 0-9, +, /). זה מגדיל את הגודל בכ-33%, אבל המחרוזת נכנסת ישירות ל-`<img src="data:image/png;base64,...">`. הדפדפן מפרש את המחרוזת ויוצר תמונה בזיכרון בלי להצטרך לפנות לקובץ חיצוני. התוצאה: `index.html` שלם ועצמאי.
- **`io.BytesIO()`** - buffer בזיכרון שמאחורי הקלעים מתפקד כאילו הוא קובץ. `wc.to_image().save(buf, format="PNG")` שומר את התמונה לתוך ה-buffer באותה סמנטיקה של `save(f)` על קובץ אמיתי. הבחירה ב-buffer בזיכרון היא כדי לא לגעת בדיסק - התמונה עוברת ישר ל-base64 ומשם ל-HTML.
- **הענן מוצג בתחתית הטאב Overview**, לאחר יתר תוכן הטאב.

---

## תא 8 - קוד - מפת Folium של ישראל

**מה התא עושה:** יוצר מפה אינטראקטיבית של ישראל (Folium מבוסס Leaflet) עם 16 עיגולים - אחד לכל תת-מחוז (Nafa). גודל העיגול פרופורציונלי למספר המשיבים באזור, לפי סקאלת שורש כדי שאזורים קטנים לא יטבעו לצד תל אביב.

**קטע קוד מלא:**

```python
nafa_counts = df["nafa"].value_counts().to_dict()

m = folium.Map(location=[31.85, 35.10], zoom_start=8,
               tiles="cartodbpositron",     # רקע בהיר ומינימלי (לא Google Maps)
               control_scale=True,
               min_zoom=7, max_zoom=12,      # מגביל zoom כדי שהמפה לא תזוז מדי
               max_bounds=True)

max_c = max(nafa_counts.values()) if nafa_counts else 1

for code_val, meta in NAFA.items():
    n = int(nafa_counts.get(code_val, 0))
    if n == 0:
        continue
    # sqrt scaling: שטח העיגול פרופורציונלי ל-n (רדיוס פרופורציונלי לשורש)
    radius = 6 + 22 * (n / max_c) ** 0.5

    folium.CircleMarker(
        location=[meta["lat"], meta["lon"]],
        radius=radius,
        color="#1f5fc4", weight=1.5,
        fill=True, fill_color="#2c7be5", fill_opacity=0.55,
        popup=folium.Popup(popup_html, max_width=280),        # פותח על click
        tooltip=folium.Tooltip(
            f"<b>{meta['name']}</b>: {n:,} respondents ({n/len(df)*100:.1f}%)",
            sticky=True                                        # נשאר עד שהעכבר עוזב
        ),
    ).add_to(m)

MAP_SRCDOC = m.get_root().render()   # HTML מלא של המפה כמחרוזת יחידה
```

**נקודות טכניות מפורטות:**

- **מה זה Leaflet?** ספריית JavaScript חינמית של פאנלים ומפות אינטראקטיביות. Folium בפייתון בעצם מייצר קובץ HTML שמטעין את `leaflet.js`, מגדיר map object, ומוסיף לו markers. התוצאה של `m.get_root().render()` היא הקוד המלא של אותו קובץ HTML כמחרוזת. הוא כולל `<script src="leaflet.js">`, `<div id="map_...">`, ו-JavaScript שמאתחל את המפה.
- **למה `sqrt scaling` ולא `linear`?** אם הרדיוס פרופורציונלי ל-`n` באופן ליניארי, אזור עם 1,093 משיבים יגרום לעיגול פי 60 מאזור עם 18 משיבים - העיגול הענק יבלע את הקטן. אם הרדיוס פרופורציונלי לשורש של `n`, השטח של העיגול (proportional ל-`radius^2`) הוא פרופורציונלי ל-`n` - זה יחס יותר הוגן וויזואלית ברור.
- **`max_bounds=True`** מונע מהמשתמש לגרור את המפה אל מחוץ לישראל. `min_zoom=7, max_zoom=12` מונעים zoom out קיצוני (שיגרום לכל המדינה להיראות כמו נקודה) או zoom in קיצוני (שיוציא את המרקרים מהמסך). כל אלה החלטות UX.
- **אין קריאה ל-`fit_bounds`.** ב-Folium 0.20, `fit_bounds` מנסה למצוא את הזום המקסימלי שמכסה בדיוק את גבולות המרקרים. עבור ישראל (שהיא קטנה בקנה מידה של Leaflet), התוצאה יוצאת ברמת רחוב - לא שימושי. הבחירה של `zoom_start=8` מייצרת מסגרת נכונה לישראל.
- **`MAP_SRCDOC` הוא מחרוזת HTML שלמה** הכוללת CSS ו-JavaScript משל עצמה. היא מוזרקת בהמשך לתוך `<iframe srcdoc>` כך ש-Leaflet רץ בסביבה מבודדת ואינו מתנגש עם ה-CSS וה-JS של הדאשבורד. זו סיבה חשובה שהמפה מוצגת ב-iframe: Leaflet מגדיר CSS גלובלי (`.leaflet-container`, `.leaflet-tile` וכו') שהיה מתנגש עם מחלקות אחרות ב-HTML.
- **`sticky=True` על ה-tooltip** משמעו: לא נעלם ברגע שהעכבר עוזב את המרקר, נעלם רק אחרי שהעכבר עוזב את התיבה שלו. UX טוב לתיאור ארוך.
- המפה מוצגת בטאב עצמאי בשם Geography.

---

## תא 9 - קוד - הרכבת תוכן HTML לכל טאב

**מה התא עושה:** התא המרכזי של Part 3. אוסף את התוצרים של שי ועובד יחד עם הגרפים והתמונות שנבנו בתאים הקודמים, ומרכיב מחרוזות HTML לכל אחד מחמשת הטאבים הראשיים: `OVERVIEW_HTML`, `DATA_QUALITY_HTML`, `DESCRIPTIVES_HTML`, `MODEL_PERF_HTML`, `MODEL_INTERP_HTML`. כל מחרוזת נכנסת בהמשך למקומה ב-`HTML_TEMPLATE` (בתא 10).

### מבנה כל טאב

#### `OVERVIEW_HTML` - טאב Overview

- ארבעה כרטיסי KPI: מספר משיבים (6,907), מספר משתנים גולמיים, מספר predictors שנכנסו למודל, שנת הסקר (2024).
- Working Principles של שי - חמישה עקרונות שיסוד הפרויקט.
- Structure summary (עשר שורות ראשונות) - טבלה מ-Part 1 של שי המציגה שם עמודה, dtype, מספר ערכים ייחודיים ואחוז חסרים.
- Raw data head(10) - עשר שורות ראשונות של `df` לאחר ניקוי בסיסי של שי.
- Features selected for the model - chips של שמות הפיצ'רים שעובד בחר.
- ענן מילים בתחתית הטאב.

#### `DATA_QUALITY_HTML` - טאב Data Quality

- Missing values report per column - רשת רב-עמודתית של אריחים קומפקטיים. כל אריח מציג שם משתנה ואחוז חסרים; צביעה לפי חומרה: אדום עבור 50% ומעלה, ענבר עבור 20% ומעלה, ירוק אחרת. הצורה הרב-עמודתית מתאימה ל-324 משתנים בפריסה קריאה בהרבה מטבלה של 324 שורות.
- Data dictionary (עשר שורות ראשונות) - טבלה של שי המרכזת שם משתנה, תיאור מה-codebook, החלטת ניקוי.
- Drop candidates (עשר שורות ראשונות).
- CBS codebook (30 שורות ראשונות) - לאחר preprocessing.

#### `DESCRIPTIVES_HTML` - טאב Descriptives

- לכל אחת משלוש הקבוצות ב-`eda_groups` של שי - התפלגות ה-`value_counts` של המשתנים בקבוצה, מוצגת כ-mini bar plot ב-HTML (עמודות אופקיות בנויות מ-`<div>` פשוט עם `width` דינמי).
- התוויות של הערכים המספריים מגיעות מהcodebook דרך פונקציית `_lookup_label(var, code)`, שמחזירה את `code_title` הרלוונטי מהקודבוק שתורגם לאנגלית בתא ה-preprocessing.
- בסוף הטאב - התפלגות של `financial_risk_score`.

#### `MODEL_PERF_HTML` - טאב Model Performance

- Baseline comparison table - טבלה של עובד המשווה בין המודלים.
- שישה גרפי ביצועים ראשונים מתוך `saved_figures`, מוטמעים כתמונות base64.

#### `MODEL_INTERP_HTML` - טאב Model Interpretation

- שישה גרפי SHAP הבאים מתוך `saved_figures`.

### קטעי קוד מרכזיים

```python
def _fig_to_img(fig, alt="figure", max_width_px=1000):
    """ממיר matplotlib.Figure ל-<img src="data:image/png;base64,..."> string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (f'<img src="data:image/png;base64,{b64}" alt="{alt}" '
            f'style="max-width:100%;width:{max_width_px}px;height:auto;'
            f'display:block;margin:12px auto;border:1px solid #e3e8ee;'
            f'border-radius:8px;background:#fff">')

def _df_to_html(df, max_rows=None, classes="datatable"):
    """המרת DataFrame ל-HTML table עם cap אופציונלי."""
    if max_rows is not None and len(df) > max_rows:
        note = f'<div class="table-note">Showing first {max_rows} of {len(df):,} rows</div>'
        html = df.head(max_rows).to_html(classes=classes, index=False,
                                          na_rep="", border=0, escape=True)
        return note + html
    return df.to_html(classes=classes, index=False, na_rep="", border=0, escape=True)

def _lookup_label(_var, _code):
    """שולף label באנגלית מה-codebook, עם fallback לקוד הגולמי אם אין."""
    try:
        _match = codebook.loc[
            (codebook["variavle_name"] == _var) & (codebook["code"] == _code),
            "code_title",
        ]
        if len(_match) and pd.notna(_match.iloc[0]):
            return str(_match.iloc[0]).strip()
    except Exception:
        pass
    return str(_code)
```

**נקודות טכניות מפורטות:**

- **הפרדה בין בניית תוכן לתבנית.** התא הזה בונה מחרוזות HTML בלבד. ההרכבה של מסמך ה-HTML המלא (עם CSS, JS, ו-Plotly.js) מתבצעת בתא הבא. ההפרדה שומרת על תחזוקיות: שינוי בתוכן טאב אינו דורש נגיעה בתבנית ה-HTML.
- **`escape=True` ב-`_df_to_html`.** pandas ברירת מחדל של `to_html` היא לא לברוח תווים מיוחדים. אם קודבוק מכיל תווים כמו `<` או `&`, `to_html` יכתוב אותם ישירות ל-HTML, מה שיכול להישבר את המסמך או להפוך לבעיית XSS (Cross-Site Scripting). `escape=True` ממיר `<` ל-`&lt;`, `&` ל-`&amp;` וכו'. אבטחה בסיסית.
- **`bbox_inches="tight"` ב-`fig.savefig`.** matplotlib ברירת מחדל שומר figure עם padding לבן מסביב לתוכן. `bbox_inches="tight"` מזרז את ה-padding למינימום נחוץ. אחרת יש שוליים לבנים מיותרים בתמונות של הדאשבורד.
- **`_lookup_label` הוא הגורם המרכזי בטיפול בתוויות שהיו בעברית.** הקוד של שי משתמש ב-`.value_counts()` על משתני הסקר, שמחזירה קודים מספריים (1, 2, 3). ה-preprocessing של הקודבוק (הוסף במחברת המאוחדת) תרגם את `code_title` לאנגלית גם בזיכרון וגם בקובץ; `_lookup_label(var, code)` מחזיר את התווית המתורגמת לכל שילוב של משתנה וקוד.

---

## תא 10 - קוד - HTML_TEMPLATE + הרכבת index.html

**מה התא עושה:** התא הגדול והקריטי ביותר של Part 3. בונה קובץ HTML עצמאי אחד (`index.html`) שכולל:

- כל ה-CSS של הדאשבורד (כ-800 שורות).
- ספריית Plotly.js המלאה מוטמעת (כ-3.5 MB) כדי שהאתר יעבוד אופליין.
- כל ה-JavaScript של השאלון, הדאשבורד, הטאבים, המפה, הסימולטור, ה-Gauge, וגרף הפרופיל.
- את כל תוצרי הטאבים (`OVERVIEW_HTML`, `DATA_QUALITY_HTML`, ...) שהתא הקודם בנה.
- את המפה שהתא של Folium בנה (`MAP_SRCDOC`).
- את השאלון (`QUIZ`) כמחרוזת JSON.

**מנגנון ה-Template:** התא בונה את `HTML_TEMPLATE` כמחרוזת Python רגילה עם placeholders - `__FIGS__`, `__QUIZ__`, `__MAP_SRCDOC__`, `__OVERVIEW__`, `__DATA_QUALITY__`, `__DESCRIPTIVES__`, `__MODEL_PERF__`, `__MODEL_INTERP__`. בסוף התא, שרשרת של `.replace(...)` מחליפה כל placeholder בערך הרלוונטי. Plotly.js נטען מ-CDN בתוך תגית `<script src=...>` בתוך התבנית עצמה, ולא מוטמע פנימה.

**קטע קוד מלא - שרשרת ההחלפה:**

```python
def js_safe(s):
    """Neutralize '</script>' inside JSON strings embedded in a <script> block."""
    return s.replace("</script>", "<\\/script>").replace("</Script>", "<\\/Script>")

figs_payload = {name: json.loads(fig.to_json()) for name, fig in FIGS.items()}

html = (HTML_TEMPLATE
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

**Plotly.js מגיע מ-CDN:** תגית `<script src="https://cdn.plot.ly/plotly-2.35.2.min.js">` בתבנית ה-HTML. הגודל הסופי של `index.html` הוא כ-2 MB במקום 5.5 MB שהיה כשספריית Plotly הוטמעה בפנים.

**נקודה חשובה על `js_safe`:** מלכודת ידועה ב-HTML embedding של JSON.

- ה-`QUIZ` וה-`MAP_SRCDOC` נכנסים ל-HTML כערך של `var` בתוך `<script>` block: `var QUIZ = {"text": "...", ...};`.
- אם ה-string הפנימי מכיל את הרצף `</script>`, הדפדפן מפרש זאת כסיום של בלוק ה-script ומתחיל לפרש את שאר התוכן כ-HTML. הקוד שאחריו נשבר.
- **הפתרון:** ב-JSON, `<\/script>` שקול ל-`</script>` (ה-`\` הוא escape שאין לו משמעות מיוחדת בהקשר של סלאש). אבל **הדפדפן, שסורק את ה-HTML**, מחפש רק את המחרוזת המדויקת `</script>` - לא `<\/script>`. אז אחרי `js_safe`, הדפדפן ממשיך לקרוא את המחרוזת כחלק מה-`<script>` block, וה-JavaScript מפרש את ה-JSON כרגיל.
- זו סוג של פרצה שקל לפספס - שאלה קלאסית של מרצה: "מה קורה אם ה-`code_title` הזה מכיל `</script>`?". התשובה: `js_safe` דואג לזה.

### הפונקציות המרכזיות ב-JavaScript של index.html

בנוסף לתוכן, מסמך ה-HTML מכיל בלוק סקריפט אחד גדול עם כל הלוגיקה. הפונקציות המרכזיות שלו:

#### `buildQuiz()` - בניית השאלון

```js
function buildQuiz() {
  var box = el("questions");                            // <div id="questions">
  QUIZ.forEach(function (q, i) {
    var d = document.createElement("div");
    d.className = "question";

    var t = document.createElement("div");
    t.className = "qtext";
    t.textContent = (i + 1) + ". " + q.text;            // "1. What is a typical..."
    d.appendChild(t);

    q.options.forEach(function (opt, oi) {
      var lab = document.createElement("label");
      lab.className = "opt";

      var inp = document.createElement("input");
      inp.type = "radio";
      inp.name = q.id;                                  // Q1, Q2, ... - קובץ קבוצה
      inp.value = oi;                                   // אינדקס האפשרות
      inp.addEventListener("change", updateProgress);
      lab.appendChild(inp);

      var sp = document.createElement("span");
      sp.textContent = opt;
      lab.appendChild(sp);
      d.appendChild(lab);
    });
    box.appendChild(d);
  });
  updateProgress();
}
```

**נקודות טכניות:**

- **`name=q.id`** על כל ה-radio buttons של השאלה הזאת. זו הדרך של HTML לקבץ radio buttons - ברירת מחדל של הדפדפן היא שרק אחד נבחר בכל קבוצה. אם היינו נותנים שם unique לכל radio, המשתמש היה יכול לבחור את כל האפשרויות.
- **`addEventListener("change", updateProgress)`** - כל שינוי בבחירה מפעיל את הפונקציה `updateProgress` שמעדכנת את הצג "Answered X/10". התגובה מיידית - הדפדפן לא מחכה שהמשתמש יסיים לבחור בכל השאלות.
- **DOM manipulation ידני.** אין framework (React/Vue) - הכל דרך `document.createElement` ו-`appendChild`. הבחירה הזאת מונעת תלות בעוד ~50 KB של framework, ומספיקה לצרכים של הדף.

#### `submitQuiz()` - שער הכניסה לדאשבורד

```js
function submitQuiz() {
  var a = currentAnswers(), missing = [];
  QUIZ.forEach(function (q, i) {
    if (a[q.id] === null) missing.push(i + 1);
  });
  var err = el("quiz-error");
  if (missing.length) {
    err.style.display = "block";
    err.innerHTML = "<b>Cannot enter the dashboard yet.</b> Please answer question" +
      (missing.length > 1 ? "s" : "") + " " + missing.join(", ") + ".";
    return;   // חזרה מיידית, לא נכנסים לדאשבורד
  }
  err.style.display = "none";
  userAnswers = a;
  var score = 0, ktot = 0;
  QUIZ.forEach(function (q) {
    if (q.kind === "knowledge") { ktot++; if (a[q.id] === q.correct) score++; }
  });
  window.userScore = score;      // גלובלי - buildGauge יקרא אותו
  window.userKtot = ktot;
  el("quiz-screen").style.display = "none";
  el("dashboard").style.display = "block";
  el("topbar-back").style.display = "inline-flex";
  buildKnowledgeTable(a);
  buildMap();
  rendered = {};                  // reset של מטמון הרינדור של גרפים
  showTab("overview");
  window.scrollTo(0, 0);
}
```

**נקודות טכניות:**

- **אף גרף אינו מרונדר לפני `submitQuiz` מוצלח.** `Plotly.newPlot` נקרא לראשונה בתוך `showTab("overview")` שבשורה הלפני אחרונה. זה מקטין את זמן הטעינה הראשוני של הדף - המשתמש רואה את השאלון כמעט מיד, ורק אחרי שהוא מגיש את התשובות מבוצע הרינדור של הגרפים.
- **`rendered = {}`** - מטמון (cache) שזוכר איזה גרפים כבר רונדרו. אם המשתמש חוזר לטאב שכבר ביקר בו, הגרף לא מרונדר שוב אלא רק resize.
- **חלוקה: ידע לעומת התנהגות.** רק שאלות ידע נחשבות ל-`score`; שאלות התנהגות אין להן "תשובה נכונה". `ktot` שומר את המספר הכולל של שאלות ידע (5).

#### `showTab(name)` - החלפת טאב + Lazy Rendering

```js
function showTab(name) {
  // 1. סימון של הכפתור הפעיל
  document.querySelectorAll(".tabbtn").forEach(function (b) {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  // 2. הצגה/הסתרה של הפאנלים
  document.querySelectorAll(".panel").forEach(function (p) {
    p.style.display = (p.id === "panel-" + name) ? "block" : "none";
  });
  // 3. Lazy render של גרפי Plotly של הטאב
  (TAB_FIGS[name] || []).forEach(function (fid) {
    if (!rendered[fid]) {
      if (fid === "fig-profile") { buildProfileChart(); }
      else if (fid === "fig-gauge") { buildGauge(); }
      else {
        var f = FIGS[fid];
        Plotly.newPlot(fid, f.data, f.layout,
                       { responsive: true, displaylogo: false });
      }
      rendered[fid] = true;
    } else {
      try { Plotly.Plots.resize(fid); } catch (e) {}
    }
  });
  // 4. טיפול מיוחד לטאבים שאינם Plotly
  if (name === "geography") { buildMap(); setTimeout(refreshMap, 80); }
  if (name === "young") { setTimeout(initSimulator, 80); }
}
```

**נקודות טכניות מפורטות:**

- **Lazy rendering - הסיבה.** `Plotly.newPlot` הוא איטי - כל גרף לוקח 50-150ms לרינדור ראשוני, ו-DOM operations יקרות. אם היינו מרנדרים את כל הגרפים בבת אחת בטעינת הדף, המשתמש היה מחכה כמה שניות עד שהדף מגיב. במקום זה, גרפים מרונדרים רק כשהטאב שלהם נפתח, וממטמנים ב-`rendered` כדי לא לרנדר שוב.
- **`Plotly.Plots.resize(fid)`** - כאשר טאב שכבר רונדר נפתח שוב, ייתכן שרוחב הפאנל השתנה (למשל, המשתמש שינה את גודל החלון). `resize` מודיע ל-Plotly לחשב מחדש את המידות של הגרף מבלי לרנדר אותו מחדש. זריז מאוד.
- **`setTimeout(..., 80)` - למה 80ms?** CSS `display: block` הוא סינכרוני, אך layout reflow של הדפדפן לוקח כמה מילישניות. Leaflet ו-Plotly קוראים ל-`getBoundingClientRect()` בזמן אתחול; אם עדיין הכל בגודל 0, הם מרנדרים תוכן קטן ואינם מתקנים אחר כך. `setTimeout(..., 80)` נותן לדפדפן זמן לחשב את הגודל. הערך 80 נבחר empirically - פחות מכך גורם ל-Leaflet לפעמים לרנדר לפני שהמסגרת מוכנה.

#### `buildMap()` + `refreshMap()` - הזרקת המפה ועדכון Leaflet

```js
function buildMap() {
  var box = el("fig-map");
  if (box && !box.querySelector("iframe")) {
    var iframe = document.createElement("iframe");
    iframe.setAttribute("srcdoc", MAP_SRCDOC);   // ה-HTML של Folium
    iframe.setAttribute("width", "100%");
    iframe.setAttribute("height", "560");
    box.appendChild(iframe);
  }
}

function refreshMap() {
  var iframe = document.querySelector('#fig-map iframe');
  if (!iframe || !iframe.contentWindow) return;

  var win = iframe.contentWindow;
  // Folium מייצר משתנה גלובלי בשם map_<hash> בתוך ה-iframe
  var mapKey = Object.keys(win).find(function (k) {
    return k.indexOf('map_') === 0;
  });
  if (mapKey && win[mapKey] && win[mapKey].invalidateSize) {
    win[mapKey].invalidateSize();  // Leaflet: "המסגרת שינתה גודל, חשב מחדש"
  }
}
```

**נקודות טכניות מפורטות:**

- **iframe srcdoc - הבידוד.** ה-iframe מקבל את ה-HTML של המפה כמחרוזת. הדפדפן מטפל בה כמו דף נפרד לחלוטין - CSS משלו, JavaScript משלו, window object משלו. זה מונע התנגשויות בין ה-CSS של Leaflet לבין ה-CSS של הדאשבורד (שני מרכיבים גדולים).
- **Same-Origin Policy.** ל-iframe עם srcdoc יש origin `null` (אין URL). זה מגביל את גישת ה-JavaScript של הדאשבורד לתוכן ה-iframe. אך `iframe.contentWindow` עדיין נגיש, כי `srcdoc` נחשב לאותו origin כמו הדף האב. זו הסיבה ש-`refreshMap` יכול לגשת ל-`iframe.contentWindow`.
- **המנגנון של `Object.keys(win).find(k => k.indexOf('map_') === 0)`.** Folium מייצר את המפה כמשתנה גלובלי ב-iframe עם שם דינמי, למשל `map_a1b2c3d4`. אני לא יודע מראש את השם המדויק, אז מסתובבים על כל המשתנים הגלובליים של ה-iframe ומחפשים את זה שמתחיל ב-`map_`. אז קוראים ל-`invalidateSize()` שהוא API סטנדרטי של Leaflet - הוא מבצע reflow ומחשב מחדש את גודל הטיילים.
- **הצורך ב-`invalidateSize`.** אם ה-iframe נטען כשהפאנל היה במצב `display: none`, `contentWindow` קיבל גודל 0×0. Leaflet מרנדר בגודל אפס וגם אחרי שהפאנל נראה - לא מנסה לתקן את עצמו. `invalidateSize` הוא הקריאה הרשמית של Leaflet לבצע reflow.

#### `buildGauge()` - Gauge חצי-עגול לציון

בונה SVG של Gauge חצי-עגול. SVG paths נכתבים ידנית עם commands של M (move), A (arc), ו-L (line):

```js
// דוגמה של path של חצי עיגול:
// M start_x,start_y A radius,radius 0 0 1 end_x,end_y
// (M) = move to start
// (A) = arc: rx,ry,rotation,large-arc,sweep,end_x,end_y

// שלוש קשתות: אדום (0-40%), כתום (40-70%), ירוק (70-100%)
var arc1 = `M ${x1},${y1} A ${r},${r} 0 0 1 ${x2},${y2}`; // אדום
```

הבחירה ב-SVG paths ידניים נובעת מכך ש-Plotly.js אינו מספק Gauge חצי-עגול מיושר. הפתרון: לצייר את הקשתות דרך `<path>` ולציין את המחט (מחט הכספית) כ-`<line>` שנסובבת בהתאם לציון.

#### `buildProfileChart()` - השוואת המשתמש לאוכלוסייה

בונה גרף Plotly של עמודות אופקיות. לכל שאלת התנהגות (Q3, Q4, Q6, Q7, Q9), עמודה אחת מציגה את התשובה של המשתמש (Yes=100, No=0), ועמודה שנייה מציגה את `popYes` של האוכלוסייה. המשתמש רואה מיידית באילו התנהגויות הוא מקדים את הממוצע ובאילו מפגר אחריו.

#### `buildKnowledgeTable(a)` - הטבלה של טאב Your Profile

```js
function buildKnowledgeTable(a) {
  var tbody = el("ktable").querySelector("tbody");
  tbody.innerHTML = "";                                 // ניקוי לפני מילוי מחדש
  QUIZ.forEach(function (q) {
    var tr = document.createElement("tr");
    var isSurvey = (q.kind === "behavior");

    // עמודה 1: תג צבע
    var tdType = document.createElement("td");
    var badge = document.createElement("span");
    badge.className = "qbadge " + (isSurvey ? "survey" : "knowl");
    badge.textContent = isSurvey ? "From social survey" : "Knowledge check";
    tdType.appendChild(badge);
    tr.appendChild(tdType);

    // עמודה 2: השאלה
    var tdQ = document.createElement("td");
    tdQ.textContent = q.text;
    tr.appendChild(tdQ);

    // עמודה 3: התשובה של המשתמש
    var tdYour = document.createElement("td");
    tdYour.textContent = q.options[a[q.id]];
    tr.appendChild(tdYour);

    // עמודה 4: הפניה (משתנה בין survey ל-knowledge)
    var tdRef = document.createElement("td");
    if (isSurvey) {
      tdRef.textContent = "Yes in population: " + q.popYes.toFixed(1) + "%";
    } else {
      tdRef.textContent = q.options[q.correct];
    }
    tr.appendChild(tdRef);

    // עמודה 5: תוצאה (רק לשאלות ידע)
    var tdRes = document.createElement("td");
    if (isSurvey) {
      tdRes.textContent = "-";
    } else {
      var ok = a[q.id] === q.correct;
      tdRes.textContent = ok ? "Correct" : "Wrong";
      tdRes.className = ok ? "ok" : "bad";
    }
    tr.appendChild(tdRes);

    tbody.appendChild(tr);
  });
}
```

**נקודות טכניות מפורטות:**

- **הפרדה ויזואלית בין סוגי שאלות.** תגי צבע ירוק (`qbadge survey`) לשאלות התנהגות הנמצאות בסקר הרשמי של הלמ"ס - יש להן משנה תוקף כי הן מבוססות על שאלה שנשאלה בסקר של 6,907 משיבים. תגי כחול (`qbadge knowl`) לשאלות ידע שהוספתי לצורך הבחינה של אוריינות פיננסית - אין להן backing סטטיסטי מהסקר. הצבע מבהיר למשתמש למה חלק מהתשובות מציגות `popYes` וחלק מציגות `Correct/Wrong`.
- **למה `tbody.innerHTML = ""`?** בפעם הראשונה שהמשתמש מגיש שאלון, ה-`tbody` ריק - אין מה לנקות. אבל אם המשתמש חוזר ל-quiz דרך כפתור Back ומגיש שוב עם תשובות שונות, ה-`tbody` יכיל את התוצאות הקודמות. הניקוי לפני מילוי מחדש מבטיח שאין שאריות של submission קודם.

#### `renderSim()` + `initSimulator()` - סימולטור ריבית דריבית

```js
function renderSim() {
  var startAge = +el("sim-start-age").value;    // '+' ממיר string ל-number
  var monthly  = +el("sim-monthly").value;
  var rate     = +el("sim-rate").value / 100;   // מ-6 ל-0.06
  var years    = 67 - startAge;                 // עד גיל פרישה 67
  var r        = rate / 12;                     // ריבית חודשית
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

**נקודה מתמטית - הנוסחה של Future Value של אנואיטי:**

הנוסחה `M * ((1+r)^n - 1) / r` מגיעה מסיכום סדרה גיאומטרית. אם מפקידים סכום `M` בסוף כל תקופה, במשך `n` תקופות, בריבית `r` לתקופה, אז:

- ההפקדה הראשונה מרוויחה ריבית ל-`n-1` תקופות → שווה `M * (1+r)^(n-1)`.
- ההפקדה השנייה מרוויחה ל-`n-2` תקופות → שווה `M * (1+r)^(n-2)`.
- ...
- ההפקדה האחרונה מרוויחה ל-0 תקופות → שווה `M`.

הסכום של סדרה גיאומטרית כזאת (התחלה `M`, יחס `(1+r)`, סה"כ `n` איברים) הוא:

```
Sum = M * ((1+r)^n - 1) / ((1+r) - 1) = M * ((1+r)^n - 1) / r
```

זו הנוסחה שמוציאה את התוצאה הפלאית - הפקדה של 500 ש"ח בחודש בין גיל 25 לגיל 67 בריבית שנתית 6% (0.5% חודשי):

```
n = 504 months
r = 0.005
future = 500 * ((1.005)^504 - 1) / 0.005 ≈ 500 * 12.1 / 0.005 ≈ 1,210,000
```

זו הנקודה החינוכית של הטאב לצעירים.

#### `backToQuiz()` - חזרה למסך השאלון

```js
function backToQuiz() {
  document.querySelectorAll("#questions input[type=radio]").forEach(function (r) {
    r.checked = false;                              // איפוס כל הבחירות
  });
  var err = el("quiz-error");
  if (err) { err.style.display = "none"; err.innerHTML = ""; }
  updateProgress();                                 // "Answered 0/10"
  el("dashboard").style.display = "none";
  el("quiz-screen").style.display = "block";
  el("topbar-back").style.display = "none";
  window.scrollTo(0, 0);
}
```

הפונקציה מאפסת את כל ה-radio buttons, מנקה שגיאה קודמת, מעדכנת את מונה ההתקדמות, מסתירה את הדאשבורד, מציגה שוב את מסך השאלון, ומסתירה את כפתור ה-Back מהטופ. הכפתור עצמו בטופ בנוי כך:

```html
<button id="topbar-back" class="home-btn" style="display:none" onclick="backToQuiz()">
  <svg>...</svg> Back to quiz
</button>
```

הכפתור מוסתר בטעינה (`style="display:none"`) ומופיע רק אחרי submitQuiz מוצלח.

**הערה חשובה על ה-cache של גרפים:** כאשר המשתמש חוזר לשאלון ומגיש שוב עם תשובות שונות, `rendered` צריך להתאפס - אחרת ה-Gauge יציג את הציון הישן. `submitQuiz` דואג לזה עם `rendered = {}`.

---

## תא 11 - קוד - שרת HTTP מקומי (Colab-safe)

**מה התא עושה:** מגיב באופן שונה לפי סביבת ההרצה:

- **מקומית (Jupyter):** מרים שרת פייתון פשוט (`http.server`) על 127.0.0.1 בפורט 8890, ופותח את `index.html` בדפדפן דרך הפורט הזה. השרת רץ ברקע ב-thread daemon.
- **Colab:** אין דפדפן ואין גישה ל-127.0.0.1 מהצד של המשתמש. במקום זאת התא קורא ל-`google.colab.files.download(OUT_HTML)`, שמפעיל הורדה של הקובץ לדפדפן של המשתמש.

**קטע קוד:**

```python
import http.server, socketserver, threading

def start_local_server(port=8890, tries=5):
    for p in range(port, port + tries):
        try:
            httpd = socketserver.TCPServer(
                ("127.0.0.1", p),
                http.server.SimpleHTTPRequestHandler,
            )
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            return httpd, p
        except OSError:
            continue
    return None, None

if IN_COLAB:
    from google.colab import files
    files.download(OUT_HTML)
else:
    _LOCAL_SERVER, _LOCAL_PORT = start_local_server()
    if _LOCAL_PORT:
        LOCAL_URL = f"http://127.0.0.1:{_LOCAL_PORT}/index.html"
        webbrowser.open(LOCAL_URL)
```

**נקודות טכניות מפורטות:**

- **הצורך בשרת מקומי - Same-Origin Policy.** כשפותחים קובץ HTML ישירות דרך `file://c:/...`, הדפדפן מפעיל מדיניות אבטחה יותר מגבילה. `iframe srcdoc` לפעמים לא נטען. `http://127.0.0.1` הוא origin רגיל שהדפדפן מטפל בו כמו כל אתר, וכל התכונות עובדות. בנוסף - סביבת GitHub Pages תמיד תגיש דרך `https://`, אז יש היגיון לפתח דרך `http://` שיותר קרוב.
- **`daemon=True` על ה-thread.** ב-Python, thread ברירת מחדל הוא לא daemon. `daemon=True` דואג לכך שכשהמחברת נסגרת, השרת נסגר איתו ולא נשאר תהליך רץ.
- **חמישה ניסיונות פורט.** אם 8890 תפוס, מנסים 8891 עד 8894. אם כל אחד תפוס - עוזבים.

---

## הקשר Cross-part: התלויות ב-Shai וב-Ovad

Part 3 מסתמך על תוצרים של Parts 1 ו-2 בשני מסלולים מקבילים:

1. **In-memory (במצב Restart & Run All):** משתני זיכרון גלובליים - `raw`, `structure_summary`, `data_dictionary`, `eda_groups`, `codebook`, `model_data`, `features`, `cv_results`, `saved_figures`, `best_model`.

2. **קבצי handoff על הדיסק (למצב Part-3-only):**
   - `model data.csv` - הדאטה הנקי של שי.
   - `xgb_model.pkl` - ה-pipeline המאומן של עובד.
   - `shai_state.pkl` - dict של כל תוצרי הזיכרון של שי.
   - `ovad_state.pkl` - dict של כל תוצרי הזיכרון של עובד.

בסביבת Colab, תא ה-Handoff הראשון של Part 3 (תא 1 לעיל) בודק אם ה-globals ריקים; במקרה כזה הוא מציע להעלות את שלושת הקבצים ופורק אותם ל-`globals()`.

---

## סיכום

- המחברת המאוחדת מורכבת משלושה חלקים הרצים ברצף. הזרימה עוברת גם בזיכרון (משתנים גלובליים) וגם בקבצי handoff לתמיכה גם ב-Part-3-only בסביבת Colab.
- הדאשבורד הוא קובץ HTML יחיד ועצמאי (`index.html`) - Plotly.js מוטמע (כ-3.5 MB), כל הגרפים embedded, כל הטבלאות embedded, המפה כ-iframe srcdoc, ענן המילים כ-base64. אין שרת ייצור.
- השאלון חוסם את הכניסה לדאשבורד - אף גרף אינו מרונדר לפני `submitQuiz()` מוצלח. `Plotly.newPlot` מופעל בעצלתיים per-tab.
- שמונה טאבים: Overview (עם word cloud בתחתית), Data Quality, Descriptives, Model Performance, Model Interpretation, Young Generation, Geography (מפת Folium), Your Profile.
- טאב Your Profile מציג את כל עשר השאלות עם תגי צבע.
- תמיכה מלאה ב-Colab: תא upload prompt לקבצי הקלט, `try/except` על `drive.mount`, הורדה אוטומטית של `index.html` במקום שרת מקומי.

---

## שאלות ותשובות שהמרצה עשויה לשאול

### דאטה

**שאלה: מדוע קודי `888888` ו-`999999` הומרו ל-NaN במקום להסיר את השורות?**
תשובה: שני הקודים מייצגים סוגים שונים של חסרים ב-CBS: `888888` = "לא יודע/מסרב לענות", `999999` = "לא רלוונטי/סינון" (למשל, שאלות שכר לא רלוונטיות למי שאינו עובד). מחיקת שורות עם קודים אלה הייתה מוחקת חצי מהמשיבים ומטה את הדאטה לטובת אוכלוסייה מסוימת. שי בחר להמיר ל-NaN כדי לשמור את השורה ולתעד את החסר; ה-imputer של Ovad ב-pipeline (`SimpleImputer(strategy="median")`) מטפל בכל fold של ה-CV בנפרד - כלומר, אין data leakage.

**שאלה: מדוע הסף ל-financial_risk_score דורש 5 מתוך 8 אינדיקטורים תקינים?**
תשובה: 8 האינדיקטורים הם: כיסוי הוצאות, ויתור על טיפול רפואי, ויתור על תרופות, ויתור על טיפול שיניים, ועוד. אם הסף היה נמוך מדי (למשל 3), משיבים עם רק 3 תשובות תקינות היו מקבלים ציון שאינו יציב סטטיסטית. סף 5 שומר על מספיק אינדיקטורים בכל תצפית כדי שהציון יהיה סמן אמין למצוקה כלכלית. הסף בכל מקרה משאיר 6,907 משיבים.

**שאלה: מדוע Phi-K ולא Pearson correlation?**
תשובה: Pearson מודד רק קשרים ליניאריים בין משתנים רציפים. הפיצ'רים כאן הם ברובם קטגוריאליים (סקאלת Likert 1-4, מדד בינארי Yes/No, בקבוצות של גיל). Phi-K עובד על כל שילוב של סוגי משתנים (רציף-רציף, קטגוריאלי-קטגוריאלי, מעורב) וגם על קשרים לא-ליניאריים.

### מודל

**שאלה: מדוע נבחר סף החלטה 0.3 ולא 0.5?**
תשובה: בעיה של סיכון פיננסי היא asymmetric - false negative (מישהו בסיכון שהמודל אינו מזהה) יקר יותר מ-false positive (מישהו שאינו בסיכון והמודל מסמן). סף 0.3 מטה את המודל להיות רגיש יותר לזיהוי מקרים חיוביים - recall גבוה על חשבון precision. הסף נבחר יחד עם ה-GridSearchCV כחלק ממרחב החיפוש, לא ידנית.

**שאלה: מה GMM מוסיף על גבי הפיצ'רים המקוריים?**
תשובה: GMM מוצא clusters של Gaussian ב-feature space; `predict_proba` מחזיר לכל דגימה את ההסתברות להשתייך לכל cluster. הפיצ'רים החדשים מייצגים "לאיזה סוג פרופיל אוכלוסייה המשיב שייך". LogReg למשל יכול ללמוד קווים ליניאריים בלבד; אם קיים interaction לא-ליניארי בין פיצ'רים (למשל, נשים בגיל 30-40 עם ילדים קטנים), ה-GMM עשוי להוציא אותו כ-cluster ולתת ל-LogReg gauge לתרגם אותו לחיזוי. ה-GridSearchCV מודד עם/בלי GMM וקובע האם התוספת אכן משפרת F1.

**שאלה: מדוע F1 כמדד ה-scoring של GridSearch ולא accuracy?**
תשובה: F1 מאזן בין precision ל-recall. עבור dataset עם class imbalance (רוב לא בסיכון), accuracy מטעה - מודל שאומר "לא בסיכון" לכולם יקבל accuracy גבוה אך F1 = 0 עבור הקבוצה החיובית. F1 מעניש מודל שמפספס את הקבוצה החיובית.

**שאלה: מה SHAP מראה שרשימת ה-`feature_importances_` של XGBoost לא מראה?**
תשובה: `feature_importances_` הוא מדד גלובלי אחד - כמה כל פיצ'ר תרם למודל בממוצע. SHAP הוא לוקאלי: לכל תצפית, מראה כמה כל פיצ'ר תרם *לאותה חיזוי הספציפי*. ה-beeswarm של SHAP מציג את ההתפלגות של התרומות בין המשיבים - פיצ'ר יכול להיות "חשוב" גלובלית אך להשפיע רק על תת-קבוצה מסוימת. SHAP גם משמר additivity - סכום התרומות של כל הפיצ'רים = הפרש בין החיזוי לבין baseline.

### Latency ו-Performance

**שאלה: למה Plotly.js מוטמע בקובץ (~3.5 MB) ולא נטען מ-CDN?**
תשובה: החלטה תלוית context. יתרונות ההטמעה: (1) הדאשבורד עובד אופליין לחלוטין, גם ב-plane או מקום בלי אינטרנט, (2) אין תלות בבריאות של ה-CDN - אם unpkg נופל, ה-CDN נופל וה-dashboard לא, (3) אין FOUC (Flash of Unstyled Content) בזמן ההמתנה לטעינת ה-JS. חסרונות: (1) הקובץ 5.5 MB במקום 2 MB, (2) המשתמש לא יוצר cache של Plotly.js בכל אתר בנפרד. עבור מטרת הפרויקט (הצגה במחשב מקומי / GitHub Pages), היתרון של אופליין חשוב יותר.

**שאלה: כמה זמן לוקח לטעון את הדאשבורד?**
תשובה: על חיבור אינטרנט מהיר (100 Mbps), 5.5 MB נטענים בכ-0.5 שניות. ה-DOM נבנה תוך פחות משנייה, וה-Plotly render של הגרף הראשון (Overview) הוא lazy - קורה רק אחרי submitQuiz. סה"כ Time-To-Interactive של השאלון: פחות משנייה.

**שאלה: מה ההיגיון של Lazy rendering?**
תשובה: 12 גרפי Plotly מרונדרים לוקח 1-2 שניות (ברמת CPU). אם מרנדרים את כולם בטעינה, המשתמש רואה מסך לבן/תקוע 1-2 שניות. במקום - מרנדרים רק את הגרפים של הטאב הפעיל. `Plotly.newPlot` נקרא רק כשהמשתמש לוחץ על טאב, ו-`rendered = {}` שומר cache - אם חוזרים לטאב, קוראים ל-`Plotly.Plots.resize` (זריז ~10ms) במקום `newPlot`.

**שאלה: למה השאלון חוסם את הכניסה לדאשבורד?**
תשובה: שני יעדים: (1) UX - המשתמש מקבל *ציון אישי* וגרף השוואה לאוכלוסייה. אם הדאשבורד היה פתוח מיד, אין דרך לבנות את זה. (2) Performance - כפי שנאמר, `Plotly.newPlot` יקר. השאלון "רוכש" את הזמן שה-render לוקח: המשתמש עסוק בעניית שאלות בזמן שה-JavaScript מוכן לרנדר (כרגע רק ה-QUIZ ו-FIGS מוגדרים ב-memory; ה-Plotly.newPlot יופעל רק בהמשך).

### Trade-offs

**שאלה: למה קובץ HTML יחיד ולא app של React/Streamlit?**
תשובה: הפרויקט מוגדר כ-Deployment של EDA + מודל. יחסי עלות-תועלת: React app היה דורש npm install, transpiler (Babel), bundler (webpack), CI/CD, hosting. Streamlit היה דורש שרת Python חי. קובץ HTML סטטי אחד - GitHub Pages מגיש חינם, בלי backend, בלי תחזוקה. חסרון: אין refresh של נתונים ללא build מחדש של המחברת. עבור פרויקט מיד-סטודנטיאלי שמוצג פעם אחת בכיתה, זו trade-off נכונה.

**שאלה: למה handoff קבצים ולא רק זיכרון?**
תשובה: זרם Restart & Run All לוקח 3-5 דקות (בגלל GridSearchCV + SHAP). אם מישהו רוצה רק לשנות את הדאשבורד ולראות תוצאה מיידית, לא סביר שיריץ 5 דקות של אימון מודל בכל פעם. `shai_state.pkl` + `ovad_state.pkl` מאפשרים "restart Part 3" תוך שניות. גם חשוב לתמיכה ב-Colab: סשן Colab מוגבל ל-12 שעות זיכרון; אם הסשן קורס באמצע, החזרה מ-Part 3 בלבד היא חיסכון עצום.

### שאלות "מטריף" קלאסיות של מרצה

**שאלה: מה יקרה אם ה-`code_title` בקודבוק מכיל `</script>`?**
תשובה: ה-`js_safe` מטפל בזה. הוא מחליף `</script>` ל-`<\/script>` (ה-backslash escape ב-JSON חוקי אבל הדפדפן לא רואה את זה כסוגר script). זו הגנה בסיסית נגד XSS + נגד תוכן שמפרק את ה-HTML.

**שאלה: מה קורה אם המפה נטענת בזמן שהפאנל מוסתר?**
תשובה: `iframe.contentWindow` מקבל גודל 0×0, Leaflet מרנדר בגודל 0 ולא מנסה לתקן. `refreshMap` מטפל בזה: מוצא את משתנה `map_<hash>` בתוך ה-iframe ומפעיל `invalidateSize()` - API רשמי של Leaflet לחישוב מחדש של הגודל. ה-`setTimeout(80ms)` נותן לדפדפן להשלים את ה-layout reflow לפני ה-invalidate.

**שאלה: יש data leakage בין train ל-test?**
תשובה: לא. הפיצול נעשה עם `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` לפני כל טרנספורמציה. ה-`SimpleImputer`, `StandardScaler`, ו-`GMM` נמצאים כולם בתוך `Pipeline`, ו-`GridSearchCV` מפעיל את ה-pipeline בכל fold בנפרד - הם לומדים median, mean, std, ו-clusters מה-train fold בלבד ומיישמים על ה-validation fold.

</div>
