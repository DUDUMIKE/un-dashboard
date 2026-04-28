"""
================================================================================
  INTERACTIVE DASHBOARD
  Theme  : United Nations Chemicals, Waste & Energy Management
  Name   : [Your Name]
  Class  : TY Computer Science (Final Semester)
  Tool   : Python + Plotly Dash
================================================================================

HOW TO RUN:
-----------
1. Install dependencies (run once in Command Prompt):
       pip install plotly dash pandas numpy scikit-learn

2. Place this file in your DS_Assignment folder (with your CSV files)

3. Open Command Prompt → navigate to your folder:
       cd Desktop\DS_Assignment

4. Run:
       python dashboard.py

5. Open your browser and go to:
       http://127.0.0.1:8050

================================================================================
"""

# ==============================================================================
#  IMPORTS
# ==============================================================================
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output, callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)


# ==============================================================================
#  DATA LOADING — uses real CSVs if present, simulated data as fallback
# ==============================================================================
def safe_load(filepath, backup):
    if os.path.exists(filepath):
        print(f"  ✅ Loaded: {filepath}")
        return pd.read_csv(filepath)
    print(f"  ⚠️  Not found: {filepath} — using simulated data")
    return backup


# ── Water Quality ─────────────────────────────────────────────────────────────
backup_water = pd.DataFrame({
    "ph"             : np.random.uniform(4, 10, 1000).round(2),
    "Hardness"       : np.random.uniform(47, 323, 1000).round(2),
    "Solids"         : np.random.uniform(320, 61227, 1000).round(2),
    "Chloramines"    : np.random.uniform(0.35, 13, 1000).round(2),
    "Sulfate"        : np.random.uniform(129, 481, 1000).round(2),
    "Conductivity"   : np.random.uniform(181, 753, 1000).round(2),
    "Organic_carbon" : np.random.uniform(2, 28, 1000).round(2),
    "Trihalomethanes": np.random.uniform(0.74, 124, 1000).round(2),
    "Turbidity"      : np.random.uniform(1.45, 6.9, 1000).round(2),
    "Potability"     : np.random.randint(0, 2, 1000)
})
df_water = safe_load("water_potability.csv", backup_water)
df_water.fillna(df_water.median(numeric_only=True), inplace=True)
df_water.drop_duplicates(inplace=True)
target_col = "Potability" if "Potability" in df_water.columns else df_water.columns[-1]

# ── Air Quality ───────────────────────────────────────────────────────────────
backup_air = pd.DataFrame({
    "City"      : np.random.choice(["Delhi","Mumbai","Chennai","Kolkata","Bangalore"], 800),
    "Date"      : pd.date_range("2019-01-01", periods=800, freq="D").strftime("%Y-%m-%d"),
    "PM2.5"     : np.random.uniform(10, 300, 800).round(2),
    "PM10"      : np.random.uniform(20, 400, 800).round(2),
    "NO2"       : np.random.uniform(5, 120, 800).round(2),
    "AQI"       : np.random.randint(30, 500, 800),
    "AQI_Bucket": np.random.choice(["Good","Moderate","Poor","Very Poor","Severe"], 800)
})
df_air = safe_load("city_day.csv", backup_air)
useful = [c for c in ["City","Date","PM2.5","PM10","NO2","AQI","AQI_Bucket"]
          if c in df_air.columns]
df_air = df_air[useful].copy()
for col in df_air.select_dtypes(include=np.number).columns:
    df_air[col].fillna(df_air[col].median(), inplace=True)
df_air["Date"] = pd.to_datetime(df_air["Date"], errors="coerce")
df_air.dropna(subset=["Date"], inplace=True)
df_air["Year"]  = df_air["Date"].dt.year
df_air["Month"] = df_air["Date"].dt.month

# ── Plastic Waste ─────────────────────────────────────────────────────────────
backup_plastic = pd.DataFrame({
    "Entity": ["India","USA","China","Brazil","Germany",
                "Nigeria","Japan","UK","France","Australia",
                "Canada","Russia","Mexico","Indonesia","Turkey"],
    "Year"  : [2019]*15,
    "Per capita plastic waste (kg/person/day)":
        np.random.uniform(0.05, 0.80, 15).round(3)
})
df_plastic = safe_load("plastic-waste-per-capita.csv", backup_plastic)
df_plastic.drop_duplicates(inplace=True)
df_plastic.fillna(df_plastic.median(numeric_only=True), inplace=True)

# ── Nuclear / Energy ──────────────────────────────────────────────────────────
backup_nuclear = pd.DataFrame({
    "Year"         : list(range(2000, 2024)),
    "Nuclear_GWh"  : np.random.uniform(2400, 2800, 24).round(1),
    "Renewable_GWh": np.random.uniform(1000, 8000, 24).round(1),
    "Fossil_GWh"   : np.random.uniform(14000, 22000, 24).round(1),
})
df_nuclear = safe_load("nuclear_energy_overview_eia.csv", backup_nuclear)
df_nuclear.fillna(df_nuclear.median(numeric_only=True), inplace=True)

# Death rates (hardcoded — reliable reference values)
df_death = pd.DataFrame({
    "Source"        : ["Coal","Oil","Natural Gas","Biomass","Hydro",
                       "Nuclear","Wind","Solar"],
    "Deaths_per_TWh": [24.62, 18.43, 2.82, 4.63, 1.30, 0.07, 0.04, 0.02]
})

# ── CO2 Emissions ─────────────────────────────────────────────────────────────
backup_co2 = pd.DataFrame({
    "Country or Area": ["India","USA","China","Brazil","Germany",
                        "Nigeria","Japan","UK","France","Australia",
                        "Canada","Russia","Mexico","Indonesia","Turkey"],
    "Year"           : [2019]*15,
    "Value"          : np.random.uniform(100, 10000, 15).round(1),
    "Series"         : ["Emissions (thousand metric tons of carbon dioxide)"]*15
})
df_co2 = safe_load("Carbon Dioxide Emission Estimates.csv", backup_co2)
df_co2.drop_duplicates(inplace=True)
df_co2.fillna(0, inplace=True)


# ==============================================================================
#  ML MODELS — train once at startup
# ==============================================================================
print("\n  Training ML models...")
feature_cols = [c for c in df_water.select_dtypes(include=np.number).columns
                if c != target_col]
X = df_water[feature_cols]
y = df_water[target_col]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm  = svm_model.predict(X_test)
svm_acc     = accuracy_score(y_test, y_pred_svm) * 100
svm_cm      = confusion_matrix(y_test, y_pred_svm)
svm_report  = classification_report(y_test, y_pred_svm,
                                     target_names=["Unsafe","Safe"],
                                     output_dict=True)

dt_model = DecisionTreeClassifier(max_depth=6, random_state=42,
                                   criterion="gini", min_samples_split=10)
dt_model.fit(X_train, y_train)
y_pred_dt  = dt_model.predict(X_test)
dt_acc     = accuracy_score(y_test, y_pred_dt) * 100
dt_cm      = confusion_matrix(y_test, y_pred_dt)
dt_report  = classification_report(y_test, y_pred_dt,
                                    target_names=["Unsafe","Safe"],
                                    output_dict=True)
feat_imp   = pd.Series(dt_model.feature_importances_,
                        index=feature_cols).sort_values(ascending=True)

print(f"  ✅ SVM Accuracy          : {svm_acc:.2f}%")
print(f"  ✅ Decision Tree Accuracy: {dt_acc:.2f}%")


# ==============================================================================
#  COLOUR PALETTE
# ==============================================================================
UN_BLUE   = "#009EDB"
UN_DARK   = "#1a1a2e"
UN_CARD   = "#16213e"
UN_ACCENT = "#0f3460"
TEXT_COL  = "#e0e0e0"
GREEN     = "#00b894"
RED       = "#d63031"
ORANGE    = "#e17055"

TAB_STYLE = {
    "backgroundColor": UN_DARK,
    "color"          : TEXT_COL,
    "border"         : "none",
    "padding"        : "10px 20px",
    "fontSize"       : "14px",
    "fontWeight"     : "500",
}
TAB_SELECTED = {
    **TAB_STYLE,
    "backgroundColor": UN_BLUE,
    "color"          : "white",
    "borderRadius"   : "6px 6px 0 0",
    "fontWeight"     : "700",
}
CARD_STYLE = {
    "backgroundColor": UN_CARD,
    "borderRadius"   : "12px",
    "padding"        : "20px",
    "marginBottom"   : "20px",
    "boxShadow"      : "0 4px 15px rgba(0,0,0,0.3)",
}
METRIC_STYLE = {
    "backgroundColor": UN_ACCENT,
    "borderRadius"   : "10px",
    "padding"        : "15px 20px",
    "textAlign"      : "center",
    "flex"           : "1",
    "margin"         : "0 8px",
}


# ==============================================================================
#  HELPER — Plotly figure base layout
# ==============================================================================
def base_layout(title=""):
    return dict(
        title       = dict(text=title, font=dict(color=TEXT_COL, size=14)),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(color=TEXT_COL, size=11),
        margin        = dict(l=40, r=20, t=50, b=40),
        legend        = dict(bgcolor="rgba(0,0,0,0.3)",
                             bordercolor="gray", borderwidth=1),
        xaxis = dict(gridcolor="rgba(255,255,255,0.08)",
                     zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis = dict(gridcolor="rgba(255,255,255,0.08)",
                     zerolinecolor="rgba(255,255,255,0.1)"),
    )


# ==============================================================================
#  BUILD DASH APP
# ==============================================================================
app = Dash(__name__, title="UN Waste & Energy Dashboard")

app.layout = html.Div(style={
    "backgroundColor": UN_DARK,
    "minHeight"      : "100vh",
    "fontFamily"     : "'Segoe UI', Arial, sans-serif",
    "color"          : TEXT_COL,
}, children=[

    # ── HEADER ────────────────────────────────────────────────────────────────
    html.Div(style={
        "background"  : f"linear-gradient(135deg, {UN_ACCENT}, {UN_BLUE})",
        "padding"     : "28px 40px",
        "marginBottom": "0px",
        "boxShadow"   : "0 4px 20px rgba(0,158,219,0.3)",
    }, children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"16px"}, children=[
            html.Div("🌍", style={"fontSize":"42px"}),
            html.Div([
                html.H1("UN Chemicals & Waste Management",
                        style={"margin":"0","fontSize":"26px",
                               "fontWeight":"800","color":"white"}),
                html.P("Data Science Dashboard — TY Computer Science Final Project",
                       style={"margin":"4px 0 0","color":"rgba(255,255,255,0.8)",
                              "fontSize":"13px"}),
            ])
        ]),

        # KPI METRICS ROW
        html.Div(style={"display":"flex","marginTop":"20px","gap":"0"}, children=[
            html.Div([
                html.Div(f"{len(df_water):,}", style={"fontSize":"24px",
                          "fontWeight":"800","color":UN_BLUE}),
                html.Div("Water Samples", style={"fontSize":"11px",
                          "color":"rgba(255,255,255,0.7)"}),
            ], style=METRIC_STYLE),
            html.Div([
                html.Div(f"{len(df_air):,}", style={"fontSize":"24px",
                          "fontWeight":"800","color":GREEN}),
                html.Div("Air Quality Records", style={"fontSize":"11px",
                          "color":"rgba(255,255,255,0.7)"}),
            ], style=METRIC_STYLE),
            html.Div([
                html.Div(f"{svm_acc:.1f}%", style={"fontSize":"24px",
                          "fontWeight":"800","color":ORANGE}),
                html.Div("SVM Accuracy", style={"fontSize":"11px",
                          "color":"rgba(255,255,255,0.7)"}),
            ], style=METRIC_STYLE),
            html.Div([
                html.Div(f"{dt_acc:.1f}%", style={"fontSize":"24px",
                          "fontWeight":"800","color":"#a29bfe"}),
                html.Div("Decision Tree Accuracy", style={"fontSize":"11px",
                          "color":"rgba(255,255,255,0.7)"}),
            ], style=METRIC_STYLE),
            html.Div([
                html.Div("5", style={"fontSize":"24px",
                          "fontWeight":"800","color":"#fd79a8"}),
                html.Div("Datasets Used", style={"fontSize":"11px",
                          "color":"rgba(255,255,255,0.7)"}),
            ], style=METRIC_STYLE),
        ]),
    ]),

    # ── TABS ──────────────────────────────────────────────────────────────────
    dcc.Tabs(id="main-tabs", value="tab-air",
             style={"backgroundColor":UN_DARK,"borderBottom":f"2px solid {UN_BLUE}"},
             children=[

        # ════════════════════════════════════════════════════════
        #  TAB 1 — AIR QUALITY
        # ════════════════════════════════════════════════════════
        dcc.Tab(label="💨 Air Quality", value="tab-air",
                style=TAB_STYLE, selected_style=TAB_SELECTED,
                children=[html.Div(style={"padding":"24px 32px"}, children=[

            html.Div(style={"display":"flex","gap":"16px","marginBottom":"16px",
                            "flexWrap":"wrap"}, children=[
                html.Div([
                    html.Label("Select City", style={"fontSize":"12px",
                                "color":TEXT_COL,"marginBottom":"4px"}),
                    dcc.Dropdown(
                        id="city-dropdown",
                        options=[{"label":c,"value":c}
                                 for c in sorted(df_air["City"].dropna().unique())],
                        value=sorted(df_air["City"].dropna().unique())[:3],
                        multi=True,
                        style={"backgroundColor":UN_CARD,"color":"black",
                               "border":f"1px solid {UN_BLUE}","borderRadius":"8px"},
                    )
                ], style={"flex":"2","minWidth":"250px"}),

                html.Div([
                    html.Label("Select Metric", style={"fontSize":"12px",
                                "color":TEXT_COL,"marginBottom":"4px"}),
                    dcc.RadioItems(
                        id="air-metric",
                        options=[{"label":c,"value":c}
                                 for c in ["AQI","PM2.5","PM10","NO2"]
                                 if c in df_air.columns],
                        value="AQI",
                        inline=True,
                        style={"color":TEXT_COL},
                        labelStyle={"marginRight":"16px","fontSize":"13px"},
                    )
                ], style={"flex":"1","minWidth":"200px",
                          "display":"flex","flexDirection":"column",
                          "justifyContent":"center"}),
            ]),

            html.Div(style={"display":"grid",
                            "gridTemplateColumns":"2fr 1fr",
                            "gap":"16px"}, children=[
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="air-trend-line",
                              style={"height":"340px"})]),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="aqi-pie",
                              style={"height":"340px"})]),
            ]),

            html.Div(style={"display":"grid",
                            "gridTemplateColumns":"1fr 1fr",
                            "gap":"16px"}, children=[
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="air-box",
                              style={"height":"300px"})]),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="air-hist",
                              style={"height":"300px"})]),
            ]),
        ])]),

        # ════════════════════════════════════════════════════════
        #  TAB 2 — WATER QUALITY
        # ════════════════════════════════════════════════════════
        dcc.Tab(label="💧 Water Quality", value="tab-water",
                style=TAB_STYLE, selected_style=TAB_SELECTED,
                children=[html.Div(style={"padding":"24px 32px"}, children=[

            html.Div(style={"display":"flex","gap":"16px",
                            "marginBottom":"16px","flexWrap":"wrap"}, children=[
                html.Div([
                    html.Label("X Axis Feature", style={"fontSize":"12px",
                                "color":TEXT_COL}),
                    dcc.Dropdown(id="water-x",
                        options=[{"label":c,"value":c} for c in feature_cols],
                        value="ph",
                        style={"backgroundColor":UN_CARD,"color":"black",
                               "borderRadius":"8px"}),
                ], style={"flex":"1","minWidth":"180px"}),
                html.Div([
                    html.Label("Y Axis Feature", style={"fontSize":"12px",
                                "color":TEXT_COL}),
                    dcc.Dropdown(id="water-y",
                        options=[{"label":c,"value":c} for c in feature_cols],
                        value="Turbidity",
                        style={"backgroundColor":UN_CARD,"color":"black",
                               "borderRadius":"8px"}),
                ], style={"flex":"1","minWidth":"180px"}),
            ]),

            html.Div(style={"display":"grid",
                            "gridTemplateColumns":"1fr 1fr",
                            "gap":"16px"}, children=[
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="water-scatter",
                              style={"height":"340px"})]),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="water-donut",
                              style={"height":"340px"})]),
            ]),

            html.Div(style=CARD_STYLE, children=[
                dcc.Graph(id="water-heatmap",
                          style={"height":"340px"})]),
        ])]),

        # ════════════════════════════════════════════════════════
        #  TAB 3 — PLASTIC WASTE
        # ════════════════════════════════════════════════════════
        dcc.Tab(label="🧴 Plastic Waste", value="tab-plastic",
                style=TAB_STYLE, selected_style=TAB_SELECTED,
                children=[html.Div(style={"padding":"24px 32px"}, children=[

            html.Div(style={"display":"grid",
                            "gridTemplateColumns":"1fr 1fr",
                            "gap":"16px"}, children=[
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="plastic-bar",
                              style={"height":"380px"})]),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="plastic-treemap",
                              style={"height":"380px"})]),
            ]),

            html.Div(style=CARD_STYLE, children=[
                dcc.Graph(id="plastic-choropleth",
                          style={"height":"380px"})]),
        ])]),

        # ════════════════════════════════════════════════════════
        #  TAB 4 — CO2 EMISSIONS
        # ════════════════════════════════════════════════════════
        dcc.Tab(label="🌫️ CO₂ Emissions", value="tab-co2",
                style=TAB_STYLE, selected_style=TAB_SELECTED,
                children=[html.Div(style={"padding":"24px 32px"}, children=[

            html.Div(style={"display":"grid",
                            "gridTemplateColumns":"1fr 1fr",
                            "gap":"16px"}, children=[
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="co2-bar",
                              style={"height":"380px"})]),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="co2-choropleth",
                              style={"height":"380px"})]),
            ]),
        ])]),

        # ════════════════════════════════════════════════════════
        #  TAB 5 — NUCLEAR & ENERGY
        # ════════════════════════════════════════════════════════
        dcc.Tab(label="⚛️ Nuclear & Energy", value="tab-nuclear",
                style=TAB_STYLE, selected_style=TAB_SELECTED,
                children=[html.Div(style={"padding":"24px 32px"}, children=[

            html.Div(style={"display":"grid",
                            "gridTemplateColumns":"2fr 1fr",
                            "gap":"16px"}, children=[
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="energy-line",
                              style={"height":"360px"})]),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="death-bar",
                              style={"height":"360px"})]),
            ]),
        ])]),

        # ════════════════════════════════════════════════════════
        #  TAB 6 — ML RESULTS
        # ════════════════════════════════════════════════════════
        dcc.Tab(label="🤖 ML Models", value="tab-ml",
                style=TAB_STYLE, selected_style=TAB_SELECTED,
                children=[html.Div(style={"padding":"24px 32px"}, children=[

            # Accuracy cards
            html.Div(style={"display":"flex","gap":"16px",
                            "marginBottom":"16px"}, children=[
                html.Div(style={**CARD_STYLE,
                                "textAlign":"center","flex":"1",
                                "border":f"2px solid {UN_BLUE}"}, children=[
                    html.Div("🔵 SVM (RBF Kernel)",
                             style={"fontSize":"16px","fontWeight":"700",
                                    "marginBottom":"8px"}),
                    html.Div(f"{svm_acc:.2f}%",
                             style={"fontSize":"40px","fontWeight":"900",
                                    "color":UN_BLUE}),
                    html.Div("Accuracy on Test Set",
                             style={"color":"gray","fontSize":"12px"}),
                ]),
                html.Div(style={**CARD_STYLE,
                                "textAlign":"center","flex":"1",
                                "border":f"2px solid {GREEN}"}, children=[
                    html.Div("🟢 Decision Tree (max_depth=6)",
                             style={"fontSize":"16px","fontWeight":"700",
                                    "marginBottom":"8px"}),
                    html.Div(f"{dt_acc:.2f}%",
                             style={"fontSize":"40px","fontWeight":"900",
                                    "color":GREEN}),
                    html.Div("Accuracy on Test Set",
                             style={"color":"gray","fontSize":"12px"}),
                ]),
            ]),

            html.Div(style={"display":"grid",
                            "gridTemplateColumns":"1fr 1fr",
                            "gap":"16px"}, children=[
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="svm-cm",
                              style={"height":"320px"})]),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="dt-cm",
                              style={"height":"320px"})]),
            ]),

            html.Div(style={"display":"grid",
                            "gridTemplateColumns":"1fr 1fr",
                            "gap":"16px"}, children=[
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="feat-imp",
                              style={"height":"320px"})]),
                html.Div(style=CARD_STYLE, children=[
                    dcc.Graph(id="acc-compare",
                              style={"height":"320px"})]),
            ]),
        ])]),

    ]),  # end Tabs

    # FOOTER
    html.Div(style={
        "textAlign":"center","padding":"16px",
        "color":"rgba(255,255,255,0.3)","fontSize":"12px",
        "borderTop":f"1px solid {UN_ACCENT}","marginTop":"8px"
    }, children=[
        "TY Computer Science — Data Science Assignment | "
        "UN Chemicals & Waste Management Study Programme | Built with Python + Plotly Dash"
    ]),
])  # end layout


# ==============================================================================
#  CALLBACKS
# ==============================================================================

# ── TAB 1: AIR QUALITY ───────────────────────────────────────────────────────
@callback(
    Output("air-trend-line","figure"),
    Output("aqi-pie","figure"),
    Output("air-box","figure"),
    Output("air-hist","figure"),
    Input("city-dropdown","value"),
    Input("air-metric","value"),
)
def update_air(cities, metric):
    cities = cities or [df_air["City"].iloc[0]]
    metric = metric if metric in df_air.columns else "AQI"

    # Line — monthly average per city
    fig_line = go.Figure()
    colors_list = [UN_BLUE, GREEN, ORANGE, "#a29bfe", "#fd79a8"]
    for i, city in enumerate(cities):
        cdf = (df_air[df_air["City"]==city]
               .set_index("Date")
               .resample("ME")[metric].mean()
               .reset_index())
        fig_line.add_trace(go.Scatter(
            x=cdf["Date"], y=cdf[metric],
            name=city, mode="lines+markers",
            line=dict(width=2, color=colors_list[i % len(colors_list)]),
            marker=dict(size=4),
        ))
    if metric == "AQI":
        fig_line.add_hline(y=100, line_dash="dash",
                           line_color="red", opacity=0.6,
                           annotation_text="Unhealthy (100)")
    fig_line.update_layout(**base_layout(f"Monthly Average {metric} by City"))

    # Pie — AQI bucket distribution
    if "AQI_Bucket" in df_air.columns:
        fdf = df_air[df_air["City"].isin(cities)]
        bucket_counts = fdf["AQI_Bucket"].value_counts().reset_index()
        bucket_counts.columns = ["Category","Count"]
        fig_pie = px.pie(bucket_counts, names="Category", values="Count",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         hole=0.4)
        fig_pie.update_layout(**base_layout("AQI Category Distribution"))
        fig_pie.update_traces(textfont_color="white")
    else:
        fig_pie = go.Figure()
        fig_pie.update_layout(**base_layout("AQI Bucket not available"))

    # Box — distribution across cities
    fdf = df_air[df_air["City"].isin(cities)]
    fig_box = px.box(fdf, x="City", y=metric,
                     color="City",
                     color_discrete_sequence=colors_list)
    fig_box.update_layout(**base_layout(f"{metric} Distribution by City"))

    # Histogram
    fig_hist = px.histogram(fdf, x=metric, nbins=30,
                             color_discrete_sequence=[UN_BLUE])
    fig_hist.update_layout(**base_layout(f"{metric} Frequency Distribution"))

    return fig_line, fig_pie, fig_box, fig_hist


# ── TAB 2: WATER QUALITY ─────────────────────────────────────────────────────
@callback(
    Output("water-scatter","figure"),
    Output("water-donut","figure"),
    Output("water-heatmap","figure"),
    Input("water-x","value"),
    Input("water-y","value"),
)
def update_water(x_col, y_col):
    x_col = x_col or "ph"
    y_col = y_col or "Turbidity"

    # Scatter — coloured by potability
    fig_sc = px.scatter(
        df_water, x=x_col, y=y_col,
        color=target_col,
        color_discrete_map={0: RED, 1: GREEN},
        opacity=0.55, size_max=6,
        labels={target_col: "Potability (1=Safe)"},
    )
    if x_col == "ph":
        fig_sc.add_vrect(x0=6.5, x1=8.5,
                         fillcolor="green", opacity=0.07,
                         annotation_text="Safe pH")
    if y_col == "Turbidity":
        fig_sc.add_hline(y=5, line_dash="dot",
                         line_color="orange", opacity=0.7,
                         annotation_text="WHO Limit")
    fig_sc.update_layout(**base_layout(f"{x_col} vs {y_col} (by Potability)"))

    # Donut — safe vs unsafe
    counts = df_water[target_col].value_counts().reset_index()
    counts.columns = ["Potability","Count"]
    counts["Label"] = counts["Potability"].map({0:"Unsafe 🔴", 1:"Safe 🟢"})
    fig_donut = px.pie(counts, names="Label", values="Count",
                       color_discrete_sequence=[RED, GREEN], hole=0.5)
    fig_donut.update_layout(**base_layout("Water Potability Split"))
    fig_donut.update_traces(textfont_color="white")

    # Heatmap — correlation
    corr = df_water.corr(numeric_only=True).round(2)
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto",
                         color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1)
    fig_heat.update_layout(**base_layout("Feature Correlation Heatmap"))

    return fig_sc, fig_donut, fig_heat


# ── TAB 3: PLASTIC WASTE ─────────────────────────────────────────────────────
@callback(
    Output("plastic-bar","figure"),
    Output("plastic-treemap","figure"),
    Output("plastic-choropleth","figure"),
    Input("main-tabs","value"),
)
def update_plastic(tab):
    entity_col = next((c for c in df_plastic.columns
                       if any(k in c.lower() for k in
                              ["entity","country","nation"])), None)
    val_col    = next((c for c in df_plastic.columns
                       if df_plastic[c].dtype in [np.float64, np.int64]
                       and "year" not in c.lower()), None)

    if not entity_col or not val_col:
        empty = go.Figure()
        empty.update_layout(**base_layout("Column not found — check CSV"))
        return empty, empty, empty

    top = df_plastic.nlargest(15, val_col)

    fig_bar = px.bar(top.sort_values(val_col),
                     x=val_col, y=entity_col,
                     orientation="h",
                     color=val_col,
                     color_continuous_scale="Reds",
                     labels={val_col: val_col[:35], entity_col:"Country"})
    fig_bar.update_layout(**base_layout("Plastic Waste by Country"))

    fig_tree = px.treemap(top, path=[entity_col],
                          values=val_col,
                          color=val_col,
                          color_continuous_scale="Oranges")
    fig_tree.update_layout(**base_layout("Treemap — Plastic Waste Share"))

    fig_choro = px.choropleth(df_plastic,
                               locations=entity_col,
                               locationmode="country names",
                               color=val_col,
                               color_continuous_scale="Reds",
                               labels={val_col: val_col[:30]})
    fig_choro.update_layout(**base_layout("World Map — Plastic Waste Per Capita"))
    fig_choro.update_layout(
        geo=dict(bgcolor="rgba(0,0,0,0)",
                 lakecolor=UN_DARK,
                 landcolor="#2d3561",
                 showframe=False))

    return fig_bar, fig_tree, fig_choro


# ── TAB 4: CO2 EMISSIONS ─────────────────────────────────────────────────────
@callback(
    Output("co2-bar","figure"),
    Output("co2-choropleth","figure"),
    Input("main-tabs","value"),
)
def update_co2(tab):
    country_col = next((c for c in df_co2.columns
                        if any(k in c.lower() for k in
                               ["country","nation","area"])), None)
    val_col     = next((c for c in df_co2.columns
                        if df_co2[c].dtype in [np.float64, np.int64]
                        and "year" not in c.lower()), None)

    if not country_col or not val_col:
        empty = go.Figure()
        empty.update_layout(**base_layout("Column not found — check CSV"))
        return empty, empty

    agg = (df_co2.groupby(country_col)[val_col]
           .sum().reset_index()
           .nlargest(15, val_col))

    fig_bar = px.bar(agg.sort_values(val_col),
                     x=val_col, y=country_col,
                     orientation="h",
                     color=val_col,
                     color_continuous_scale="Oranges",
                     labels={val_col:"CO₂ (thousand metric tons)",
                             country_col:"Country"})
    fig_bar.update_layout(**base_layout("Top 15 CO₂ Emitting Countries"))

    fig_map = px.choropleth(agg,
                             locations=country_col,
                             locationmode="country names",
                             color=val_col,
                             color_continuous_scale="YlOrRd",
                             labels={val_col:"CO₂ Emissions"})
    fig_map.update_layout(**base_layout("CO₂ Emissions World Map"))
    fig_map.update_layout(
        geo=dict(bgcolor="rgba(0,0,0,0)",
                 lakecolor=UN_DARK,
                 landcolor="#2d3561",
                 showframe=False))

    return fig_bar, fig_map


# ── TAB 5: NUCLEAR & ENERGY ──────────────────────────────────────────────────
@callback(
    Output("energy-line","figure"),
    Output("death-bar","figure"),
    Input("main-tabs","value"),
)
def update_energy(tab):
    year_col = next((c for c in df_nuclear.columns
                     if "year" in c.lower()), None)
    num_cols = [c for c in df_nuclear.select_dtypes(include=np.number).columns
                if c != year_col]

    fig_line = go.Figure()
    energy_colors = [UN_BLUE, GREEN, ORANGE, "#a29bfe", "#fd79a8"]
    if year_col and num_cols:
        for i, col in enumerate(num_cols[:5]):
            fig_line.add_trace(go.Scatter(
                x=df_nuclear[year_col], y=df_nuclear[col],
                name=col[:25], mode="lines+markers",
                line=dict(width=2.5,
                          color=energy_colors[i % len(energy_colors)]),
                marker=dict(size=5),
            ))
    fig_line.update_layout(**base_layout("Energy Production Trends Over Years"))

    # Death rates bar
    death_colors = [RED if d > 5 else ORANGE if d > 1 else GREEN
                    for d in df_death["Deaths_per_TWh"]]
    fig_death = go.Figure(go.Bar(
        x=df_death["Deaths_per_TWh"],
        y=df_death["Source"],
        orientation="h",
        marker=dict(color=death_colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{d:.2f}" for d in df_death["Deaths_per_TWh"]],
        textposition="outside",
        textfont=dict(color=TEXT_COL),
    ))
    fig_death.add_vline(x=1, line_dash="dash",
                        line_color="orange", opacity=0.7,
                        annotation_text="1 death/TWh")
    fig_death.update_layout(**base_layout("Deaths per TWh by Energy Source"))
    fig_death.update_xaxes(type="log", title="Deaths per TWh (log scale)")

    return fig_line, fig_death


# ── TAB 6: ML MODELS ─────────────────────────────────────────────────────────
@callback(
    Output("svm-cm","figure"),
    Output("dt-cm","figure"),
    Output("feat-imp","figure"),
    Output("acc-compare","figure"),
    Input("main-tabs","value"),
)
def update_ml(tab):
    labels = ["Unsafe","Safe"]

    # SVM Confusion Matrix
    fig_svm = px.imshow(svm_cm, text_auto=True,
                        x=labels, y=labels,
                        color_continuous_scale="Blues",
                        aspect="auto")
    fig_svm.update_layout(**base_layout(f"SVM Confusion Matrix ({svm_acc:.1f}%)"))
    fig_svm.update_xaxes(title="Predicted")
    fig_svm.update_yaxes(title="Actual")

    # DT Confusion Matrix
    fig_dt = px.imshow(dt_cm, text_auto=True,
                       x=labels, y=labels,
                       color_continuous_scale="Greens",
                       aspect="auto")
    fig_dt.update_layout(**base_layout(f"Decision Tree Confusion Matrix ({dt_acc:.1f}%)"))
    fig_dt.update_xaxes(title="Predicted")
    fig_dt.update_yaxes(title="Actual")

    # Feature Importance
    fig_feat = go.Figure(go.Bar(
        x=feat_imp.values, y=feat_imp.index,
        orientation="h",
        marker=dict(
            color=feat_imp.values,
            colorscale="Viridis",
            line=dict(color="rgba(0,0,0,0)"),
        ),
        text=[f"{v:.3f}" for v in feat_imp.values],
        textposition="outside",
        textfont=dict(color=TEXT_COL),
    ))
    fig_feat.update_layout(**base_layout("Decision Tree — Feature Importance"))
    fig_feat.update_xaxes(title="Importance Score")

    # Accuracy comparison
    fig_acc = go.Figure(go.Bar(
        x=["SVM\n(RBF Kernel)", "Decision Tree\n(max_depth=6)"],
        y=[svm_acc, dt_acc],
        marker=dict(color=[UN_BLUE, GREEN],
                    line=dict(color="rgba(0,0,0,0)")),
        text=[f"{svm_acc:.2f}%", f"{dt_acc:.2f}%"],
        textposition="outside",
        textfont=dict(color=TEXT_COL, size=14),
        width=0.4,
    ))
    fig_acc.add_hline(y=80, line_dash="dash",
                      line_color="red", opacity=0.6,
                      annotation_text="80% Benchmark")
    fig_acc.update_layout(**base_layout("Model Accuracy Comparison"))
    fig_acc.update_yaxes(range=[0, 110], title="Accuracy (%)")

    return fig_svm, fig_dt, fig_feat, fig_acc
server = app.server

# ==============================================================================
#  RUN
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🌍 UN Waste Management Dashboard")
    print("=" * 55 + "\n")
    app.run(debug=False)