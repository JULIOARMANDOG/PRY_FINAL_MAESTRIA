import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import plotly.figure_factory as ff
import numpy as np

# Simulación de métricas para modelos
model_scores = {
    "RandomForest": {"precision": 0.91, "recall": 0.88, "f1": 0.89, "conf_matrix": [[50, 2], [4, 44]]},
    "SVM": {"precision": 0.85, "recall": 0.83, "f1": 0.84, "conf_matrix": [[48, 4], [5, 43]]},
    "KNN": {"precision": 0.80, "recall": 0.78, "f1": 0.79, "conf_matrix": [[45, 7], [6, 42]]},
}

# Datos tabulados
df_scores = pd.DataFrame([
    {"Modelo": name, "Precisión": data["precision"], "Recall": data["recall"], "F1-score": data["f1"]}
    for name, data in model_scores.items()
])

# Crear app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Comparador de Modelos ML"

app.layout = dbc.Container([
    html.H2("Dashboard de Comparación de Modelos de Machine Learning", className="my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Selecciona un modelo:"),
            dcc.Dropdown(
                id='modelo-dropdown',
                options=[{"label": name, "value": name} for name in model_scores.keys()],
                value="RandomForest",
                clearable=False
            ),
        ], md=4),
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='bar-metricas'),
        ], md=6),
        dbc.Col([
            dcc.Graph(id='matriz-confusion'),
        ], md=6),
    ]),

    html.Hr(),

    html.H4("Resumen de métricas"),
    dbc.Table.from_dataframe(df_scores.round(3), striped=True, bordered=True, hover=True),
], fluid=True)


@app.callback(
    Output('bar-metricas', 'figure'),
    Output('matriz-confusion', 'figure'),
    Input('modelo-dropdown', 'value')
)
def actualizar_graficas(modelo):
    datos = model_scores[modelo]
    
    # Gráfica de barras (métricas)
    fig_bar = px.bar(
        x=["Precisión", "Recall", "F1-score"],
        y=[datos["precision"], datos["recall"], datos["f1"]],
        labels={"x": "Métrica", "y": "Valor"},
        title=f"Métricas del modelo: {modelo}",
        text_auto=".2f"
    )
    
    # Matriz de confusión
    cm = np.array(datos["conf_matrix"])
    labels = ["Clase 0", "Clase 1"]
    fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=labels,
    y=labels,
    colorscale="Viridis",
    showscale=True,
    annotation_text=cm.astype(str)
    ) 
    fig_cm.update_layout(title_text=f"Matriz de Confusión: {modelo}")

    return fig_bar, fig_cm


if __name__ == '__main__':
    app.run(debug=True)