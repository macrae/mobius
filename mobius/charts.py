import plotly.express as px


def plot_3d(df, x: str, y: str, z: str, c: str, symbol: str, opacity: float, save_path: str):
    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        color=c,
        symbol=symbol,
        opacity=opacity,
        color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.write_html(save_path)
