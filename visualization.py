import numpy as np
import plotly.graph_objects as go


def plot_swarm_paths(terrain, paths, output_plot_path):
    """
    Plot the terrain surface and UAV paths in 3D using Plotly.
    terrain: Terrain object.
    paths: list of list of (x,y,z) points for each UAV path.
    """
    # Generate terrain surface mesh for plotting
    x_vals = np.linspace(terrain.x_min, terrain.x_max, 41)
    y_vals = np.linspace(terrain.y_min, terrain.y_max, 41)
    z_matrix = []
    for yi in y_vals:
        row = []
        for xi in x_vals:
            row.append(terrain.get_height(xi, yi))
        z_matrix.append(row)
    # Create Plotly figure
    fig = go.Figure()
    # Add terrain surface
    fig.add_trace(
        go.Surface(
            x=x_vals,
            y=y_vals,
            z=z_matrix,
            colorscale="Viridis",
            opacity=0.5,
            name="Terrain",
        )
    )
    # Add each UAV path as a line
    for idx, path in enumerate(paths):
        xs = [pt[0] for pt in path]
        ys = [pt[1] for pt in path]
        zs = [pt[2] for pt in path]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=6),
                name=f"UAV{idx+1}",
            )
        )
        # Mark start and end points specially
        fig.add_trace(
            go.Scatter3d(
                x=[xs[0]],
                y=[ys[0]],
                z=[zs[0]],
                mode="markers",
                marker=dict(size=6, symbol="circle", color="green"),
                name=f"UAV{idx+1} Start",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[xs[-1]],
                y=[ys[-1]],
                z=[zs[-1]],
                mode="markers",
                marker=dict(size=6, symbol="x", color="red"),
                name=f"UAV{idx+1} End",
            )
        )
    # Set titles and axes labels
    fig.update_layout(
        title="UAV Swarm Paths over Terrain",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z (Altitude)"),
    )
    fig.write_html(
        output_plot_path,
        auto_open=False,
        include_plotlyjs="cdn",
        config={"responsive": True},
    )
