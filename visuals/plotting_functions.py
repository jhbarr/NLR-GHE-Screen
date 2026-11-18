import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import folium
import networkx as nx
import geopandas as gpd

# ---------------------------------------------------------------------------
# MILP Plotting Functions
# ---------------------------------------------------------------------------

def plot_timeseries_loads(
        building_loads: list[float],
        building_types: list[str],
        title: str,
        x_axis: str,
        y_axis: str,
        show_legend: bool = False
):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors  
    for i, (load, type) in enumerate(zip(building_loads, building_types)):
        ax.plot(
            range(len(load)),
            load,
            color=colors[i % len(colors)],
            label=type
        )

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(title)

    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
        plt.tight_layout()
    plt.show()


def plot_pareto_points(points, title="Pareto Frontier"):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
    })

    
    peak    = [p[0] for p in points]
    imbalance = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Frontier line + markers
    ax.plot(
        peak, imbalance,
        color='#4E79A7', linewidth=1.6, alpha=0.9,
        marker='o', markersize=8, markerfacecolor='#4E79A7',
        markeredgecolor='white', markeredgewidth=1.2,
        label='Pareto point',
        zorder=5,
    )

    # Label each point with its objective value
    for x, y in zip(peak, imbalance):
        ax.annotate(
            f"Peak: {x:.2f}, Imbalance: {y:.2f}",
            xy=(x, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            color='#000080',
            fontweight='semibold',
        )

    # Remove the border (all four spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Subtle horizontal grid instead of a box
    ax.grid(axis='y', color='#E5E5E5', linewidth=0.8)
    ax.set_axisbelow(True)

    # Remove tick marks but keep the labels
    ax.tick_params(length=0)

    # Padding so labels don't clip
    ax.margins(x=0.08, y=0.15)

    # Legend without a frame, outside the right edge
    ax.legend(
        frameon=False,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
    )

    ax.set_xlabel("Peak")
    ax.set_ylabel("Energy Imbalance")
    plt.title(title)

    fig.tight_layout()
    plt.show()




# ---------------------------------------------------------------------------
# Network Plotting Functions
# ---------------------------------------------------------------------------

def plot_network_graph(
    network: nx.MultiDiGraph, 
    extra_layers: list
):
    edge_lines = []
    for u, v, data in network.edges(data=True):
        if 'geometry' in data:
            geom = data['geometry']
        else:
            # Build a LineString from node coordinates
            x1, y1 = network.nodes[u]['x'], network.nodes[u]['y']
            x2, y2 = network.nodes[v]['x'], network.nodes[v]['y']
            geom = LineString([(x1, y1), (x2, y2)])
        edge_lines.append({'u': u, 'v': v, 'geometry': geom})

    gdf_edges = gpd.GeoDataFrame(edge_lines, crs=network.graph['crs'])

    # Convert NetworkX nodes to a GeoDataFrame of Points
    node_points = []
    for n, data in network.nodes(data=True):
        node_points.append(
            {'node': n, 'geometry': Point(data['x'], data['y']), **data}
        )

    gdf_nodes = gpd.GeoDataFrame(node_points, crs=network.graph['crs'])

    # 3. Create the base interactive map using GeoPandas .explore()
    # m = base_gdf.explore(color='green', name='Base Map')

    m = gdf_edges.explore(color='blue', weight=2, name='Network Edges')
    m = gdf_nodes.explore(m=m, color='red', marker_kwds={'radius': 4}, name='Network Nodes')

    for name, gdf, color in (extra_layers or []):
        m = gdf.explore(m=m, color=color, name=name)

    # Fit to everything combined, since each explore() call re-fits to its own data
    all_bounds = [gdf_edges.to_crs(4326).total_bounds]
    all_bounds += [g.to_crs(4326).total_bounds for _, g, _ in (extra_layers or [])]
    minx = min(b[0] for b in all_bounds); miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds); maxy = max(b[3] for b in all_bounds)
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    folium.LayerControl().add_to(m)
    return m